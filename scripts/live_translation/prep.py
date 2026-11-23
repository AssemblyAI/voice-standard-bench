#!/usr/bin/env python3
"""Build the live_translation dataset: normalized/live_translation/fleurs.jsonl.

FLEURS (google/fleurs, CC BY 4.0) has the same sentences read aloud in 100+ languages, so every clip
comes with a professional translation in every other language. This script:

1. Picks N sentence ids that exist in every language used, from the dev split.
2. Joins each source language's clips into ~2 minute streams, with a pause between sentences.
3. Streams each stream once through the notetaker model (assemblyai-stt-3-7-realtime) at real-time
   speed, and gives each clip the words the model returned for it: one record turn per clip.
4. Runs every clip through async transcription with speech understanding translation, as the bar
   the streaming prompt has to reach (the async-su-translation row in the report).
5. Writes one record per (clip, target language), with the previous 3 turns of its stream as context.

Run from the repo root (Python 3.9+; needs `pip install websockets`, and ASSEMBLYAI_API_KEY):
    python3 scripts/live_translation/prep.py
Downloads are about 1 GB of FLEURS audio, cached under scripts/live_translation/.cache. Streaming
runs every stream at once, at real-time speed: about 3 minutes. Re-running reuses everything cached.
"""
import argparse
import array
import asyncio
import concurrent.futures
import csv
import hashlib
import io
import json
import os
import struct
import sys
import tarfile
import time
import urllib.error
import urllib.request
import wave
from pathlib import Path
from urllib.parse import urlencode

ROOT = Path(__file__).resolve().parents[2]
CACHE = Path(__file__).resolve().parent / ".cache"
OUT = ROOT / "normalized" / "live_translation" / "fleurs.jsonl"
HF = "https://huggingface.co/datasets/google/fleurs/resolve/main/data"
API = "https://api.assemblyai.com"
WS = "wss://streaming.assemblyai.com/v3/ws"
RATE = 16000

# FLEURS config -> ISO 639-1 code used by the API and the benchmark.
FLEURS = {"en": "en_us", "es": "es_419", "fr": "fr_fr", "de": "de_de", "pt": "pt_br", "ja": "ja_jp",
          "it": "it_it", "hi": "hi_in", "zh": "cmn_hans_cn", "ko": "ko_kr", "nl": "nl_nl", "pl": "pl_pl"}

# source -> targets. English out to the main languages, and the main languages back into English.
DEFAULT_PAIRS = {"en": ["es", "fr", "de", "pt", "ja"], "es": ["en"], "fr": ["en"], "de": ["en"], "pt": ["en"]}


def fetch(url, dest, tries=5):
    """Download url to dest once, retrying a busy server (429/5xx) with a growing wait."""
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  downloading {url.split('/data/')[-1] if '/data/' in url else url.rsplit('/', 1)[-1]}")
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "voice-standard-bench"})
            with urllib.request.urlopen(req, timeout=600) as r, open(tmp, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            break
        except urllib.error.HTTPError as e:
            if e.code != 429 and e.code < 500 or attempt == tries - 1:
                raise
            wait = 10 * 2 ** attempt
            print(f"  {e.code} from the server, retrying in {wait}s")
            time.sleep(wait)
    tmp.rename(dest)
    return dest


def read_tsv(lang, split):
    """FLEURS tsv: id, file_name, raw_transcription, transcription, phonemes, num_samples, gender."""
    path = fetch(f"{HF}/{FLEURS[lang]}/{split}.tsv", CACHE / FLEURS[lang] / f"{split}.tsv")
    rows = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            if len(row) >= 3 and row[0] not in rows:  # several speakers read each sentence; keep the first
                rows[row[0]] = {"file": row[1], "text": row[2]}
    return rows


def read_clips(lang, split, files):
    """Pull just the wanted wav files out of the language's audio tarball, as 16 kHz mono PCM16."""
    out_dir = CACHE / FLEURS[lang] / split
    missing = [f for f in files if not (out_dir / f).exists()]
    if missing:
        tar = fetch(f"{HF}/{FLEURS[lang]}/audio/{split}.tar.gz", CACHE / FLEURS[lang] / f"{split}.tar.gz")
        want = set(missing)
        out_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tar, "r:gz") as t:
            for m in t:
                name = os.path.basename(m.name)
                if m.isfile() and name in want:
                    (out_dir / name).write_bytes(t.extractfile(m).read())
                    want.discard(name)
                    if not want:
                        break
        if want:
            sys.exit(f"{lang}: {len(want)} clips not found in the tarball")
    return {f: pcm16_mono_16k((out_dir / f).read_bytes()) for f in files}


def read_wav(wav_bytes):
    """(channels, rate, int16 samples) from a WAV of 16-bit PCM or 32-bit float (FLEURS uses float),
    which the wave module can't read."""
    if wav_bytes[:4] != b"RIFF" or wav_bytes[8:12] != b"WAVE":
        sys.exit("not a WAV file")
    pos, fmt, data = 12, None, None
    while pos + 8 <= len(wav_bytes):
        cid, size = wav_bytes[pos:pos + 4], struct.unpack("<I", wav_bytes[pos + 4:pos + 8])[0]
        body = wav_bytes[pos + 8:pos + 8 + size]
        if cid == b"fmt ":
            tag, ch, rate, _, _, bits = struct.unpack("<HHIIHH", body[:16])
            if tag == 0xFFFE and len(body) >= 26:  # WAVE_FORMAT_EXTENSIBLE: real tag is the subformat GUID's start
                tag = struct.unpack("<H", body[24:26])[0]
            fmt = (tag, ch, rate, bits)
        elif cid == b"data":
            data = body
        pos += 8 + size + (size & 1)
    if not fmt or data is None:
        sys.exit("WAV without fmt or data chunk")
    tag, ch, rate, bits = fmt
    if tag == 1 and bits == 16:
        return ch, rate, array.array("h", data[:len(data) // 2 * 2])
    if tag == 3 and bits == 32:
        f = array.array("f", data[:len(data) // 4 * 4])
        return ch, rate, array.array("h", (max(-32768, min(32767, int(x * 32767))) for x in f))
    sys.exit(f"unsupported WAV format tag {tag}, {bits} bits")


def pcm16_mono_16k(wav_bytes):
    ch, rate, a = read_wav(wav_bytes)
    if ch > 1:
        a = array.array("h", (sum(a[i:i + ch]) // ch for i in range(0, len(a), ch)))
    if rate != RATE:
        n = int(len(a) * RATE / rate)
        a = array.array("h", (a[min(int(i * rate / RATE), len(a) - 1)] for i in range(n)))
    return a.tobytes()


def wav_bytes(pcm):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(pcm)
    return buf.getvalue()


def build_streams(lang, ids, clips, tsv, per_stream, gap_ms):
    """Join clips into streams; remember where each clip starts and ends, in ms."""
    gap = b"\x00\x00" * (RATE * gap_ms // 1000)
    streams = []
    for s in range(0, len(ids), per_stream):
        pcm, spans = bytearray(gap), []
        for sid in ids[s:s + per_stream]:
            start = len(pcm) // 2 * 1000 // RATE
            pcm += clips[tsv[sid]["file"]]
            spans.append({"id": sid, "start": start, "end": len(pcm) // 2 * 1000 // RATE})
            pcm += gap
        streams.append({"name": f"{lang}-s{s // per_stream:02d}", "pcm": bytes(pcm), "spans": spans})
    return streams


async def stream_once(key, stream, lang, model, mode):
    """Stream at real-time speed; return the final turns with their words."""
    import websockets

    cache = CACHE / "streaming" / model / (mode or "default") / f"{stream['name']}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    q = {"sample_rate": RATE, "encoding": "pcm_s16le", "speech_model": model, "language_codes": json.dumps([lang])}
    if mode:
        q["mode"] = mode
    major = int(str(getattr(websockets, "__version__", "0")).split(".")[0] or 0)
    hdr = "additional_headers" if major >= 14 else "extra_headers"  # renamed in websockets 14
    turns = {}
    pcm, step = stream["pcm"], RATE * 2 // 20  # 50 ms chunks
    async with websockets.connect(f"{WS}?{urlencode(q)}", max_size=None, **{hdr: {"Authorization": key}}) as ws:
        async def send():
            t0 = time.monotonic()
            for i in range(0, len(pcm), step):
                await ws.send(pcm[i:i + step])
                wait = t0 + (i // step + 1) * 0.05 - time.monotonic()
                if wait > 0:
                    await asyncio.sleep(wait)
            await asyncio.sleep(1.5)
            await ws.send(json.dumps({"type": "Terminate"}))

        async def recv():
            async for msg in ws:
                if isinstance(msg, bytes):
                    continue
                m = json.loads(msg)
                if m.get("type") == "Turn" and m.get("end_of_turn"):
                    turns[m.get("turn_order", len(turns))] = {"transcript": m.get("transcript", ""), "words": m.get("words") or []}
                elif m.get("type") == "Termination":
                    break
                elif m.get("type") == "Error" or "error" in m:
                    raise RuntimeError(f"{stream['name']}: {str(m)[:300]}")

        await asyncio.gather(send(), recv())
    result = [turns[k] for k in sorted(turns)]
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(result, ensure_ascii=False))
    return result


def turns_per_clip(turns, spans):
    """Give each clip the speech the model returned during it. A turn that falls inside one clip
    keeps its formatted transcript; one that spans clips is split by word timestamps."""
    bounds = []
    for i, s in enumerate(spans):  # a clip owns half of the pause on each side
        lo = 0 if i == 0 else (spans[i - 1]["end"] + s["start"]) / 2
        hi = float("inf") if i == len(spans) - 1 else (s["end"] + spans[i + 1]["start"]) / 2
        bounds.append((lo, hi))

    def clip_of(ms):
        return next(i for i, (lo, hi) in enumerate(bounds) if lo <= ms < hi)

    parts = [[] for _ in spans]
    for t in turns:
        words = [w for w in t["words"] if w.get("text")]
        if not words:
            continue
        owners = [clip_of((w["start"] + w["end"]) / 2) for w in words]
        if len(set(owners)) == 1:
            parts[owners[0]].append(t["transcript"] or " ".join(w["text"] for w in words))
        else:
            for i in sorted(set(owners)):
                parts[i].append(" ".join(w["text"] for w, o in zip(words, owners) if o == i))
    return [" ".join(p).strip() for p in parts]


def http(method, url, key, body=None, data=None):
    headers = {"authorization": key}
    if body is not None:
        data = json.dumps(body).encode()
        headers["content-type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < 4:
                time.sleep(5 * (attempt + 1))
                continue
            raise RuntimeError(f"{method} {url}: {e.code} {e.read().decode(errors='replace')[:300]}")


def async_translate(key, lang, sid, pcm, targets):
    """Async transcription with speech understanding translation of one clip."""
    cache = CACHE / "async" / lang / f"{sid}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    url = http("POST", f"{API}/v2/upload", key, data=wav_bytes(pcm))["upload_url"]
    job = http("POST", f"{API}/v2/transcript", key, body={
        "audio_url": url,
        "speech_models": ["assemblyai-stt-3-7"],
        "language_code": lang,
        "speech_understanding": {"request": {"translation": {"target_languages": targets}}},
    })
    while True:
        t = http("GET", f"{API}/v2/transcript/{job['id']}", key)
        if t["status"] == "completed":
            break
        if t["status"] == "error":
            raise RuntimeError(f"async {lang} {sid}: {t.get('error')}")
        time.sleep(3)
    result = {"text": t.get("text") or "", "translated_texts": t.get("translated_texts") or {}}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(result, ensure_ascii=False))
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pairs", default=json.dumps(DEFAULT_PAIRS), help="JSON {source: [targets]}")
    ap.add_argument("--clips", type=int, default=40, help="sentences per source language")
    ap.add_argument("--per-stream", type=int, default=8, help="sentences per stream (about 15s each)")
    ap.add_argument("--gap-ms", type=int, default=1000, help="pause between sentences in a stream")
    ap.add_argument("--split", default="dev")
    ap.add_argument("--streaming-model", default="assemblyai-stt-3-7-realtime")
    ap.add_argument("--history", type=int, default=3, help="previous turns given as context")
    ap.add_argument("--mode", default="", help="streaming mode (min_latency, balanced, max_accuracy); default: the model's own. "
                    "A set mode writes its own file and its own report buckets, e.g. en-es/max_accuracy")
    args = ap.parse_args()

    key = os.environ.get("ASSEMBLYAI_API_KEY", "")
    if not key:
        sys.exit("Set ASSEMBLYAI_API_KEY first.")
    try:
        import websockets  # noqa: F401
    except ImportError:
        sys.exit("pip3 install websockets")

    pairs = json.loads(args.pairs)
    langs = sorted(set(pairs) | {t for ts in pairs.values() for t in ts})
    unknown = [l for l in langs if l not in FLEURS]
    if unknown:
        sys.exit(f"no FLEURS config for {unknown}; add it to FLEURS")

    print("reading FLEURS transcripts")
    tsv = {l: read_tsv(l, args.split) for l in langs}
    common = set.intersection(*(set(t) for t in tsv.values()))
    ids = sorted(common, key=lambda i: hashlib.md5(i.encode()).hexdigest())[:args.clips]
    print(f"  {len(common)} sentences in every language; using {len(ids)}")

    streams = {}
    for src in pairs:
        print(f"preparing {src} audio")
        clips = read_clips(src, args.split, [tsv[src][i]["file"] for i in ids])
        streams[src] = build_streams(src, ids, clips, tsv[src], args.per_stream, args.gap_ms)

    print(f"streaming {sum(len(s) for s in streams.values())} streams through {args.streaming_model} at real-time speed")

    async def run_all():
        jobs = [(src, s) for src in streams for s in streams[src]]
        res = await asyncio.gather(*(stream_once(key, s, src, args.streaming_model, args.mode) for src, s in jobs), return_exceptions=True)
        failed = [(s["name"], r) for (_, s), r in zip(jobs, res) if isinstance(r, Exception)]
        for name, err in failed:
            print(f"  FAILED {name}: {err}")
        if failed:
            sys.exit("streaming failed; re-run to retry (finished streams are cached)")
        return {s["name"]: r for (_, s), r in zip(jobs, res)}

    turns = asyncio.run(run_all())

    print("async speech understanding translation, per clip")
    async_out = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        futs = {}
        for src in streams:
            for s in streams[src]:
                for span in s["spans"]:
                    pcm = s["pcm"][span["start"] * RATE // 1000 * 2: span["end"] * RATE // 1000 * 2]
                    futs[ex.submit(async_translate, key, src, span["id"], pcm, pairs[src])] = (src, span["id"])
        for f in concurrent.futures.as_completed(futs):
            async_out[futs[f]] = f.result()

    out_path = OUT.with_name(f"fleurs-{args.mode}.jsonl") if args.mode else OUT
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = empty = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for src in streams:
            for s in streams[src]:
                texts = turns_per_clip(turns[s["name"]], s["spans"])
                for i, span in enumerate(s["spans"]):
                    empty += not texts[i]
                    for tgt in pairs[src]:
                        out.write(json.dumps({
                            "id": f"{s['name']}-c{i:02d}-{tgt}" + (f"-{args.mode}" if args.mode else ""),
                            "dataset": f"fleurs/{args.split}",
                            "input": {
                                "source_language": src,
                                "target_language": tgt,
                                "turn": texts[i],
                                "history": [h for h in texts[max(0, i - args.history):i] if h],
                                "source_truth": tsv[src][span["id"]]["text"],
                                "fleurs_id": span["id"],
                                "streaming_model": args.streaming_model,
                                "variant": args.mode,
                            },
                            "expected": {"reference": tsv[tgt][span["id"]]["text"]},
                            "baselines": {"async_su_translation": async_out[(src, span["id"])]["translated_texts"].get(tgt, "")},
                        }, ensure_ascii=False) + "\n")
                        n += 1
    print(f"wrote {n} records to {out_path.relative_to(ROOT)} ({empty} clips got no words from the streaming model)")


if __name__ == "__main__":
    main()
