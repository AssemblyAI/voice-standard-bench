#!/usr/bin/env python3
"""End-to-end check of streaming translation through the real llm_gateway parameter.

The live_translation task benchmarks the LLM step on cached transcripts, one sentence at a time. This
script runs the product path instead: the same FLEURS streams as prep.py go through
wss://streaming.assemblyai.com/v3/ws with llm_gateway set, one session per stream and target language,
and every LLMGatewayResponse is kept. It then scores whole streams, which needs no turn-to-sentence
alignment:

- e2e:         the streaming translations of a stream, joined in turn order
- async_full:  async speech understanding translation of the whole stream file (full context)

Both are scored against the stream's joined FLEURS references with chrF and an LLM judge (1-5). Latency
is the time from receiving a final Turn to receiving its translation.

Run from the repo root after prep.py (it reuses prep.py's FLEURS cache and sentence choice):
    python3 scripts/live_translation/e2e.py                           # interpreter_v1
    python3 scripts/live_translation/e2e.py --prompt docs_oneliner
    python3 scripts/live_translation/e2e.py --model qwen3.5-4b-32k-fast
    python3 scripts/live_translation/e2e.py --dataset fleurs_hard   # no pauses, another voice underneath
    python3 scripts/live_translation/e2e.py --dataset ami           # real meetings, judged against the human transcript
Needs ASSEMBLYAI_API_KEY and websockets. About 3 minutes of streaming per prompt and model; results
are cached under scripts/live_translation/.cache/e2e and summarised in results/live_translation_e2e/.
"""
import argparse
import asyncio
import concurrent.futures
import hashlib
import json
import os
import re
import statistics as st
import sys
import time
import unicodedata
from pathlib import Path
from urllib.parse import urlencode

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prep  # noqa: E402
import prep_ami  # noqa: E402

CACHE = prep.CACHE / "e2e"
OUT = prep.ROOT / "results" / "live_translation_e2e"
LLM = "https://llm-gateway.assemblyai.com/v1/chat/completions"

NAMES = {"en": "English", "es": "Spanish", "fr": "French", "de": "German", "pt": "Portuguese", "ja": "Japanese",
         "it": "Italian", "hi": "Hindi", "zh": "Chinese", "ko": "Korean", "nl": "Dutch", "pl": "Polish"}

# The same prompts as src/prompts/live_translation.go, as llm_gateway templates.
INTERPRETER = ("You are a simultaneous interpreter translating a live conversation, one speaker turn at a time.\n\n"
               "The user message gives the target language, the previous turns for context only (already translated, "
               "do not repeat them), and the current turn.\n\n"
               "Translate ONLY the current turn into natural, fluent text in the target language. Keep names, numbers "
               "and units as spoken. Do not add explanations, quotes, or any text other than the translation. If the "
               "turn is only a filler such as \"um\" or \"okay\", translate it to the closest natural equivalent.")
PROMPTS = {
    "interpreter_v1": lambda s, t: [
        {"role": "system", "content": INTERPRETER},
        {"role": "user", "content": f"Target language: {t}\n\nPrevious turns (context only):\n{{{{history_3}}}}\n\nCurrent turn:\n{{{{turn}}}}"}],
    "docs_oneliner": lambda s, t: [
        {"role": "user", "content": f"Translate the following {s} into natural {t} and reply with only the translation: {{{{turn}}}}"}],
    "docs_history": lambda s, t: [
        {"role": "user", "content": f"Previous turns, for context only:\n{{{{history_3}}}}\n\nTranslate the following {s} into natural {t} and reply with only the translation: {{{{turn}}}}"}],
}


async def stream_translate(key, stream, src, tgt, args):
    import websockets

    base = CACHE if args.dataset == "fleurs" else CACHE / args.dataset  # fleurs keeps the cache of earlier runs
    cache = base / args.prompt / args.model / f"{stream['name']}-{tgt}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    gw = {"model": args.model, "max_tokens": 400, "messages": PROMPTS[args.prompt](NAMES[src], NAMES[tgt])}
    q = {"sample_rate": prep.RATE, "encoding": "pcm_s16le", "speech_model": args.streaming_model,
         "language_codes": json.dumps([src]), "llm_gateway": json.dumps(gw)}
    major = int(str(getattr(websockets, "__version__", "0")).split(".")[0] or 0)
    hdr = "additional_headers" if major >= 14 else "extra_headers"
    turns, trans, turn_at, trans_at, errors = {}, {}, {}, {}, []
    pcm, step = stream["pcm"], prep.RATE * 2 // 20
    async with websockets.connect(f"{prep.WS}?{urlencode(q)}", max_size=None, **{hdr: {"Authorization": key}}) as ws:
        async def send():
            t0 = time.monotonic()
            for i in range(0, len(pcm), step):
                await ws.send(pcm[i:i + step])
                wait = t0 + (i // step + 1) * 0.05 - time.monotonic()
                if wait > 0:
                    await asyncio.sleep(wait)
            await asyncio.sleep(4)  # room for the last turn's translation
            await ws.send(json.dumps({"type": "Terminate"}))

        async def recv():
            async for msg in ws:
                if isinstance(msg, bytes):
                    continue
                m = json.loads(msg)
                if m.get("type") == "Turn" and m.get("end_of_turn"):
                    k = m.get("turn_order")
                    turns[k] = m.get("transcript", "")
                    turn_at[k] = time.monotonic()
                elif m.get("type") == "LLMGatewayResponse":
                    k = m.get("turn_order")
                    try:
                        trans[k] = m["data"]["choices"][0]["message"]["content"] or ""
                    except (KeyError, IndexError, TypeError):
                        errors.append(str(m)[:200])
                        continue
                    trans_at[k] = time.monotonic()
                elif m.get("type") == "Termination":
                    break
                elif m.get("type") == "Error" or "error" in m:
                    raise RuntimeError(f"{stream['name']}-{tgt}: {str(m)[:300]}")

        await asyncio.gather(send(), recv())
    order = sorted(turns)
    result = {
        "turns": [turns[k] for k in order],
        "translations": [trans.get(k, "") for k in order],
        "missing": [k for k in order if turns[k].strip() and k not in trans],
        "latency_s": [round(trans_at[k] - turn_at[k], 3) for k in order if k in trans_at and k in turn_at],
        "errors": errors,
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(result, ensure_ascii=False))
    return result


def async_full(key, stream, src, targets):
    """Async speech understanding translation of the whole stream file."""
    cache = CACHE / "async_full" / f"{stream['name']}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    url = prep.http("POST", f"{prep.API}/v2/upload", key, data=prep.wav_bytes(stream["pcm"]))["upload_url"]
    job = prep.http("POST", f"{prep.API}/v2/transcript", key, body={
        "audio_url": url, "speech_models": ["assemblyai-stt-3-7"], "language_code": src,
        "speech_understanding": {"request": {"translation": {"target_languages": targets}}}})
    while True:
        t = prep.http("GET", f"{prep.API}/v2/transcript/{job['id']}", key)
        if t["status"] == "completed":
            break
        if t["status"] == "error":
            raise RuntimeError(f"async {stream['name']}: {t.get('error')}")
        time.sleep(3)
    result = t.get("translated_texts") or {}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(result, ensure_ascii=False))
    return result


def chrf(hyp, ref, max_n=6, beta=2.0):
    """Same as ChrF in src/live_translation.go: character n-grams 1-6, whitespace removed, beta 2."""
    h = [c for c in hyp if not c.isspace()]
    r = [c for c in ref if not c.isspace()]
    if not r:
        return 100.0 if not h else 0.0
    ps = rs = 0.0
    counted = 0
    for n in range(1, max_n + 1):
        hc, rc = {}, {}
        for i in range(len(h) - n + 1):
            g = "".join(h[i:i + n]); hc[g] = hc.get(g, 0) + 1
        for i in range(len(r) - n + 1):
            g = "".join(r[i:i + n]); rc[g] = rc.get(g, 0) + 1
        ht, rt = sum(hc.values()), sum(rc.values())
        if ht == 0 and rt == 0:
            continue
        counted += 1
        if ht == 0 or rt == 0:
            continue
        m = sum(min(c, rc.get(g, 0)) for g, c in hc.items())
        ps += m / ht
        rs += m / rt
    if counted == 0:
        return 0.0
    p, rr = ps / counted, rs / counted
    if p == 0 and rr == 0:
        return 0.0
    return 100 * (1 + beta ** 2) * p * rr / (beta ** 2 * p + rr)


JUDGE = ("You grade machine translations of live speech. You get the TARGET LANGUAGE, a REFERENCE translation of a "
         "passage written by a professional translator, and a CANDIDATE translation of the same passage. Score the "
         "candidate on two 1-5 scales. translation_accuracy: does it carry the same meaning as the reference? 5 = same "
         "meaning, nothing missing or added; 3 = the gist survives but details (names, numbers, who did what) are wrong "
         "or missing; 1 = unrelated, untranslated, or in the wrong language. fluency: is it natural, grammatical text in "
         "the target language? Ignore sentence boundaries and punctuation differences. Reply with brief reasoning, then "
         'a JSON object: {"translation_accuracy": <1-5>, "fluency": <1-5>}')


TRANSCRIPT_JUDGE = ("You grade machine translations of live speech. You get the TARGET LANGUAGE, WHAT WAS SAID (a human "
                    "transcript of a meeting passage in the original language, with several speakers, filler words and "
                    "false starts) and a CANDIDATE translation of it. There is no reference translation. Score the "
                    "candidate on two 1-5 scales. translation_accuracy: does it carry the meaning of what was said? 5 = "
                    "same meaning, nothing missing or added; 3 = the gist survives but details (names, numbers, who said "
                    "what) are wrong or missing; 1 = unrelated, untranslated, or in the wrong language. Fillers, false "
                    "starts and cross-talk may be dropped without penalty. fluency: is it natural, grammatical text in "
                    "the target language? Ignore sentence boundaries. Reply with brief reasoning, then a JSON object: "
                    '{"translation_accuracy": <1-5>, "fluency": <1-5>}')


def judge(key, tgt, ref, cand, judge_model, transcript=False):
    h = hashlib.md5(f"{judge_model}|{tgt}|{ref}|{cand}|{transcript}".encode()).hexdigest()
    cache = CACHE / "judge" / f"{h}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    r = prep.http("POST", LLM, key, body={"model": judge_model, "max_tokens": 1500, "messages": [
        {"role": "system", "content": TRANSCRIPT_JUDGE if transcript else JUDGE},
        {"role": "user", "content": f"TARGET LANGUAGE: {NAMES[tgt]}\n\n{'WHAT WAS SAID' if transcript else 'REFERENCE'}:\n{ref}\n\nCANDIDATE:\n{cand}"}]})
    text = r["choices"][0]["message"]["content"]
    obj = json.loads(re.findall(r"\{[^{}]*\}", text)[-1])
    scores = {k: float(obj[k]) for k in ("translation_accuracy", "fluency")}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(scores))
    return scores


def with_babble(stream, babble_pcm, gain=0.25):
    """Mix another speaker in another language under the whole stream, about 12 dB down."""
    import array
    a = array.array("h", stream["pcm"])
    b = array.array("h", babble_pcm)
    for i in range(len(a)):
        a[i] = max(-32768, min(32767, a[i] + int(b[i % len(b)] * gain)))
    return {**stream, "name": "hard-" + stream["name"], "pcm": a.tobytes()}


def build(args, key):
    """Sessions to run: (source, target, stream, text to score against, judged against a transcript?)."""
    pairs = json.loads(args.pairs)
    if args.dataset == "ami":
        targets = [t for t in pairs.get("en", [])] or ["es"]
        out = []
        for m in args.meetings.split(","):
            meeting = prep_ami.load_meeting(m, args.minutes)
            said = prep_ami.join_words(meeting["gold"])
            out += [("en", tgt, meeting, said, True) for tgt in targets]
        return out, pairs | {"en": targets}
    langs = sorted(set(pairs) | {t for ts in pairs.values() for t in ts})
    tsv = {l: prep.read_tsv(l, args.split) for l in langs}
    common = set.intersection(*(set(t) for t in tsv.values()))
    ids = sorted(common, key=lambda i: hashlib.md5(i.encode()).hexdigest())[:args.clips]
    hard = args.dataset == "fleurs_hard"
    out = []
    for src in pairs:
        clips = prep.read_clips(src, args.split, [tsv[src][i]["file"] for i in ids])
        streams = prep.build_streams(src, ids, clips, tsv[src], args.per_stream, 0 if hard else args.gap_ms)
        if hard:
            other = "es" if src != "es" else "en"
            other_clips = prep.read_clips(other, args.split, [tsv[other][i]["file"] for i in ids[::-1]])
            babble = b"".join(other_clips.values())
            streams = [with_babble(s, babble) for s in streams]
        for s in streams:
            for tgt in pairs[src]:
                out.append((src, tgt, s, " ".join(tsv[tgt][span["id"]]["text"] for span in s["spans"]), False))
    return out, pairs


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dataset", default="fleurs", choices=["fleurs", "fleurs_hard", "ami"])
    ap.add_argument("--meetings", default=",".join(prep_ami.DEFAULT_MEETINGS), help="ami only")
    ap.add_argument("--minutes", type=float, default=8, help="ami only: minutes of each meeting")
    ap.add_argument("--prompt", default="interpreter_v1", choices=sorted(PROMPTS))
    ap.add_argument("--model", default="gemini-3.1-flash-lite")
    ap.add_argument("--judge-model", default="claude-opus-5-5")
    ap.add_argument("--pairs", default=json.dumps(prep.DEFAULT_PAIRS))
    ap.add_argument("--clips", type=int, default=40)
    ap.add_argument("--per-stream", type=int, default=8)
    ap.add_argument("--gap-ms", type=int, default=1000)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--streaming-model", default="assemblyai-stt-3-7-realtime")
    args = ap.parse_args()

    key = os.environ.get("ASSEMBLYAI_API_KEY", "")
    if not key:
        sys.exit("Set ASSEMBLYAI_API_KEY first.")

    sessions, pairs = build(args, key)
    jobs = [(src, tgt, stream) for src, tgt, stream, _, _ in sessions]
    print(f"streaming {len(jobs)} sessions with llm_gateway ({args.prompt}, {args.model}) at real-time speed...")

    async def run_all():
        res = await asyncio.gather(*(stream_translate(key, s, src, tgt, args) for src, tgt, s in jobs), return_exceptions=True)
        failed = [(f"{s['name']}-{tgt}", r) for (src, tgt, s), r in zip(jobs, res) if isinstance(r, Exception)]
        for name, err in failed:
            print(f"  FAILED {name}: {err}")
        if failed:
            sys.exit("some sessions failed; re-run to retry (finished sessions are cached)")
        return res

    e2e = asyncio.run(run_all())

    print("async translation of each whole stream...")
    uniq = {}
    for src, tgt, stream, _, _ in sessions:
        uniq.setdefault(stream["name"], (stream, src, sorted({t for s2, t, st2, _, _ in sessions if st2["name"] == stream["name"]})))
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        full = dict(zip(uniq, ex.map(lambda a: async_full(key, *a), uniq.values())))

    print("scoring...")
    rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
        futs = []
        for (src, tgt, s, ref, transcript), r in zip(sessions, e2e):
            for system, text in (("e2e", " ".join(t for t in r["translations"] if t)), ("async_full", full[s["name"]].get(tgt, ""))):
                futs.append((src, tgt, s["name"], system, text, ref, transcript, ex.submit(judge, key, tgt, ref, text, args.judge_model, transcript)))
        for src, tgt, name, system, text, ref, transcript, f in futs:
            j = f.result()
            row = {"pair": f"{src}-{tgt}", "stream": name, "system": system, **j}
            if not transcript:
                row["chrf"] = round(chrf(text, ref), 2)
            rows.append(row)

    lat = [x for r in e2e for x in r["latency_s"]]
    missing = sum(len(r["missing"]) for r in e2e)
    turns_n = sum(len(r["turns"]) for r in e2e)
    against = "the human transcript (no chrF: no reference translations)" if args.dataset == "ami" else "joined references"
    print(f"\n{args.dataset}: prompt {args.prompt}, model {args.model}, judge {args.judge_model}. Whole streams, scored against {against}.")
    print(f"{'pair':6}  {'e2e chrF':>8} {'acc':>5} {'flu':>5}   {'async_full chrF':>15} {'acc':>5} {'flu':>5}")
    summary = {}
    for pair in sorted({r["pair"] for r in rows}) + ["ALL"]:
        line = f"{pair:6} "
        for system in ("e2e", "async_full"):
            sel = [r for r in rows if r["system"] == system and (pair == "ALL" or r["pair"] == pair)]
            m = {k: round(st.mean(r.get(k, float("nan")) for r in sel), 2) for k in ("chrf", "translation_accuracy", "fluency")}
            summary.setdefault(pair, {})[system] = m
            line += f"  {m['chrf']:8.1f} {m['translation_accuracy']:5.2f} {m['fluency']:5.2f}   " if system == "e2e" else f"{m['chrf']:15.1f} {m['translation_accuracy']:5.2f} {m['fluency']:5.2f}"
        print(line)
    lat_sorted = sorted(lat)
    p = lambda q: lat_sorted[min(len(lat_sorted) - 1, int(q * len(lat_sorted)))] if lat_sorted else float("nan")
    print(f"\nturn-end to translation: p50 {p(.5):.2f}s, p95 {p(.95):.2f}s over {len(lat)} turns; "
          f"{missing} of {turns_n} turns got no translation")
    OUT.mkdir(parents=True, exist_ok=True)
    prefix = "" if args.dataset == "fleurs" else f"{args.dataset}__"
    out = OUT / f"{prefix}{args.prompt}__{args.model}.json"
    out.write_text(json.dumps({"args": vars(args), "summary": summary, "latency_p50": p(.5), "latency_p95": p(.95),
                               "missing": missing, "turns": turns_n, "rows": rows}, ensure_ascii=False, indent=1))
    print(f"wrote {out.relative_to(prep.ROOT)}")


if __name__ == "__main__":
    main()
