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
    python3 scripts/live_translation/e2e.py --dataset ami           # real meetings, judged against the human transcript
Needs ASSEMBLYAI_API_KEY and websockets. About 3 minutes of streaming per prompt and model; results
are cached under scripts/live_translation/.cache/e2e and summarised in results/live_translation_e2e/.
"""
import argparse
import asyncio
import base64
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
        cached = json.loads(cache.read_text())
        if "speech_latency_s" in cached:
            return cached  # older runs lack speech-end latency, so they stream again
    gw = {"model": args.model, "max_tokens": 400, "messages": PROMPTS[args.prompt](NAMES[src], NAMES[tgt])}
    q = {"sample_rate": prep.RATE, "encoding": "pcm_s16le", "speech_model": args.streaming_model,
         "language_codes": json.dumps([src]), "llm_gateway": json.dumps(gw)}
    major = int(str(getattr(websockets, "__version__", "0")).split(".")[0] or 0)
    hdr = "additional_headers" if major >= 14 else "extra_headers"
    turns, trans, turn_at, trans_at, errors = {}, {}, {}, {}, []
    speech_end = {}  # turn -> seconds into the audio where its last word ended
    clock = {}
    pcm, step = stream["pcm"], prep.RATE * 2 // 20
    async with websockets.connect(f"{prep.WS}?{urlencode(q)}", max_size=None, **{hdr: {"Authorization": key}}) as ws:
        async def send():
            t0 = clock["t0"] = time.monotonic()
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
                    ends = [w.get("end") for w in m.get("words", []) if w.get("end") is not None]
                    if ends:
                        speech_end[k] = max(ends) / 1000
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
        # From the moment the speaker's last word ended (the audio is sent in real time, so that is
        # t0 + its timestamp) to the translation arriving: the delay a viewer sees, comparable across vendors.
        "speech_latency_s": [round(trans_at[k] - clock["t0"] - speech_end[k], 3)
                             for k in order if k in trans_at and k in speech_end and "t0" in clock],
        "errors": errors,
    }
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(result, ensure_ascii=False))
    return result


ELEVENLABS_WS = "wss://api.elevenlabs.io/v1/translation/realtime"
ELEVENLABS_STT_WS = "wss://api.elevenlabs.io/v1/speech-to-text/realtime"


async def elevenlabs_scribe_edit(stream, src, tgt, args):
    """ElevenLabs' text-only route: Scribe live transcription with a transcript_edit instruction asking
    for the translation, the same shape as our streaming + llm_gateway. ElevenLabs doesn't sell this as
    translation (Scribe "does not translate"); it is the nearest thing without the enterprise speech-to-speech
    API. Latency is from each committed_transcript to its edited_transcript."""
    import websockets

    base = CACHE if args.dataset == "fleurs" else CACHE / args.dataset
    cache = base / "elevenlabs_scribe_edit" / f"{stream['name']}-{tgt}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    instruction = (f"Translate the transcript from {NAMES[src]} into natural {NAMES[tgt]}. "
                   f"Output only the {NAMES[tgt]} translation, nothing else.")
    q = {"model_id": "scribe_v3_live", "audio_format": "pcm_16000", "language_code": src,
         "commit_strategy": "vad", "transcript_edit": instruction, "enable_logging": "false"}
    major = int(str(getattr(websockets, "__version__", "0")).split(".")[0] or 0)
    hdr = "additional_headers" if major >= 14 else "extra_headers"
    finals, translations, latency, errors, pending = [], [], [], [], []
    done = {"sending": False}
    pcm, step = stream["pcm"], prep.RATE * 2 // 20
    async with websockets.connect(f"{ELEVENLABS_STT_WS}?{urlencode(q)}", max_size=None,
                                  **{hdr: {"xi-api-key": os.environ["ELEVENLABS_API_KEY"]}}) as ws:
        async def send():
            t0 = time.monotonic()
            for i in range(0, len(pcm), step):
                await ws.send(json.dumps({"message_type": "input_audio_chunk", "sample_rate": prep.RATE,
                                          "audio_base_64": base64.b64encode(pcm[i:i + step]).decode()}))
                wait = t0 + (i // step + 1) * 0.05 - time.monotonic()
                if wait > 0:
                    await asyncio.sleep(wait)
            done["sending"] = True
            await ws.send(json.dumps({"message_type": "input_audio_chunk", "sample_rate": prep.RATE,
                                      "audio_base_64": "", "commit": True}))  # flush the last segment
            await asyncio.sleep(6)
            await ws.close()

        async def recv():
            while True:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=15)
                except (asyncio.TimeoutError, websockets.ConnectionClosed):
                    return
                if isinstance(msg, bytes):
                    continue
                m = json.loads(msg)
                kind = m.get("message_type", "")
                if kind == "committed_transcript" and m.get("text", "").strip():
                    finals.append(m["text"])
                    pending.append(time.monotonic())
                elif kind == "edited_transcript":
                    translations.append(m.get("edited_text", m.get("text", "")))
                    if pending:
                        latency.append(round(time.monotonic() - pending.pop(0), 3))
                elif kind == "warning":
                    errors.append(str(m)[:200])
                elif "error" in kind.lower() or kind in ("rate_limited", "auth_error", "quota_exceeded"):
                    if done["sending"]:  # e.g. the empty closing commit refused; the audio is all in
                        errors.append(str(m)[:200])
                        continue
                    raise RuntimeError(f"elevenlabs scribe {stream['name']}-{tgt}: {str(m)[:300]}")

        await asyncio.gather(send(), recv())
    if finals and not translations:
        raise RuntimeError(f"elevenlabs scribe {stream['name']}-{tgt}: transcripts came back but no edited_transcript")
    result = {"turns": finals, "translations": translations, "missing": [], "latency_s": latency, "errors": errors}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(result, ensure_ascii=False))
    return result
ELEVENLABS_VOICE = "21m00Tcm4TlvDq8EMAEw"  # a premade voice; the API requires one and speaks every translation


async def elevenlabs_translate(stream, src, tgt, args):
    """The same audio through ElevenLabs realtime translation, at real-time speed. It decides its own
    segments, so latency is from each final_transcript to the next translation after it."""
    import websockets

    base = CACHE if args.dataset == "fleurs" else CACHE / args.dataset
    cache = base / "elevenlabs" / args.elevenlabs_eagerness / f"{stream['name']}-{tgt}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    q = {"source_language": src, "target_language": tgt, "voice_id": ELEVENLABS_VOICE, "input_format": "pcm_16000",
         "tts_model_id": "eleven_flash_v2_5", "eagerness": args.elevenlabs_eagerness, "enable_logging": "false"}
    major = int(str(getattr(websockets, "__version__", "0")).split(".")[0] or 0)
    hdr = "additional_headers" if major >= 14 else "extra_headers"
    finals, translations, latency, errors = [], [], [], []
    last_final = None
    pcm, step = stream["pcm"], prep.RATE * 2 // 20
    async with websockets.connect(f"{ELEVENLABS_WS}?{urlencode(q)}", max_size=None,
                                  **{hdr: {"xi-api-key": os.environ["ELEVENLABS_API_KEY"]}}) as ws:
        async def send():
            t0 = time.monotonic()
            for i in range(0, len(pcm), step):
                await ws.send(json.dumps({"message_type": "input_audio_chunk",
                                          "audio_base_64": base64.b64encode(pcm[i:i + step]).decode()}))
                wait = t0 + (i // step + 1) * 0.05 - time.monotonic()
                if wait > 0:
                    await asyncio.sleep(wait)
            await asyncio.sleep(2)
            await ws.send(json.dumps({"message_type": "end_of_stream"}))

        async def recv():
            nonlocal last_final
            while True:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=15)
                except (asyncio.TimeoutError, websockets.ConnectionClosed):
                    return
                if isinstance(msg, bytes):
                    continue
                m = json.loads(msg)
                kind = m.get("message_type")
                if kind == "final_transcript" and m.get("text", "").strip():
                    finals.append(m["text"])
                    last_final = time.monotonic()
                elif kind == "translation":
                    translations.append(m.get("text", ""))
                    if last_final is not None:
                        latency.append(round(time.monotonic() - last_final, 3))
                        last_final = None
                elif kind in ("translation_error", "audio_output_error", "transcriber_error"):
                    errors.append(str(m)[:200])
                elif kind == "status" and m.get("status") == "stopped":
                    return
                elif kind and ("error" in kind or kind in ("rate_limited", "queue_overflow")):
                    raise RuntimeError(f"elevenlabs {stream['name']}-{tgt}: {str(m)[:300]}")

        await asyncio.gather(send(), recv())
    result = {"turns": finals, "translations": translations, "missing": [], "latency_s": latency, "errors": errors}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(result, ensure_ascii=False))
    return result


SPEECHMATICS_WS = "wss://eu.rt.speechmatics.com/v2"


async def speechmatics_translate(stream, src, tgt, args):
    """The same audio through Speechmatics real-time transcription with translation, at real-time speed.
    Finals only (AddTranslation); partial translations are off. Latency is from each final AddTranscript
    to the next AddTranslation after it."""
    import websockets

    base = CACHE if args.dataset == "fleurs" else CACHE / args.dataset
    cache = base / "speechmatics" / f"{stream['name']}-{tgt}.json"
    if cache.exists():
        cached = json.loads(cache.read_text())
        if "speech_latency_s" in cached:
            return cached
    start = {"message": "StartRecognition",
             "audio_format": {"type": "raw", "encoding": "pcm_s16le", "sample_rate": prep.RATE},
             "transcription_config": {"language": src, "operating_point": "enhanced"},
             "translation_config": {"target_languages": [tgt], "enable_partials": False}}
    major = int(str(getattr(websockets, "__version__", "0")).split(".")[0] or 0)
    hdr = "additional_headers" if major >= 14 else "extra_headers"
    finals, translations, latency, speech_latency, errors = [], [], [], [], []
    last_final = None
    clock = {}
    pcm, step = stream["pcm"], prep.RATE * 2 // 20
    async with websockets.connect(SPEECHMATICS_WS, max_size=None,
                                  **{hdr: {"Authorization": f"Bearer {os.environ['SPEECHMATICS_API_KEY']}"}}) as ws:
        await ws.send(json.dumps(start))
        started = asyncio.get_running_loop().create_future()

        async def send():
            await asyncio.wait_for(started, timeout=30)
            t0, seq = time.monotonic(), 0
            clock["t0"] = t0
            for i in range(0, len(pcm), step):
                await ws.send(pcm[i:i + step])
                seq += 1
                wait = t0 + (i // step + 1) * 0.05 - time.monotonic()
                if wait > 0:
                    await asyncio.sleep(wait)
            await ws.send(json.dumps({"message": "EndOfStream", "last_seq_no": seq}))

        async def recv():
            nonlocal last_final
            while True:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=60)
                except (asyncio.TimeoutError, websockets.ConnectionClosed):
                    return
                if isinstance(msg, bytes):
                    continue
                m = json.loads(msg)
                kind = m.get("message")
                if kind == "RecognitionStarted" and not started.done():
                    started.set_result(True)
                elif kind == "AddTranscript":
                    text = (m.get("metadata") or {}).get("transcript", "").strip()
                    if text:
                        finals.append(text)
                        last_final = time.monotonic()
                elif kind == "AddTranslation":
                    text = " ".join(r.get("content", "") for r in m.get("results", [])).strip()
                    if text:
                        translations.append(text)
                        ends = [r.get("end_time") for r in m.get("results", []) if r.get("end_time") is not None]
                        if ends and "t0" in clock:
                            speech_latency.append(round(time.monotonic() - clock["t0"] - max(ends), 3))
                        if last_final is not None:
                            latency.append(round(time.monotonic() - last_final, 3))
                            last_final = None
                elif kind == "Warning":
                    errors.append(str(m)[:200])
                elif kind == "Error":
                    if not started.done():
                        started.set_exception(RuntimeError(str(m)[:300]))
                    raise RuntimeError(f"speechmatics {stream['name']}-{tgt}: {str(m)[:300]}")
                elif kind == "EndOfTranscript":
                    return

        await asyncio.gather(send(), recv())
    result = {"turns": finals, "translations": translations, "missing": [], "latency_s": latency,
              "speech_latency_s": speech_latency, "errors": errors}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(result, ensure_ascii=False))
    return result


SONIOX_WS = "wss://stt-rt.soniox.com/transcribe-websocket"


async def soniox_translate(stream, src, tgt, args):
    """The same audio through Soniox real-time transcription with one-way translation, at real-time speed.
    Soniox streams translated tokens mid-sentence; we keep the final ones. Latency is from the last final
    original token to the first final translated token after it."""
    import websockets

    base = CACHE if args.dataset == "fleurs" else CACHE / args.dataset
    cache = base / "soniox" / args.soniox_model / f"{stream['name']}-{tgt}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    key = os.environ["SONIOX_API_KEY"]
    config = {"api_key": key, "model": args.soniox_model, "audio_format": "pcm_s16le", "sample_rate": prep.RATE,
              "num_channels": 1, "language_hints": [src], "enable_endpoint_detection": True,
              "translation": {"type": "one_way", "target_language": tgt}}
    major = int(str(getattr(websockets, "__version__", "0")).split(".")[0] or 0)
    hdr = "additional_headers" if major >= 14 else "extra_headers"
    original, translated, latency, errors = [], [], [], []
    last_original = None
    pcm, step = stream["pcm"], prep.RATE * 2 // 20
    async with websockets.connect(SONIOX_WS, max_size=None, **{hdr: {"Authorization": f"Bearer {key}"}}) as ws:
        await ws.send(json.dumps(config))

        async def send():
            t0 = time.monotonic()
            for i in range(0, len(pcm), step):
                await ws.send(pcm[i:i + step])
                wait = t0 + (i // step + 1) * 0.05 - time.monotonic()
                if wait > 0:
                    await asyncio.sleep(wait)
            await ws.send(b"")  # end of audio; Soniox finalizes and replies with finished: true

        async def recv():
            nonlocal last_original
            async for msg in ws:
                m = json.loads(msg)
                if m.get("error_code") or m.get("error_message"):
                    raise RuntimeError(f"soniox {stream['name']}-{tgt}: {m.get('error_code')} {m.get('error_message')}")
                for tok in m.get("tokens", []):
                    text = tok.get("text", "")
                    if not tok.get("is_final") or (text.startswith("<") and text.endswith(">")):
                        continue  # partials get re-sent as finals; <end>, <fin> are markers
                    if tok.get("translation_status") == "translation":
                        if last_original is not None:
                            latency.append(round(time.monotonic() - last_original, 3))
                            last_original = None
                        translated.append(text)
                    else:
                        original.append(text)
                        last_original = time.monotonic()
                if m.get("finished"):
                    return

        await asyncio.gather(send(), recv())
    result = {"turns": ["".join(original).strip()], "translations": ["".join(translated).strip()],
              "missing": [], "latency_s": latency, "errors": errors}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(result, ensure_ascii=False))
    return result


ELEVENLABS_API = "https://api.elevenlabs.io/v1"


def elevenlabs_request(method, path, data=None, content_type=None):
    import urllib.error
    import urllib.request
    headers = {"xi-api-key": os.environ["ELEVENLABS_API_KEY"]}
    if content_type:
        headers["content-type"] = content_type
    req = urllib.request.Request(f"{ELEVENLABS_API}{path}", data=data, headers=headers, method=method)
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504) and attempt < 4:
                time.sleep(10 * (attempt + 1))
                continue
            raise RuntimeError(f"ElevenLabs {method} {path}: {e.code} {e.read().decode(errors='replace')[:300]}")


def elevenlabs_dub(stream, src, tgt, dataset):
    """ElevenLabs batch dubbing of the whole stream file; we keep only its translated transcript."""
    base = CACHE if dataset == "fleurs" else CACHE / dataset
    cache = base / "elevenlabs_dub" / f"{stream['name']}-{tgt}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    boundary = "----vsb" + hashlib.md5(stream["name"].encode()).hexdigest()
    fields = {"source_lang": src, "target_lang": tgt, "num_speakers": "0", "watermark": "true", "name": f"vsb-{stream['name']}-{tgt}"}
    body = b"".join(f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode() for k, v in fields.items())
    body += (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{stream["name"]}.wav"\r\n'
             f"Content-Type: audio/wav\r\n\r\n").encode() + prep.wav_bytes(stream["pcm"]) + f"\r\n--{boundary}--\r\n".encode()
    dub = elevenlabs_request("POST", "/dubbing", data=body, content_type=f"multipart/form-data; boundary={boundary}")["dubbing_id"]
    t0 = time.monotonic()
    while True:
        status = elevenlabs_request("GET", f"/dubbing/{dub}")
        if status.get("status") == "dubbed":
            break
        if status.get("status") == "failed" or time.monotonic() - t0 > 1800:
            raise RuntimeError(f"ElevenLabs dub {stream['name']}-{tgt}: {str(status)[:300]}")
        time.sleep(10)
    tr = elevenlabs_request("GET", f"/dubbing/{dub}/transcripts/{tgt}/format/json")
    utterances = (tr.get("json") or {}).get("utterances") or []
    text = " ".join(u.get("text", "").strip() for u in sorted(utterances, key=lambda u: u.get("start_s", 0)) if u.get("text"))
    result = {"dubbing_id": dub, "text": text, "seconds": round(time.monotonic() - t0, 1)}
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
    if not cand.strip():
        return {"translation_accuracy": 1.0, "fluency": 1.0}  # nothing came back: the rubric's floor
    scores = None
    for attempt in range(3):  # a long meeting can push the score past max_tokens, or the reply can skip the JSON
        r = prep.http("POST", LLM, key, body={"model": judge_model, "max_tokens": 3000, "messages": [
            {"role": "system", "content": TRANSCRIPT_JUDGE if transcript else JUDGE},
            {"role": "user", "content": f"TARGET LANGUAGE: {NAMES[tgt]}\n\n{'WHAT WAS SAID' if transcript else 'REFERENCE'}:\n{ref}\n\nCANDIDATE:\n{cand}"
             + ("" if attempt == 0 else "\n\nKeep the reasoning to two sentences and end with the JSON object.")}]})
        text = r["choices"][0]["message"]["content"] or ""
        found = re.findall(r"\{[^{}]*\}", text)
        try:
            obj = json.loads(found[-1])
            scores = {k: float(obj[k]) for k in ("translation_accuracy", "fluency")}
            break
        except (IndexError, KeyError, ValueError, TypeError):
            continue
    if scores is None:
        print(f"  judge gave no scores for one {tgt} candidate after 3 tries; left out of the means")
        return {}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(scores))
    return scores


def build(args, key):
    """Sessions to run: (source, target, stream, text to score against, judged against a transcript?)."""
    pairs = json.loads(args.pairs)
    if args.dataset == "ami":
        targets = [t for t in pairs.get("en", [])] or ["es"]
        out = []
        names = args.meetings.split(",")
        for m in names:
            meeting = prep_ami.load_meeting(m, args.minutes, names)
            said = prep_ami.join_words(meeting["gold"])
            out += [("en", tgt, meeting, said, True) for tgt in targets]
        return out, pairs | {"en": targets}
    langs = sorted(set(pairs) | {t for ts in pairs.values() for t in ts})
    tsv = {l: prep.read_tsv(l, args.split) for l in langs}
    common = set.intersection(*(set(t) for t in tsv.values()))
    ids = sorted(common, key=lambda i: hashlib.md5(i.encode()).hexdigest())[:args.clips]
    out = []
    for src in pairs:
        clips = prep.read_clips(src, args.split, [tsv[src][i]["file"] for i in ids])
        streams = prep.build_streams(src, ids, clips, tsv[src], args.per_stream, args.gap_ms)
        for s in streams:
            for tgt in pairs[src]:
                out.append((src, tgt, s, " ".join(tsv[tgt][span["id"]]["text"] for span in s["spans"]), False))
    return out, pairs


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dataset", default="fleurs", choices=["fleurs", "ami"])
    ap.add_argument("--meetings", default=",".join(prep_ami.DEFAULT_MEETINGS), help="ami only")
    ap.add_argument("--minutes", type=float, default=8, help="ami only: minutes of each meeting")
    ap.add_argument("--prompt", default="interpreter_v1", choices=sorted(PROMPTS))
    ap.add_argument("--elevenlabs", action="store_true",
                    help="also run ElevenLabs on the same audio (needs ELEVENLABS_API_KEY)")
    ap.add_argument("--elevenlabs-api", default="scribe_edit", choices=["scribe_edit", "translation"],
                    help="scribe_edit: Scribe live + a transcript_edit asking for the translation (text only, any plan). "
                         "translation: the speech-to-speech realtime translation API (enterprise only)")
    ap.add_argument("--speechmatics", action="store_true",
                    help="also run Speechmatics real-time translation, finals only (needs SPEECHMATICS_API_KEY)")
    ap.add_argument("--speechmatics-concurrency", type=int, default=2,
                    help="sessions at once; Speechmatics refuses more than the account's concurrent quota")
    ap.add_argument("--speechmatics-per-pair", type=int, default=2,
                    help="streams per language pair to run through Speechmatics (0 = all); ours is also scored on the same ones")
    ap.add_argument("--soniox", action="store_true", help="also run Soniox real-time translation (needs SONIOX_API_KEY)")
    ap.add_argument("--soniox-model", default="stt-rt-v6", help="Soniox real-time model (stt-rt-v5 if v6 is refused)")
    ap.add_argument("--elevenlabs-dub", action="store_true",
                    help="also run ElevenLabs batch dubbing on each whole stream and score its translated transcript "
                         "(needs ELEVENLABS_API_KEY; bills dubbing minutes per stream and target)")
    ap.add_argument("--elevenlabs-eagerness", default="medium", choices=["low", "medium", "high"])
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

    eleven = [None] * len(jobs)
    if args.elevenlabs:
        if not os.environ.get("ELEVENLABS_API_KEY"):
            sys.exit("--elevenlabs needs ELEVENLABS_API_KEY")
        print(f"ElevenLabs ({args.elevenlabs_api}) on the same {len(jobs)} sessions...")

        async def run_eleven():
            sem = asyncio.Semaphore(10)  # stay under a workspace's concurrent session limit

            async def one(src, tgt, s):
                async with sem:
                    if args.elevenlabs_api == "scribe_edit":
                        return await elevenlabs_scribe_edit(s, src, tgt, args)
                    return await elevenlabs_translate(s, src, tgt, args)

            return await asyncio.gather(*(one(src, tgt, s) for src, tgt, s in jobs), return_exceptions=True)

        eleven = asyncio.run(run_eleven())
        for (src, tgt, s), r in zip(jobs, eleven):
            if isinstance(r, Exception):
                print(f"  ElevenLabs FAILED {s['name']}-{tgt}: {r}")
        if all(isinstance(r, Exception) for r in eleven):
            sys.exit("every ElevenLabs session failed; see the errors above, or re-run without --elevenlabs")
        eleven = [None if isinstance(r, Exception) else r for r in eleven]

    smx = [None] * len(jobs)
    if args.speechmatics:
        if not os.environ.get("SPEECHMATICS_API_KEY"):
            sys.exit("--speechmatics needs SPEECHMATICS_API_KEY")
        picked, per_pair = [], {}
        for i, (src, tgt, s) in enumerate(jobs):
            per_pair[(src, tgt)] = per_pair.get((src, tgt), 0) + 1
            if args.speechmatics_per_pair <= 0 or per_pair[(src, tgt)] <= args.speechmatics_per_pair:
                picked.append(i)
        print(f"Speechmatics real-time translation of {len(picked)} of the {len(jobs)} sessions "
              f"({args.speechmatics_concurrency} at a time)...")

        async def run_speechmatics():
            sem = asyncio.Semaphore(args.speechmatics_concurrency)

            async def one(src, tgt, s):
                async with sem:
                    for attempt in range(40):
                        try:
                            return await speechmatics_translate(s, src, tgt, args)
                        except RuntimeError as e:
                            if "quota_exceeded" not in str(e) or attempt == 39:
                                raise
                            await asyncio.sleep(15)  # another run holds the account's sessions; wait for one

            out = await asyncio.gather(*(one(*jobs[i]) for i in picked), return_exceptions=True)
            full_list = [None] * len(jobs)
            for i, r in zip(picked, out):
                full_list[i] = r
            return full_list

        smx = asyncio.run(run_speechmatics())
        for (src, tgt, s), r in zip(jobs, smx):
            if isinstance(r, Exception):
                print(f"  Speechmatics FAILED {s['name']}-{tgt}: {r}")
        smx = [None if isinstance(r, Exception) else r for r in smx]
        print(f"  Speechmatics: {sum(r is not None for r in smx)} sessions done")

    sonx = [None] * len(jobs)
    if args.soniox:
        if not os.environ.get("SONIOX_API_KEY"):
            sys.exit("--soniox needs SONIOX_API_KEY")
        print(f"Soniox real-time translation ({args.soniox_model}) of the same {len(jobs)} sessions...")

        async def run_soniox():
            sem = asyncio.Semaphore(10)

            async def one(src, tgt, s):
                async with sem:
                    return await soniox_translate(s, src, tgt, args)

            return await asyncio.gather(*(one(src, tgt, s) for src, tgt, s in jobs), return_exceptions=True)

        sonx = asyncio.run(run_soniox())
        for (src, tgt, s), r in zip(jobs, sonx):
            if isinstance(r, Exception):
                print(f"  Soniox FAILED {s['name']}-{tgt}: {r}")
        sonx = [None if isinstance(r, Exception) else r for r in sonx]

    dubs = [None] * len(jobs)
    if args.elevenlabs_dub:
        if not os.environ.get("ELEVENLABS_API_KEY"):
            sys.exit("--elevenlabs-dub needs ELEVENLABS_API_KEY")
        print(f"ElevenLabs batch dubbing of {len(jobs)} stream x target files (polls until each is dubbed)...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:
            futs = [ex.submit(elevenlabs_dub, s, src, tgt, args.dataset) for src, tgt, s in jobs]
            for i, ((src, tgt, s), f) in enumerate(zip(jobs, futs)):
                try:
                    dubs[i] = f.result()
                except Exception as e:  # noqa: BLE001 - one failed dub shouldn't sink the run
                    print(f"  ElevenLabs dub FAILED {s['name']}-{tgt}: {e}")

    print("scoring...")
    rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
        futs = []
        for (src, tgt, s, ref, transcript), r, el, dub, sx, sm in zip(sessions, e2e, eleven, dubs, sonx, smx):
            systems = [("e2e", " ".join(t for t in r["translations"] if t)), ("async_full", full[s["name"]].get(tgt, ""))]
            if el is not None:
                systems.append(("elevenlabs", " ".join(t for t in el["translations"] if t)))
            if dub is not None:
                systems.append(("elevenlabs_dub", dub["text"]))
            if sx is not None:
                systems.append(("soniox", sx["translations"][0]))
            if sm is not None:
                systems.append(("speechmatics", " ".join(sm["translations"])))
                systems.append(("e2e_vs_speechmatics", systems[0][1]))  # ours on exactly the sessions Speechmatics ran
            for system, text in systems:
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
    systems = [x for x in ("e2e", "async_full", "e2e_vs_speechmatics", "speechmatics", "soniox", "elevenlabs", "elevenlabs_dub") if any(r["system"] == x for r in rows)]
    print(f"{'pair':6}" + "".join(f" | {x + ' chrF':>16} {'acc':>5} {'flu':>5}" for x in systems))
    summary = {}
    for pair in sorted({r["pair"] for r in rows}) + ["ALL"]:
        line = f"{pair:6}"
        for system in systems:
            sel = [r for r in rows if r["system"] == system and (pair == "ALL" or r["pair"] == pair)]
            vals = lambda k: [r[k] for r in sel if r.get(k) is not None and r[k] == r[k]]
            m = {k: round(st.mean(vals(k)), 2) if vals(k) else float("nan")
                 for k in ("chrf", "translation_accuracy", "fluency")}
            summary.setdefault(pair, {})[system] = m
            line += f" | {m['chrf']:16.1f} {m['translation_accuracy']:5.2f} {m['fluency']:5.2f}"
        print(line)
    lat_sorted = sorted(lat)
    p = lambda q: lat_sorted[min(len(lat_sorted) - 1, int(q * len(lat_sorted)))] if lat_sorted else float("nan")
    print(f"\nturn-end to translation: p50 {p(.5):.2f}s, p95 {p(.95):.2f}s over {len(lat)} turns; "
          f"{missing} of {turns_n} turns got no translation")
    pct = lambda xs, q: sorted(xs)[min(len(xs) - 1, int(q * len(xs)))] if xs else float("nan")
    ours_speech = [x for r in e2e for x in r.get("speech_latency_s", [])]
    ours_speech_same = [x for r, sm in zip(e2e, smx) if sm is not None for x in r.get("speech_latency_s", [])]
    print(f"OURS speech end to translation: p50 {pct(ours_speech, .5):.2f}s, p95 {pct(ours_speech, .95):.2f}s over {len(ours_speech)} turns")
    sm_lat = sorted(x for r in smx if r for x in r["latency_s"])
    sm_p = lambda q: sm_lat[min(len(sm_lat) - 1, int(q * len(sm_lat)))] if sm_lat else float("nan")
    sm_speech = [x for r in smx if r for x in r.get("speech_latency_s", [])]
    if args.speechmatics:
        print(f"Speechmatics speech end to translation: p50 {pct(sm_speech, .5):.2f}s, p95 {pct(sm_speech, .95):.2f}s over {len(sm_speech)} segments; "
              f"final transcript to translation p50 {sm_p(.5):.2f}s; {sum(r is None for r in smx)} sessions failed")
    sx_lat = sorted(x for r in sonx if r for x in r["latency_s"])
    sx_p = lambda q: sx_lat[min(len(sx_lat) - 1, int(q * len(sx_lat)))] if sx_lat else float("nan")
    if args.soniox:
        print(f"Soniox last original token to translation: p50 {sx_p(.5):.2f}s, p95 {sx_p(.95):.2f}s over {len(sx_lat)} segments; "
              f"{sum(r is None for r in sonx)} sessions failed")
    el_lat = sorted(x for r in eleven if r for x in r["latency_s"])
    el_p = lambda q: el_lat[min(len(el_lat) - 1, int(q * len(el_lat)))] if el_lat else float("nan")
    if args.elevenlabs:
        print(f"ElevenLabs segment end to translation: p50 {el_p(.5):.2f}s, p95 {el_p(.95):.2f}s over {len(el_lat)} segments; "
              f"{sum(r is None for r in eleven)} sessions failed")
    OUT.mkdir(parents=True, exist_ok=True)
    prefix = "" if args.dataset == "fleurs" else f"{args.dataset}__"
    out = OUT / f"{prefix}{args.prompt}__{args.model}.json"
    out.write_text(json.dumps({"args": vars(args), "summary": summary, "latency_p50": p(.5), "latency_p95": p(.95),
                               "speechmatics_latency_p50": sm_p(.5), "speechmatics_latency_p95": sm_p(.95),
                               "speech_latency_p50": pct(ours_speech, .5), "speech_latency_p95": pct(ours_speech, .95),
                               "e2e_vs_speechmatics_speech_latency_p50": pct(ours_speech_same, .5),
                               "e2e_vs_speechmatics_speech_latency_p95": pct(ours_speech_same, .95),
                               "speechmatics_speech_latency_p50": pct(sm_speech, .5), "speechmatics_speech_latency_p95": pct(sm_speech, .95),
                               "soniox_latency_p50": sx_p(.5), "soniox_latency_p95": sx_p(.95),
                               "elevenlabs_latency_p50": el_p(.5), "elevenlabs_latency_p95": el_p(.95),
                               "missing": missing, "turns": turns_n, "rows": rows}, ensure_ascii=False, indent=1))
    print(f"wrote {out.relative_to(prep.ROOT)}")


if __name__ == "__main__":
    main()
