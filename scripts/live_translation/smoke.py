#!/usr/bin/env python3
"""Quick check that a competitor client works before a full run: streams ~20 seconds of one cached
FLEURS stream (Spanish) through it for English and prints what came back.

    python3 scripts/live_translation/smoke.py soniox
    python3 scripts/live_translation/smoke.py elevenlabs
    python3 scripts/live_translation/smoke.py speechmatics
"""
import asyncio
import hashlib
import sys
import time
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e2e  # noqa: E402
import prep  # noqa: E402


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "soniox"
    src, tgt = "es", "en"
    tsv, ref = prep.read_tsv(src, "test"), prep.read_tsv(tgt, "test")
    ids = sorted(set(tsv) & set(ref), key=lambda i: hashlib.md5(i.encode()).hexdigest())[:3]
    clips = prep.read_clips(src, "test", [tsv[i]["file"] for i in ids])
    stream = {"name": f"smoke-{which}-{int(time.time())}",
              "pcm": b"".join(clips[tsv[i]["file"]] + b"\0" * prep.RATE * 2 for i in ids)}  # 1 s pause between clips
    args = types.SimpleNamespace(dataset="smoke", soniox_model=sys.argv[2] if len(sys.argv) > 2 else "stt-rt-v6",
                                 elevenlabs_eagerness="medium")
    fn = {"soniox": e2e.soniox_translate, "elevenlabs": e2e.elevenlabs_scribe_edit,
          "speechmatics": e2e.speechmatics_translate}[which]
    print(f"streaming {len(stream['pcm']) / prep.RATE / 2:.0f}s of Spanish through {which}...")
    r = asyncio.run(fn(stream, src, tgt, args))
    print("\nwhat was said:   ", " ".join(tsv[i]["text"] for i in ids))
    print("their transcript:", " | ".join(r["turns"]))
    print("their translation:", " | ".join(r["translations"]))
    print("reference:       ", " ".join(ref[i]["text"] for i in ids))
    print("latency (s):     ", r["latency_s"], " errors:", r["errors"])
    print("\nOK" if any(t.strip() for t in r["translations"]) else "\nNO TRANSLATION CAME BACK")


if __name__ == "__main__":
    main()
