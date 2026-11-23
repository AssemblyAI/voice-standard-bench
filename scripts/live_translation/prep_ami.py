#!/usr/bin/env python3
"""Build normalized/live_translation/ami.jsonl: real multi-speaker meetings for the live_translation task.

The AMI Meeting Corpus (CC BY 4.0) has English meetings of about four people, with overlap, cross-talk
and a human word-level transcript. There are no reference translations, so these records carry the
human transcript of what was said in each turn (input.source_truth) and are judged against it rather
than scored with chrF.

1. Downloads the headset mix of each meeting and the manual annotations (about 25 MB, plus about
   60 MB per meeting).
2. Streams the first --minutes of each meeting once through the notetaker model at real-time speed.
3. Gives every final streaming turn the human-transcribed words spoken during it.
4. Runs async speech understanding translation on each turn's audio, the async-su-translation row.
5. Writes one record per (turn, target language) with the previous 3 turns as context.

Run from the repo root (needs ASSEMBLYAI_API_KEY and websockets):
    python3 scripts/live_translation/prep_ami.py
Streaming runs every meeting at once at real-time speed, so it takes about --minutes (default 8).
Report buckets are en-<target>/ami.
"""
import argparse
import asyncio
import concurrent.futures
import io
import json
import os
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prep  # noqa: E402

ANNOTATIONS_URL = "https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip"
WAV_URL = "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/{m}/audio/{m}.Mix-Headset.wav"
# From the AMI full-corpus test set: three scenario meetings and one free meeting.
DEFAULT_MEETINGS = ["ES2004a", "IS1009a", "TS3003a", "EN2002a"]
CACHE = prep.CACHE / "ami"


def gold_words(meeting):
    """Every human-transcribed word of the meeting, all speakers, sorted by start time (seconds)."""
    zpath = prep.fetch(ANNOTATIONS_URL, CACHE / "ami_public_manual_1.6.2.zip")
    words = []
    with zipfile.ZipFile(zpath) as z:
        names = [n for n in z.namelist() if n.split("/")[-1].startswith(f"{meeting}.") and n.endswith(".words.xml")]
        if not names:
            sys.exit(f"no word annotations for {meeting} in the AMI zip")
        for name in names:
            speaker = name.split("/")[-1].split(".")[1]
            for el in ET.fromstring(z.read(name)).iter():
                if el.tag.split("}")[-1] != "w" or "starttime" not in el.attrib or not (el.text or "").strip():
                    continue
                words.append({"start": float(el.attrib["starttime"]), "end": float(el.attrib.get("endtime", el.attrib["starttime"])),
                              "text": el.text.strip(), "punc": "punc" in el.attrib, "speaker": speaker})
    return sorted(words, key=lambda w: w["start"])


def join_words(words):
    out = ""
    for w in words:
        out += w["text"] if w["punc"] or not out else " " + w["text"]
    return out


def load_meeting(meeting, minutes):
    wav = prep.fetch(WAV_URL.format(m=meeting), CACHE / f"{meeting}.Mix-Headset.wav")
    pcm = prep.pcm16_mono_16k(wav.read_bytes())[: int(minutes * 60 * prep.RATE) * 2]
    gold = [w for w in gold_words(meeting) if w["start"] < minutes * 60]
    return {"name": f"ami-{meeting}", "pcm": pcm, "gold": gold}


def turns_with_gold(turns, gold):
    """Final turns that have words, each with its time span and the human words spoken during it.
    A human word goes to the turn whose span holds its midpoint, else to the nearest turn within 1s;
    words the stream dropped entirely belong to no turn, so a missed utterance shows up only in the
    end-to-end check, not here."""
    spans = []
    for t in turns:
        ws = [w for w in t["words"] if w.get("text")]
        if ws:
            spans.append({"text": t["transcript"] or " ".join(w["text"] for w in ws),
                          "start": min(w["start"] for w in ws) / 1000, "end": max(w["end"] for w in ws) / 1000, "gold": []})
    for w in gold:
        mid = (w["start"] + w["end"]) / 2
        best, dist = None, 1.0
        for i, s in enumerate(spans):
            d = 0.0 if s["start"] <= mid <= s["end"] else min(abs(mid - s["start"]), abs(mid - s["end"]))
            if d <= dist:
                best, dist = i, d
                if d == 0.0:
                    break
        if best is not None:
            spans[best]["gold"].append(w)
    for s in spans:
        s["gold"] = join_words(sorted(s["gold"], key=lambda w: w["start"]))
    return spans


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--meetings", default=",".join(DEFAULT_MEETINGS))
    ap.add_argument("--minutes", type=float, default=8)
    ap.add_argument("--targets", default="es,fr,de,pt,ja")
    ap.add_argument("--streaming-model", default="assemblyai-stt-3-7-realtime")
    ap.add_argument("--mode", default="")
    ap.add_argument("--history", type=int, default=3)
    args = ap.parse_args()

    key = os.environ.get("ASSEMBLYAI_API_KEY", "")
    if not key:
        sys.exit("Set ASSEMBLYAI_API_KEY first.")
    targets = args.targets.split(",")
    meetings = [load_meeting(m, args.minutes) for m in args.meetings.split(",")]
    print(f"streaming {len(meetings)} meetings ({args.minutes:g} min each) through {args.streaming_model} at real-time speed")

    async def run_all():
        res = await asyncio.gather(*(prep.stream_once(key, m, "en", args.streaming_model, args.mode) for m in meetings), return_exceptions=True)
        bad = [(m["name"], r) for m, r in zip(meetings, res) if isinstance(r, Exception)]
        for name, err in bad:
            print(f"  FAILED {name}: {err}")
        if bad:
            sys.exit("streaming failed; re-run to retry (finished meetings are cached)")
        return res

    turns = dict(zip([m["name"] for m in meetings], asyncio.run(run_all())))

    spans = {m["name"]: turns_with_gold(turns[m["name"]], m["gold"]) for m in meetings}
    print(f"async speech understanding translation, per turn ({sum(len(v) for v in spans.values())} turns)")
    jobs = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        for m in meetings:
            for i, s in enumerate(spans[m["name"]]):
                a = max(0, int((s["start"] - 0.3) * prep.RATE)) * 2
                b = int((s["end"] + 0.3) * prep.RATE) * 2
                jobs[(m["name"], i)] = ex.submit(prep.async_translate, key, "en", f"{m['name']}-t{i:03d}", m["pcm"][a:b], targets)
        async_out = {k: f.result() for k, f in jobs.items()}

    variant = "ami" + (f"-{args.mode}" if args.mode else "")
    out_path = prep.OUT.with_name(f"{variant}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for m in meetings:
            ss = spans[m["name"]]
            for i, s in enumerate(ss):
                if not s["gold"]:
                    continue  # nothing a person said; noise the model turned into words
                for tgt in targets:
                    out.write(json.dumps({
                        "id": f"{m['name']}-t{i:03d}-{tgt}" + (f"-{args.mode}" if args.mode else ""),
                        "dataset": "ami/manual-1.6.2",
                        "input": {"source_language": "en", "target_language": tgt, "turn": s["text"],
                                  "history": [h["text"] for h in ss[max(0, i - args.history):i]],
                                  "source_truth": s["gold"], "variant": variant,
                                  "streaming_model": args.streaming_model},
                        "expected": {"reference": ""},
                        "baselines": {"async_su_translation": async_out[(m["name"], i)]["translated_texts"].get(tgt, "")},
                    }, ensure_ascii=False) + "\n")
                    n += 1
    print(f"wrote {n} records to {out_path.relative_to(prep.ROOT)}")


if __name__ == "__main__":
    main()
