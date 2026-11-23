#!/usr/bin/env python3
"""Run the whole live translation benchmark in one go.

    python3 scripts/live_translation/run_all.py
    python3 scripts/live_translation/run_all.py --model qwen3.5-4b-32k-fast --skip-go

1. Builds the datasets that are missing (FLEURS, AMI meetings).
2. Streams every dataset (fleurs, fleurs_hard, ami) through llm_gateway with every prompt (e2e.py).
   The three datasets run side by side; each takes real time, so the whole step is ~25 minutes.
   Finished sessions are cached, so a re-run only redoes what failed.
3. Runs the per-turn Go benchmark once per prompt (skip with --skip-go).
4. Prints one table of everything e2e.py wrote.

Logs go to results/live_translation_e2e/logs/. AMI needs `pip install pyarrow`.
"""
import argparse
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
DATA = ROOT / "normalized" / "live_translation"
OUT = ROOT / "results" / "live_translation_e2e"
LOGS = OUT / "logs"
PROMPTS = ["interpreter_v1", "docs_oneliner", "docs_history"]
DATASETS = ["fleurs", "fleurs_hard", "ami"]


def run(cmd, log, cwd=ROOT, env=None):
    """Run cmd, write its output to log, and say whether it worked."""
    with open(log, "w") as f:
        code = subprocess.call(cmd, cwd=cwd, stdout=f, stderr=subprocess.STDOUT, env=env)
    print(f"  {'ok  ' if code == 0 else 'FAIL'} {' '.join(cmd[1:4] if cmd[0] == sys.executable else cmd[:5])}  (log: {log.relative_to(ROOT)})")
    return code == 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="gemini-3.1-flash-lite")
    ap.add_argument("--prompts", default=",".join(PROMPTS))
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--skip-go", action="store_true", help="skip the per-turn Go benchmark")
    ap.add_argument("--no-elevenlabs", action="store_true", help="skip ElevenLabs even when ELEVENLABS_API_KEY is set")
    ap.add_argument("--elevenlabs-api", default="scribe_edit", choices=["scribe_edit", "translation"])
    args = ap.parse_args()
    prompts, datasets = args.prompts.split(","), args.datasets.split(",")
    if not os.environ.get("ASSEMBLYAI_API_KEY"):
        sys.exit("set ASSEMBLYAI_API_KEY first")
    LOGS.mkdir(parents=True, exist_ok=True)
    py = sys.executable
    eleven = bool(os.environ.get("ELEVENLABS_API_KEY")) and not args.no_elevenlabs
    print(f"ElevenLabs comparison: {'on' if eleven else 'off (set ELEVENLABS_API_KEY to add it)'}")

    print("1. datasets")
    if not (DATA / "fleurs.jsonl").exists():
        if not run([py, str(HERE / "prep.py")], LOGS / "prep_fleurs.log"):
            sys.exit("FLEURS prep failed, see the log")
    if "ami" in datasets and not (DATA / "ami.jsonl").exists():
        if not run([py, str(HERE / "prep_ami.py")], LOGS / "prep_ami.log"):
            print("  AMI download failed (the Edinburgh server is often busy); carrying on without it. Re-run later to add it.")
            datasets.remove("ami")
    print("  datasets ready")

    print(f"2. streaming end to end: {len(datasets)} datasets x {len(prompts)} prompts, datasets in parallel")
    failed = []

    def one_dataset(d):
        for p in prompts:
            log = LOGS / f"e2e_{d}_{p}_{args.model}.log"
            extra = ["--elevenlabs", "--elevenlabs-api", args.elevenlabs_api] if eleven else []
            if not run([py, str(HERE / "e2e.py"), "--dataset", d, "--prompt", p, "--model", args.model] + extra, log):
                failed.append(log)

    threads = [threading.Thread(target=one_dataset, args=(d,)) for d in datasets]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    if not args.skip_go:
        print(f"3. per-turn Go benchmark, one run per prompt")
        env = dict(os.environ)
        env.setdefault("ASSEMBLY_AI_TOKEN", env.get("ASSEMBLYAI_API_KEY", ""))
        for p in prompts:
            env["LIVE_TRANSLATION_PROMPT"] = p
            log = LOGS / f"go_{p}_{args.model}.log"
            if not run(["go", "run", ".", "-tasks", "live_translation", "-models", args.model,
                        "-max-samples-per-bucket", "0"], log, cwd=ROOT / "src", env=env):
                failed.append(log)

    print("\n4. end-to-end results (whole streams; ami has no chrF, it's judged against the human transcript)\n")
    print(f"{'dataset':12} {'prompt':15} | {'ours acc':>8} {'flu':>5} {'chrF':>6} | {'async acc':>9} {'flu':>5} {'chrF':>6} | "
          f"{'11labs acc':>10} {'flu':>5} {'chrF':>6} | {'ours p50':>8} {'p95':>5} | {'11labs p50':>10} {'p95':>5}")
    for d in datasets:
        for p in prompts:
            f = OUT / (f"{p}__{args.model}.json" if d == "fleurs" else f"{d}__{p}__{args.model}.json")
            if not f.exists():
                print(f"{d:12} {p:15} | (no result)")
                continue
            r = json.loads(f.read_text())
            e, a = r["summary"]["ALL"]["e2e"], r["summary"]["ALL"]["async_full"]
            el = r["summary"]["ALL"].get("elevenlabs")
            c = lambda v: "   -  " if v != v else f"{v:6.1f}"  # nan when there's no reference
            el_cols = (f"{el['translation_accuracy']:10.2f} {el['fluency']:5.2f} {c(el['chrf'])}" if el else f"{'-':>10} {'-':>5} {'-':>6}")
            el_lat = (f"{r['elevenlabs_latency_p50']:9.2f}s {r['elevenlabs_latency_p95']:4.2f}s"
                      if el and r.get("elevenlabs_latency_p50") == r.get("elevenlabs_latency_p50") else f"{'-':>10} {'-':>5}")
            print(f"{d:12} {p:15} | {e['translation_accuracy']:8.2f} {e['fluency']:5.2f} {c(e['chrf'])} | "
                  f"{a['translation_accuracy']:9.2f} {a['fluency']:5.2f} {c(a['chrf'])} | {el_cols} | "
                  f"{r['latency_p50']:7.2f}s {r['latency_p95']:4.2f}s | {el_lat}")
    if failed:
        print("\nfailed (re-run to retry; finished parts are cached):")
        for log in failed:
            print(f"  {log.relative_to(ROOT)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
