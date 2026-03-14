from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description="Run rollout evaluation across HotpotQA/TriviaQA/PopQA/PubHealth/MuSIQue.")
    ap.add_argument("--datasets", nargs="+", default=["hotpotqa", "triviaqa", "popqa", "pubhealth", "musique"])
    ap.add_argument("--controller_type", choices=["heuristic", "il", "offline"], default="heuristic")
    ap.add_argument("--policy_ckpt", default=None)
    ap.add_argument("--llm_device", default=None)
    ap.add_argument("--cache_dir", default="./huggingface_cache")
    ap.add_argument("--out", default="logs/multidataset_benchmark.json")
    args = ap.parse_args()

    out_rows = {}
    for ds in args.datasets:
        out_file = Path("logs") / f"eval_{args.controller_type}_{ds}.json"
        cmd = [
            "python",
            "scripts/run_cbvrag_eval.py",
            "--dataset",
            ds,
            "--controller_type",
            args.controller_type,
            "--cache_dir",
            args.cache_dir,
            "--output",
            str(out_file),
        ]
        if args.policy_ckpt:
            cmd += ["--policy_ckpt", args.policy_ckpt]
        if args.llm_device:
            cmd += ["--llm_device", args.llm_device]

        subprocess.run(cmd, check=True)
        payload = json.loads(out_file.read_text(encoding="utf-8"))
        key = f"cbvrag_{args.controller_type}"
        out_rows[ds] = payload[key]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out_rows, indent=2), encoding="utf-8")
    print(json.dumps({"output": args.out, "datasets": args.datasets}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
