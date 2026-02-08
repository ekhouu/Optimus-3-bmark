#!/usr/bin/env python3
"""Export high-quality planning traces into SFT jsonl chat format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert benchmark traces into SFT chat dataset.")
    parser.add_argument("--traces", required=True, help="Path to traces_*.jsonl from mine_diamonds_model_bench.py")
    parser.add_argument("--out", required=True, help="Output jsonl path")
    parser.add_argument("--min-score", type=float, default=0.65)
    parser.add_argument("--require-success", action="store_true")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    total = 0
    with open(args.traces, encoding="utf-8") as inp, open(out_path, "w", encoding="utf-8") as out:
        for line in inp:
            total += 1
            row = json.loads(line)
            if row.get("error"):
                continue
            if float(row.get("score", 0.0)) < args.min_score:
                continue
            if args.require_success and int(row.get("success", 0)) != 1:
                continue
            task = row["task"]
            response = row["response"]
            sample = {
                "messages": [
                    {"role": "system", "content": "You are an expert Minecraft planner."},
                    {"role": "user", "content": f"How to {task} from scratch?"},
                    {"role": "assistant", "content": response},
                ],
                "meta": {
                    "source_model": row.get("model"),
                    "score": row.get("score"),
                    "success": row.get("success"),
                    "milestone_coverage": row.get("milestone_coverage"),
                    "order_score": row.get("order_score"),
                },
            }
            out.write(json.dumps(sample, ensure_ascii=True) + "\n")
            kept += 1

    print(f"kept={kept} total={total} out={out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
