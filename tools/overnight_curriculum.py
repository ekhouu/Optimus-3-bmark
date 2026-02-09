#!/usr/bin/env python3
"""Autonomous stone->iron curriculum runner.

For each stage:
- Run `prime_rl_campaign.py` for N episodes per cycle.
- Run `distill_openai_to_open_model.py`.
- Save cycle checkpoint artifacts.

Designed for long unattended runs (overnight).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def _timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _append_command_log(path: Path, command: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(f"## {_timestamp()}\n\n")
        f.write("```bash\n")
        f.write(" ".join(command) + "\n")
        f.write("```\n\n")


def _run_command(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    log_file: Path,
    command_log: Path,
) -> int:
    _append_command_log(command_log, command)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as lf:
        lf.write(f"\n[{_timestamp()}] CMD: {' '.join(command)}\n")
        lf.flush()
        proc = subprocess.Popen(
            command,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="", flush=True)
            lf.write(line)
        rc = proc.wait()
        lf.write(f"[{_timestamp()}] RC={rc}\n")
        return rc


def _latest_dir_with_prefix(parent: Path, prefix: str) -> Path | None:
    if not parent.exists():
        return None
    dirs = [p for p in parent.iterdir() if p.is_dir() and p.name.startswith(prefix)]
    if not dirs:
        return None
    return sorted(dirs)[-1]


def _copytree_replace(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run stone->iron overnight curriculum with periodic retraining.")

    parser.add_argument("--base-url", default="http://127.0.0.1:9500")
    parser.add_argument("--planning-task-type", choices=["planning", "orchestrate"], default="orchestrate")
    parser.add_argument("--device", default="cuda:0")

    parser.add_argument("--episodes-per-cycle", type=int, default=100)
    parser.add_argument("--cycles-per-stage", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--replan-threshold-seconds", type=int, default=120)
    parser.add_argument("--step-sleep-seconds", type=float, default=0.4)
    parser.add_argument("--continue-on-error", action="store_true")

    parser.add_argument("--stone-tasks-file", default="tools/obtain_stone_tasks.txt")
    parser.add_argument("--iron-tasks-file", default="tools/obtain_iron_tasks.txt")

    parser.add_argument("--student-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--teacher-model", default="gpt-4.1-mini")
    parser.add_argument("--openai-base-url", default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--openai-api-key", default=os.getenv("OPENAI_API_KEY", ""))
    parser.add_argument("--repeats-per-task", type=int, default=6)
    parser.add_argument("--distill-epochs", type=float, default=1.0)
    parser.add_argument("--distill-max-length", type=int, default=1024)
    parser.add_argument("--distill-learning-rate", type=float, default=2e-4)
    parser.add_argument("--distill-train-batch-size", type=int, default=1)
    parser.add_argument("--distill-grad-accum-steps", type=int, default=8)
    parser.add_argument("--skip-distill", action="store_true")

    parser.add_argument("--discord-webhook-url", default=os.getenv("DISCORD_WEBHOOK_URL", ""))
    parser.add_argument("--discord-min-interval-s", type=int, default=20)
    parser.add_argument("--discord-verbose", action="store_true")

    parser.add_argument("--out-root", default="/workspace/outputs/overnight_curriculum")
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cwd = Path(__file__).resolve().parents[1]

    out_root = Path(args.out_root)
    run_root = out_root / f"run_{_timestamp()}"
    run_root.mkdir(parents=True, exist_ok=True)

    cmd_log_md = run_root / "train_commands.md"
    run_log_txt = run_root / "run.log"
    manifest_jsonl = run_root / "manifest.jsonl"

    stages = [
        {
            "name": "stone",
            "rubric_profile": "stone",
            "tasks_file": str(Path(args.stone_tasks_file)),
        },
        {
            "name": "iron",
            "rubric_profile": "iron",
            "tasks_file": str(Path(args.iron_tasks_file)),
        },
    ]

    for stage in stages:
        tasks_file = Path(stage["tasks_file"])
        if not tasks_file.exists():
            raise SystemExit(f"Missing tasks file: {tasks_file}")

    if not args.skip_distill and not args.openai_api_key:
        raise SystemExit("OPENAI_API_KEY / --openai-api-key is required unless --skip-distill is set.")

    env = os.environ.copy()
    if args.openai_api_key:
        env["OPENAI_API_KEY"] = args.openai_api_key
    if args.openai_base_url:
        env["OPENAI_BASE_URL"] = args.openai_base_url
    if args.discord_webhook_url:
        env["DISCORD_WEBHOOK_URL"] = args.discord_webhook_url

    print(f"[overnight] run_root={run_root}", flush=True)
    print(
        f"[overnight] episodes_per_cycle={args.episodes_per_cycle} cycles_per_stage={args.cycles_per_stage} "
        f"skip_distill={args.skip_distill}",
        flush=True,
    )

    for stage in stages:
        stage_name = stage["name"]
        stage_dir = run_root / stage_name
        stage_dir.mkdir(parents=True, exist_ok=True)

        for cycle in range(1, args.cycles_per_stage + 1):
            cycle_id = f"cycle_{cycle:02d}"
            cycle_dir = stage_dir / cycle_id
            cycle_dir.mkdir(parents=True, exist_ok=True)

            campaign_out_dir = cycle_dir / "campaign"
            distill_out_dir = cycle_dir / "distill"
            checkpoints_dir = cycle_dir / "checkpoints"
            checkpoints_dir.mkdir(parents=True, exist_ok=True)

            campaign_cmd = [
                args.python_bin,
                "tools/prime_rl_campaign.py",
                "--base-url",
                args.base_url,
                "--tasks-file",
                stage["tasks_file"],
                "--episodes",
                str(args.episodes_per_cycle),
                "--out-dir",
                str(campaign_out_dir),
                "--planning-task-type",
                args.planning_task_type,
                "--max-steps",
                str(args.max_steps),
                "--replan-threshold-seconds",
                str(args.replan_threshold_seconds),
                "--step-sleep-seconds",
                str(args.step_sleep_seconds),
                "--device",
                args.device,
                "--rubric-profile",
                stage["rubric_profile"],
                "--discord-min-interval-s",
                str(args.discord_min_interval_s),
                "--discord-send-final-artifacts",
                "--continue-on-error",
            ]
            if args.discord_webhook_url:
                campaign_cmd.extend(["--discord-webhook-url", args.discord_webhook_url])
            if args.discord_verbose:
                campaign_cmd.append("--discord-verbose")

            print(f"[overnight] stage={stage_name} {cycle_id} campaign start", flush=True)
            if args.dry_run:
                _append_command_log(cmd_log_md, campaign_cmd)
                campaign_rc = 0
            else:
                campaign_rc = _run_command(
                    campaign_cmd,
                    cwd=cwd,
                    env=env,
                    log_file=run_log_txt,
                    command_log=cmd_log_md,
                )

            latest_campaign = _latest_dir_with_prefix(campaign_out_dir, "campaign_")
            if latest_campaign and (latest_campaign / "aggregate.json").exists():
                shutil.copy2(latest_campaign / "aggregate.json", checkpoints_dir / "aggregate.json")
                if (latest_campaign / "campaign_progress.png").exists():
                    shutil.copy2(latest_campaign / "campaign_progress.png", checkpoints_dir / "campaign_progress.png")

            distill_rc = 0
            adapter_path = None
            if not args.skip_distill:
                distill_cmd = [
                    args.python_bin,
                    "tools/distill_openai_to_open_model.py",
                    "--openai-base-url",
                    args.openai_base_url,
                    "--openai-api-key",
                    args.openai_api_key,
                    "--teacher-model",
                    args.teacher_model,
                    "--student-model",
                    args.student_model,
                    "--tasks-file",
                    stage["tasks_file"],
                    "--repeats-per-task",
                    str(args.repeats_per_task),
                    "--epochs",
                    str(args.distill_epochs),
                    "--max-length",
                    str(args.distill_max_length),
                    "--learning-rate",
                    str(args.distill_learning_rate),
                    "--train-batch-size",
                    str(args.distill_train_batch_size),
                    "--grad-accum-steps",
                    str(args.distill_grad_accum_steps),
                    "--output-dir",
                    str(distill_out_dir),
                    "--discord-min-interval-s",
                    str(args.discord_min_interval_s),
                ]
                if args.discord_webhook_url:
                    distill_cmd.extend(["--discord-webhook-url", args.discord_webhook_url])

                print(f"[overnight] stage={stage_name} {cycle_id} distill start", flush=True)
                if args.dry_run:
                    _append_command_log(cmd_log_md, distill_cmd)
                    distill_rc = 0
                else:
                    distill_rc = _run_command(
                        distill_cmd,
                        cwd=cwd,
                        env=env,
                        log_file=run_log_txt,
                        command_log=cmd_log_md,
                    )
                    candidate = distill_out_dir / "student" / "final_adapter"
                    if candidate.exists():
                        adapter_path = candidate
                        _copytree_replace(candidate, checkpoints_dir / "final_adapter")

            manifest_row = {
                "ts": time.time(),
                "stage": stage_name,
                "cycle": cycle,
                "campaign_rc": campaign_rc,
                "distill_rc": distill_rc,
                "campaign_dir": str(latest_campaign) if latest_campaign else "",
                "adapter_path": str(adapter_path) if adapter_path else "",
            }
            with manifest_jsonl.open("a", encoding="utf-8") as mf:
                mf.write(json.dumps(manifest_row, ensure_ascii=True) + "\n")

            print(
                f"[overnight] stage={stage_name} {cycle_id} done campaign_rc={campaign_rc} distill_rc={distill_rc}",
                flush=True,
            )
            if (campaign_rc != 0 or distill_rc != 0) and not args.continue_on_error:
                return 1

        stage_latest = stage_dir / "latest_checkpoint"
        last_cycle = stage_dir / f"cycle_{args.cycles_per_stage:02d}" / "checkpoints"
        if last_cycle.exists():
            _copytree_replace(last_cycle, stage_latest)

    print("[overnight] complete", flush=True)
    print(f"[overnight] artifacts: {run_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
