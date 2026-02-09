#!/usr/bin/env python3
"""Run multi-episode Prime-style rollouts and export graphable verifier datasets.

Per campaign, outputs include:
- episodes/ep_*/... rollout artifacts from prime_rl_rollout_logger
- episodes.jsonl / episodes.csv
- prime_verifier_records.jsonl / prime_verifier_records.csv
- task_breakdown.csv
- aggregate.json
- campaign_progress.png
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Support both `python tools/prime_rl_campaign.py` and module imports.
if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from tools.prime_rl_rollout_logger import DiscordNotifier, RolloutConfig, run_rollout


def _load_tasks(tasks_file: str | None) -> list[str]:
    if not tasks_file:
        return ["obtain diamond"]
    tasks: list[str] = []
    with open(tasks_file, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith("{"):
                tasks.append(json.loads(line)["task"])
            else:
                tasks.append(line)
    return tasks


def _load_single_summary(episode_dir: Path) -> dict[str, Any]:
    summaries = sorted(episode_dir.glob("summary_*.json"))
    if not summaries:
        raise RuntimeError(f"No summary_*.json produced in {episode_dir}")
    return json.loads(summaries[-1].read_text(encoding="utf-8"))


def _required_diamond_count(task: str) -> int:
    text = (task or "").lower()
    if "diamond" not in text:
        return 1
    match = re.search(r"(\d+)\s+diamonds?", text)
    if match:
        try:
            return max(1, int(match.group(1)))
        except ValueError:
            return 1
    return 1


def compute_verifier_reward(summary: dict[str, Any], task: str, bonus_enabled: bool = True) -> dict[str, float]:
    success = 1.0 if bool(summary.get("success")) else 0.0
    max_progress = float(summary.get("max_progress_ratio", 0.0) or 0.0)
    replan_count = float(summary.get("replan_count", 0.0) or 0.0)
    steps_taken = float(summary.get("steps_taken", 0.0) or 0.0)
    max_steps = float(summary.get("max_steps", 1.0) or 1.0)
    final_inventory_counts = summary.get("final_inventory_counts") or {}
    completed_goal_items = set((summary.get("completed_goal_items") or []))
    required_diamonds = float(_required_diamond_count(task))
    diamond_count = float(final_inventory_counts.get("diamond", 0.0) or 0.0)
    final_plan_length = float(summary.get("final_plan_length", 0.0) or 0.0)
    final_sub_task_index = float(summary.get("final_sub_task_index", 0.0) or 0.0)
    final_seconds_since_progress = summary.get("final_seconds_since_progress")
    stalled = 1.0 if (final_seconds_since_progress is not None and float(final_seconds_since_progress) >= 120.0) else 0.0

    def _have_item(item: str) -> float:
        if float(final_inventory_counts.get(item, 0.0) or 0.0) > 0:
            return 1.0
        if item in completed_goal_items:
            return 1.0
        return 0.0

    m_crafting_table = _have_item("crafting_table")
    m_wooden_pickaxe = _have_item("wooden_pickaxe")
    m_stone_pickaxe = _have_item("stone_pickaxe")
    m_furnace = _have_item("furnace")
    m_iron = (
        1.0
        if (
            _have_item("iron_ore") > 0
            or _have_item("iron_ingot") > 0
            or _have_item("iron_pickaxe") > 0
            or float(final_inventory_counts.get("iron_ore", 0.0) or 0.0) > 0
            or float(final_inventory_counts.get("iron_ingot", 0.0) or 0.0) > 0
            or float(final_inventory_counts.get("iron_pickaxe", 0.0) or 0.0) > 0
        )
        else 0.0
    )
    m_goal_progress = (
        1.0 if success > 0 else (1.0 if (final_plan_length > 0 and (final_sub_task_index / final_plan_length) >= 0.80) else 0.0)
    )
    m_diamonds = 1.0 if diamond_count >= required_diamonds or _have_item("diamond") > 0 else 0.0

    milestone_values = [
        m_crafting_table,
        m_wooden_pickaxe,
        m_stone_pickaxe,
        m_iron,
        m_furnace,
        m_goal_progress,
        m_diamonds,
    ]
    milestone_score = sum(milestone_values) / len(milestone_values)

    b_iron_pickaxe = _have_item("iron_pickaxe")
    b_low_replans = 1.0 if replan_count <= 1.0 else 0.0
    b_efficiency = 1.0 if (steps_taken / max(max_steps, 1.0)) <= 0.70 else 0.0
    b_extra_diamonds = 1.0 if diamond_count >= (required_diamonds + 1.0) else 0.0
    bonus_values = [b_iron_pickaxe, b_low_replans, b_efficiency, b_extra_diamonds]
    bonus_score = (sum(bonus_values) / len(bonus_values)) if bonus_enabled else 0.0

    rubric_score = (0.85 * milestone_score + 0.15 * bonus_score) if bonus_enabled else milestone_score
    efficiency_ratio = steps_taken / max(max_steps, 1.0)

    components = {
        "required_diamonds": required_diamonds,
        "final_diamond_count": diamond_count,
        "m_crafting_table": m_crafting_table,
        "m_wooden_pickaxe": m_wooden_pickaxe,
        "m_stone_pickaxe": m_stone_pickaxe,
        "m_iron": m_iron,
        "m_furnace": m_furnace,
        "m_goal_progress": m_goal_progress,
        "m_diamonds": m_diamonds,
        "milestone_score": milestone_score,
        "b_iron_pickaxe": b_iron_pickaxe,
        "b_low_replans": b_low_replans,
        "b_efficiency": b_efficiency,
        "b_extra_diamonds": b_extra_diamonds,
        "bonus_score": bonus_score,
        "rubric_score": rubric_score,
        "success_reward": 1.10 * success,
        "progress_reward": 0.35 * max_progress,
        "rubric_reward": 0.90 * rubric_score,
        "efficiency_penalty": 0.08 * efficiency_ratio,
        "replan_penalty": 0.03 * replan_count,
        "stall_penalty": 0.12 * stalled,
    }
    reward = (
        components["success_reward"]
        + components["progress_reward"]
        + components["rubric_reward"]
        - components["efficiency_penalty"]
        - components["replan_penalty"]
        - components["stall_penalty"]
    )
    components["reward"] = round(reward, 6)
    return components


def build_aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    if n == 0:
        return {"episodes": 0}
    success_values = [1 if bool(r.get("success")) else 0 for r in rows]
    progress_values = [float(r.get("max_progress_ratio", 0.0) or 0.0) for r in rows]
    steps_values = [float(r.get("steps_taken", 0.0) or 0.0) for r in rows]
    reward_values = [float(r.get("reward", 0.0) or 0.0) for r in rows]
    rubric_values = [float(r.get("rubric_score", 0.0) or 0.0) for r in rows]
    milestone_values = [float(r.get("milestone_score", 0.0) or 0.0) for r in rows]
    bonus_values = [float(r.get("bonus_score", 0.0) or 0.0) for r in rows]
    replan_values = [float(r.get("replan_count", 0.0) or 0.0) for r in rows]
    per_task: dict[str, list[int]] = {}
    for row in rows:
        per_task.setdefault(row.get("task", "unknown"), []).append(1 if bool(row.get("success")) else 0)

    return {
        "episodes": n,
        "success_rate": round(sum(success_values) / n, 6),
        "mean_max_progress_ratio": round(statistics.mean(progress_values), 6),
        "mean_steps": round(statistics.mean(steps_values), 3),
        "mean_reward": round(statistics.mean(reward_values), 6),
        "mean_rubric_score": round(statistics.mean(rubric_values), 6),
        "mean_milestone_score": round(statistics.mean(milestone_values), 6),
        "mean_bonus_score": round(statistics.mean(bonus_values), 6),
        "mean_replans": round(statistics.mean(replan_values), 4),
        "task_success_rate": {task: round(sum(vals) / len(vals), 6) for task, vals in sorted(per_task.items())},
    }


def _plot_campaign(rows: list[dict[str, Any]], out_png: Path) -> bool:
    if not rows:
        return False
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False

    xs = list(range(1, len(rows) + 1))
    success = [1 if bool(r.get("success")) else 0 for r in rows]
    cumulative_success_rate: list[float] = []
    running = 0
    for i, s in enumerate(success, 1):
        running += s
        cumulative_success_rate.append(running / i)
    max_progress = [float(r.get("max_progress_ratio", 0.0) or 0.0) for r in rows]
    rewards = [float(r.get("reward", 0.0) or 0.0) for r in rows]

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(xs, cumulative_success_rate, color="#1f77b4", linewidth=2, label="cumulative_success_rate")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best")

    axes[1].plot(xs, max_progress, color="#2ca02c", linewidth=2, label="max_progress_ratio")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="best")

    axes[2].plot(xs, rewards, color="#d62728", linewidth=2, label="verifier_reward")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="best")
    axes[2].set_xlabel("Episode")

    fig.suptitle("Prime-Style Mine-Diamond Campaign")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return True


@dataclass
class CampaignConfig:
    base_url: str
    tasks: list[str]
    episodes: int
    out_dir: Path
    max_steps: int
    replan_threshold_seconds: int
    step_sleep_seconds: float
    device: str
    skip_reset_between: bool
    task_mode: str
    seed: int
    episode_cooldown_s: float
    continue_on_error: bool
    rubric_bonus_enabled: bool
    discord_webhook_url: str | None
    discord_min_interval_s: int
    discord_timeout_s: int
    discord_max_retries: int
    discord_retry_backoff_s: float
    discord_verbose: bool
    discord_episode_interval: int
    discord_artifact_interval: int
    discord_test_on_start: bool
    discord_send_episode_artifacts: bool
    discord_send_final_artifacts: bool


def run_campaign(cfg: CampaignConfig) -> int:
    random.seed(cfg.seed)
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = cfg.out_dir / f"campaign_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    episodes_root = run_dir / "episodes"
    episodes_root.mkdir(parents=True, exist_ok=True)

    episodes_jsonl = run_dir / "episodes.jsonl"
    episodes_csv = run_dir / "episodes.csv"
    prime_jsonl = run_dir / "prime_verifier_records.jsonl"
    prime_csv = run_dir / "prime_verifier_records.csv"
    aggregate_json = run_dir / "aggregate.json"
    task_breakdown_csv = run_dir / "task_breakdown.csv"
    campaign_plot_png = run_dir / "campaign_progress.png"

    notifier = DiscordNotifier(
        webhook_url=cfg.discord_webhook_url,
        min_interval_s=cfg.discord_min_interval_s,
        timeout_s=cfg.discord_timeout_s,
        max_retries=cfg.discord_max_retries,
        retry_backoff_s=cfg.discord_retry_backoff_s,
        verbose=cfg.discord_verbose,
    )

    if cfg.discord_test_on_start:
        notifier.send(f"[campaign] webhook test run_dir={run_dir}", force=True)

    print(f"[campaign] run_dir={run_dir}")
    print(f"[campaign] episodes={cfg.episodes} tasks={len(cfg.tasks)} task_mode={cfg.task_mode}")
    notifier.send(
        f"[campaign] start episodes={cfg.episodes} tasks={len(cfg.tasks)} mode={cfg.task_mode} run_dir={run_dir}",
        force=True,
    )

    row_fields = [
        "episode",
        "task",
        "success",
        "steps_taken",
        "max_steps",
        "max_progress_ratio",
        "replan_count",
        "first_diamond_step",
        "required_diamonds",
        "final_diamond_count",
        "m_crafting_table",
        "m_wooden_pickaxe",
        "m_stone_pickaxe",
        "m_iron",
        "m_furnace",
        "m_goal_progress",
        "m_diamonds",
        "milestone_score",
        "b_iron_pickaxe",
        "b_low_replans",
        "b_efficiency",
        "b_extra_diamonds",
        "bonus_score",
        "rubric_score",
        "reward",
        "success_reward",
        "progress_reward",
        "rubric_reward",
        "efficiency_penalty",
        "replan_penalty",
        "stall_penalty",
        "error",
        "episode_dir",
        "summary_path",
    ]
    rows: list[dict[str, Any]] = []

    with episodes_csv.open("w", newline="", encoding="utf-8") as episodes_f, prime_csv.open(
        "w", newline="", encoding="utf-8"
    ) as prime_f:
        episodes_writer = csv.DictWriter(episodes_f, fieldnames=row_fields)
        prime_writer = csv.DictWriter(prime_f, fieldnames=row_fields)
        episodes_writer.writeheader()
        prime_writer.writeheader()

        for episode_idx in range(1, cfg.episodes + 1):
            if cfg.task_mode == "random":
                task = random.choice(cfg.tasks)
            else:
                task = cfg.tasks[(episode_idx - 1) % len(cfg.tasks)]
            episode_dir = episodes_root / f"ep_{episode_idx:04d}"
            episode_dir.mkdir(parents=True, exist_ok=True)

            rollout_cfg = RolloutConfig(
                base_url=cfg.base_url,
                task=task,
                out_dir=episode_dir,
                max_steps=cfg.max_steps,
                replan_threshold_seconds=cfg.replan_threshold_seconds,
                step_sleep_seconds=cfg.step_sleep_seconds,
                device=cfg.device,
                skip_reset=(cfg.skip_reset_between and episode_idx > 1),
                discord_webhook_url=cfg.discord_webhook_url,
                discord_min_interval_s=cfg.discord_min_interval_s,
                discord_timeout_s=cfg.discord_timeout_s,
                discord_max_retries=cfg.discord_max_retries,
                discord_retry_backoff_s=cfg.discord_retry_backoff_s,
                discord_verbose=cfg.discord_verbose,
                discord_send_run_artifacts=cfg.discord_send_episode_artifacts,
            )

            error_text = ""
            try:
                print(f"[campaign] episode={episode_idx}/{cfg.episodes} task={task!r}")
                rc = run_rollout(rollout_cfg)
                summary = _load_single_summary(episode_dir)
                summary_path = sorted(episode_dir.glob("summary_*.json"))[-1]
            except Exception as exc:
                rc = 2
                error_text = str(exc)
                summary = {
                    "task": task,
                    "success": False,
                    "steps_taken": 0,
                    "max_steps": cfg.max_steps,
                    "max_progress_ratio": 0.0,
                    "replan_count": 0,
                    "first_diamond_step": None,
                    "final_seconds_since_progress": None,
                    "final_inventory_counts": {},
                }
                summary_path = Path("")
                print(f"[campaign] episode={episode_idx} error={error_text}")
                if not cfg.continue_on_error:
                    raise

            reward = compute_verifier_reward(summary, task=task, bonus_enabled=cfg.rubric_bonus_enabled)
            row = {
                "episode": episode_idx,
                "task": task,
                "success": bool(summary.get("success")),
                "steps_taken": int(summary.get("steps_taken", 0) or 0),
                "max_steps": int(summary.get("max_steps", cfg.max_steps) or cfg.max_steps),
                "max_progress_ratio": float(summary.get("max_progress_ratio", 0.0) or 0.0),
                "replan_count": int(summary.get("replan_count", 0) or 0),
                "first_diamond_step": summary.get("first_diamond_step"),
                "required_diamonds": reward["required_diamonds"],
                "final_diamond_count": reward["final_diamond_count"],
                "m_crafting_table": reward["m_crafting_table"],
                "m_wooden_pickaxe": reward["m_wooden_pickaxe"],
                "m_stone_pickaxe": reward["m_stone_pickaxe"],
                "m_iron": reward["m_iron"],
                "m_furnace": reward["m_furnace"],
                "m_goal_progress": reward["m_goal_progress"],
                "m_diamonds": reward["m_diamonds"],
                "milestone_score": reward["milestone_score"],
                "b_iron_pickaxe": reward["b_iron_pickaxe"],
                "b_low_replans": reward["b_low_replans"],
                "b_efficiency": reward["b_efficiency"],
                "b_extra_diamonds": reward["b_extra_diamonds"],
                "bonus_score": reward["bonus_score"],
                "rubric_score": reward["rubric_score"],
                "reward": reward["reward"],
                "success_reward": reward["success_reward"],
                "progress_reward": reward["progress_reward"],
                "rubric_reward": reward["rubric_reward"],
                "efficiency_penalty": reward["efficiency_penalty"],
                "replan_penalty": reward["replan_penalty"],
                "stall_penalty": reward["stall_penalty"],
                "error": error_text,
                "episode_dir": str(episode_dir),
                "summary_path": str(summary_path) if summary_path else "",
            }
            rows.append(row)
            episodes_writer.writerow(row)
            prime_writer.writerow(row)
            episodes_f.flush()
            prime_f.flush()

            with episodes_jsonl.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=True) + "\n")
            with prime_jsonl.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=True) + "\n")

            aggregate = build_aggregate(rows)
            aggregate_json.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
            _plot_campaign(rows, campaign_plot_png)

            msg = (
                f"[campaign] ep={episode_idx}/{cfg.episodes} success={row['success']} "
                f"progress={row['max_progress_ratio']:.3f} reward={row['reward']:.3f} "
                f"overall_success={aggregate.get('success_rate', 0.0):.3f}"
            )
            print(msg)

            if episode_idx % max(cfg.discord_episode_interval, 1) == 0:
                notifier.send(msg, force=True)
            if (
                cfg.discord_send_final_artifacts
                and episode_idx % max(cfg.discord_artifact_interval, 1) == 0
                and campaign_plot_png.exists()
            ):
                notifier.send_files(
                    f"[campaign] artifact checkpoint ep={episode_idx}/{cfg.episodes}",
                    files=[campaign_plot_png, aggregate_json],
                    force=True,
                )

            if cfg.episode_cooldown_s > 0:
                time.sleep(cfg.episode_cooldown_s)
            _ = rc

    aggregate = build_aggregate(rows)
    aggregate_json.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    with task_breakdown_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["task", "success_rate"])
        writer.writeheader()
        for task, rate in sorted((aggregate.get("task_success_rate") or {}).items()):
            writer.writerow({"task": task, "success_rate": rate})

    _plot_campaign(rows, campaign_plot_png)
    done_msg = (
        f"[campaign] done episodes={len(rows)} success_rate={aggregate.get('success_rate', 0.0):.3f} "
        f"mean_progress={aggregate.get('mean_max_progress_ratio', 0.0):.3f} "
        f"mean_reward={aggregate.get('mean_reward', 0.0):.3f}"
    )
    print(done_msg)
    notifier.send(done_msg, force=True)
    if cfg.discord_send_final_artifacts and campaign_plot_png.exists():
        notifier.send_files(done_msg, files=[campaign_plot_png, aggregate_json, episodes_csv], force=True)

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Big Prime-style rollout campaign runner.")
    parser.add_argument("--base-url", default="http://127.0.0.1:9500")
    parser.add_argument("--tasks-file", default="tools/mine_diamonds_tasks.txt")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--task-mode", choices=["random", "sequential"], default="random")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", default="/workspace/outputs/prime_campaigns")
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--replan-threshold-seconds", type=int, default=120)
    parser.add_argument("--step-sleep-seconds", type=float, default=0.4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip-reset-between", action="store_true")
    parser.add_argument("--episode-cooldown-s", type=float, default=0.0)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--disable-rubric-bonus", action="store_true")

    parser.add_argument("--discord-webhook-url", default="")
    parser.add_argument("--discord-min-interval-s", type=int, default=10)
    parser.add_argument("--discord-timeout-s", type=int, default=15)
    parser.add_argument("--discord-max-retries", type=int, default=4)
    parser.add_argument("--discord-retry-backoff-s", type=float, default=1.5)
    parser.add_argument("--discord-verbose", action="store_true")
    parser.add_argument("--discord-episode-interval", type=int, default=1)
    parser.add_argument("--discord-artifact-interval", type=int, default=10)
    parser.add_argument("--discord-test-on-start", action="store_true")
    parser.add_argument("--discord-send-episode-artifacts", action="store_true")
    parser.add_argument("--discord-send-final-artifacts", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tasks = _load_tasks(args.tasks_file)
    if not tasks:
        raise SystemExit(f"No tasks found in {args.tasks_file}")
    cfg = CampaignConfig(
        base_url=args.base_url.rstrip("/"),
        tasks=tasks,
        episodes=args.episodes,
        out_dir=Path(args.out_dir),
        max_steps=args.max_steps,
        replan_threshold_seconds=args.replan_threshold_seconds,
        step_sleep_seconds=args.step_sleep_seconds,
        device=args.device,
        skip_reset_between=args.skip_reset_between,
        task_mode=args.task_mode,
        seed=args.seed,
        episode_cooldown_s=args.episode_cooldown_s,
        continue_on_error=args.continue_on_error,
        rubric_bonus_enabled=(not args.disable_rubric_bonus),
        discord_webhook_url=(args.discord_webhook_url or None),
        discord_min_interval_s=args.discord_min_interval_s,
        discord_timeout_s=args.discord_timeout_s,
        discord_max_retries=args.discord_max_retries,
        discord_retry_backoff_s=args.discord_retry_backoff_s,
        discord_verbose=args.discord_verbose,
        discord_episode_interval=args.discord_episode_interval,
        discord_artifact_interval=args.discord_artifact_interval,
        discord_test_on_start=args.discord_test_on_start,
        discord_send_episode_artifacts=args.discord_send_episode_artifacts,
        discord_send_final_artifacts=args.discord_send_final_artifacts,
    )
    return run_campaign(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
