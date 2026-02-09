#!/usr/bin/env python3
"""Run an Optimus rollout and log fine-grained progress for Prime-style verifier analysis.

Outputs (per run):
- events_*.jsonl  : full per-step event stream
- metrics_*.csv   : compact numeric timeseries
- summary_*.json  : final run summary
- progress_*.png  : optional graph (if matplotlib is installed)
"""

from __future__ import annotations

import argparse
import csv
import json
import mimetypes
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import error, request


def _http_json(method: str, url: str, payload: dict | None = None, timeout: int = 60) -> dict:
    data = None
    headers = {"Content-Type": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    req = request.Request(url=url, method=method, data=data, headers=headers)
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body) if body else {}
    except error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {url} failed: {exc.code} {details}") from exc


@dataclass
class DiscordNotifier:
    webhook_url: str | None
    min_interval_s: int = 30
    timeout_s: int = 15
    max_retries: int = 3
    retry_backoff_s: float = 1.5
    verbose: bool = False
    user_agent: str = "Optimus-3-PrimeCampaign/1.0"

    def __post_init__(self):
        self._last_sent = 0.0

    def _send_request(self, req: request.Request) -> bool:
        for attempt in range(1, self.max_retries + 1):
            try:
                with request.urlopen(req, timeout=self.timeout_s):
                    return True
            except Exception as exc:
                if self.verbose:
                    print(f"[discord] send failed attempt={attempt}/{self.max_retries}: {exc}", flush=True)
                if attempt < self.max_retries:
                    time.sleep(self.retry_backoff_s * attempt)
        return False

    def send(self, content: str, force: bool = False) -> bool:
        if not self.webhook_url:
            return False
        now = time.time()
        if not force and (now - self._last_sent) < self.min_interval_s:
            return False
        payload = {"content": content[:1900]}
        req = request.Request(
            url=self.webhook_url,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "User-Agent": self.user_agent,
            },
            data=json.dumps(payload).encode("utf-8"),
        )
        ok = self._send_request(req)
        if ok:
            self._last_sent = now
            if self.verbose:
                print("[discord] sent text message", flush=True)
        return ok

    def send_files(self, content: str, files: list[Path], force: bool = True) -> bool:
        if not self.webhook_url:
            return False
        now = time.time()
        if not force and (now - self._last_sent) < self.min_interval_s:
            return False

        boundary = f"----optimus-boundary-{uuid.uuid4().hex}"
        parts: list[bytes] = []

        payload_json = json.dumps({"content": content[:1800]})
        parts.append(f"--{boundary}\r\n".encode("utf-8"))
        parts.append(b'Content-Disposition: form-data; name="payload_json"\r\n')
        parts.append(b"Content-Type: application/json\r\n\r\n")
        parts.append(payload_json.encode("utf-8"))
        parts.append(b"\r\n")

        for idx, path in enumerate(files):
            if not path.exists() or not path.is_file():
                continue
            filename = path.name
            mime, _ = mimetypes.guess_type(filename)
            mime = mime or "application/octet-stream"
            parts.append(f"--{boundary}\r\n".encode("utf-8"))
            parts.append(
                f'Content-Disposition: form-data; name="file{idx}"; filename="{filename}"\r\n'.encode("utf-8")
            )
            parts.append(f"Content-Type: {mime}\r\n\r\n".encode("utf-8"))
            parts.append(path.read_bytes())
            parts.append(b"\r\n")

        parts.append(f"--{boundary}--\r\n".encode("utf-8"))
        body = b"".join(parts)

        req = request.Request(
            url=self.webhook_url,
            method="POST",
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "User-Agent": self.user_agent,
            },
            data=body,
        )
        ok = self._send_request(req)
        if ok:
            self._last_sent = now
            if self.verbose:
                print(f"[discord] sent file message files={len(files)}", flush=True)
        return ok


@dataclass
class RolloutConfig:
    base_url: str
    task: str
    out_dir: Path
    max_steps: int = 1200
    replan_threshold_seconds: int = 300
    step_sleep_seconds: float = 0.2
    device: str = "cuda:0"
    skip_reset: bool = False
    discord_webhook_url: str | None = None
    discord_min_interval_s: int = 30
    discord_timeout_s: int = 15
    discord_max_retries: int = 3
    discord_retry_backoff_s: float = 1.5
    discord_verbose: bool = False
    discord_send_run_artifacts: bool = False


def _ensure_inventory_counts(state: dict[str, Any]) -> dict[str, int]:
    counts = state.get("inventory_counts")
    if isinstance(counts, dict):
        out: dict[str, int] = {}
        for k, v in counts.items():
            try:
                out[str(k)] = int(v)
            except Exception:
                continue
        return out

    # Fallback: parse summary format "item:count, item2:count"
    summary = (state.get("inventory_summary") or "").strip()
    if not summary or summary == "empty":
        return {}
    out: dict[str, int] = {}
    for chunk in summary.split(","):
        chunk = chunk.strip()
        if ":" not in chunk:
            continue
        name, raw = chunk.split(":", 1)
        name = name.strip()
        raw = raw.strip()
        if not name:
            continue
        try:
            out[name] = int(raw)
        except ValueError:
            continue
    return out


def _write_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=True) + "\n")


def _plot_metrics(metrics_rows: list[dict[str, Any]], png_path: Path) -> bool:
    if not metrics_rows:
        return False
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False

    steps = [int(r["step"]) for r in metrics_rows]
    progress = [float(r["progress_ratio"]) for r in metrics_rows]
    diamond = [int(r["diamond_count"]) for r in metrics_rows]
    replans = [int(r["replanned"]) for r in metrics_rows]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(steps, progress, label="progress_ratio", color="#1f77b4", linewidth=2)
    axes[0].set_ylabel("Progress")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title("Mine-Diamond Rollout Progress")

    axes[1].plot(steps, diamond, label="diamond_count", color="#2ca02c", linewidth=2)
    axes[1].set_ylabel("Diamonds")
    axes[1].set_xlabel("Step")
    axes[1].grid(True, alpha=0.3)

    replan_x = [s for s, rp in zip(steps, replans) if rp > 0]
    if replan_x:
        y = [diamond[steps.index(x)] for x in replan_x]
        axes[1].scatter(replan_x, y, label="replan", color="#d62728", s=40, zorder=3)

    axes[0].legend(loc="best")
    axes[1].legend(loc="best")
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    return True


def run_rollout(cfg: RolloutConfig) -> int:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    events_path = cfg.out_dir / f"events_{ts}.jsonl"
    metrics_path = cfg.out_dir / f"metrics_{ts}.csv"
    summary_path = cfg.out_dir / f"summary_{ts}.json"
    plot_path = cfg.out_dir / f"progress_{ts}.png"

    notifier = DiscordNotifier(
        webhook_url=cfg.discord_webhook_url,
        min_interval_s=cfg.discord_min_interval_s,
        timeout_s=cfg.discord_timeout_s,
        max_retries=cfg.discord_max_retries,
        retry_backoff_s=cfg.discord_retry_backoff_s,
        verbose=cfg.discord_verbose,
    )

    print(
        f"[rollout] start task={cfg.task!r} max_steps={cfg.max_steps} "
        f"replan_threshold={cfg.replan_threshold_seconds}s out_dir={cfg.out_dir}",
        flush=True,
    )

    if not cfg.skip_reset:
        print(f"[rollout] reset device={cfg.device}", flush=True)
        _http_json("POST", f"{cfg.base_url}/reset", {"device": cfg.device})
    else:
        print("[rollout] skip_reset=true", flush=True)

    planning_payload = {"text": cfg.task, "task": "planning"}
    planning_resp = _http_json("POST", f"{cfg.base_url}/send_text", planning_payload)
    planning_text = planning_resp.get("response", "")
    print(f"[rollout] planning_response={planning_text[:220]!r}", flush=True)
    _write_jsonl(
        events_path,
        {
            "ts": time.time(),
            "phase": "planning",
            "request": planning_payload,
            "response": planning_text,
        },
    )
    notifier.send(f"[rollout] planning submitted task={cfg.task!r}", force=True)

    prev_sub_task_index = None
    prev_active_goal_item: str | None = None
    completed_goal_items: list[str] = []
    completed_goal_counts: dict[str, int] = {}
    first_diamond_step = None
    max_progress_ratio = 0.0
    replan_count = 0
    steps_taken = 0
    success = False
    final_state: dict[str, Any] = {}
    metrics_rows: list[dict[str, Any]] = []

    for step in range(cfg.max_steps):
        steps_taken = step + 1
        action_resp = _http_json("POST", f"{cfg.base_url}/send_text", {"text": "", "task": "action"})
        action_text = (action_resp.get("response") or "").strip().lower()
        state = _http_json("GET", f"{cfg.base_url}/plan_state")
        final_state = state

        sub_task_index = int(state.get("sub_task_index") or 0)
        plan_length = int(state.get("plan_length") or 0)
        progress_ratio = (sub_task_index / plan_length) if plan_length > 0 else 0.0
        max_progress_ratio = max(max_progress_ratio, progress_ratio)
        inventory_counts = _ensure_inventory_counts(state)
        diamond_count = int(inventory_counts.get("diamond", 0))
        if diamond_count > 0 and first_diamond_step is None:
            first_diamond_step = step

        did_progress = prev_sub_task_index is not None and sub_task_index > prev_sub_task_index
        prev_sub_task_index = sub_task_index
        completed_goal_item = None
        if did_progress and prev_active_goal_item:
            completed_goal_item = prev_active_goal_item
            completed_goal_items.append(completed_goal_item)
            completed_goal_counts[completed_goal_item] = completed_goal_counts.get(completed_goal_item, 0) + 1

        active_goal = state.get("active_goal")
        if isinstance(active_goal, dict) and active_goal.get("item"):
            prev_active_goal_item = str(active_goal.get("item"))
        elif sub_task_index >= plan_length:
            prev_active_goal_item = None

        sec_since_progress = state.get("seconds_since_progress")
        replanned = 0
        replan_detail = None

        if sec_since_progress is not None and float(sec_since_progress) >= cfg.replan_threshold_seconds:
            replan_resp = _http_json(
                "POST",
                f"{cfg.base_url}/maybe_replan",
                {"threshold_seconds": cfg.replan_threshold_seconds, "force": False},
            )
            if bool(replan_resp.get("replanned")):
                replanned = 1
                replan_count += 1
            replan_detail = replan_resp.get("detail")
            print(
                f"[rollout] replan_check step={step} replanned={bool(replanned)} detail={replan_detail!r}",
                flush=True,
            )

        event = {
            "ts": time.time(),
            "phase": "step",
            "step": step,
            "action_response": action_text,
            "sub_task_index": sub_task_index,
            "plan_length": plan_length,
            "progress_ratio": progress_ratio,
            "did_progress": bool(did_progress),
            "active_task": state.get("active_task"),
            "active_goal": active_goal,
            "completed_goal_item": completed_goal_item,
            "seconds_since_progress": sec_since_progress,
            "inventory_counts": inventory_counts,
            "diamond_count": diamond_count,
            "replanned": replanned,
            "replan_detail": replan_detail,
        }
        _write_jsonl(events_path, event)

        metrics_row = {
            "step": step,
            "sub_task_index": sub_task_index,
            "plan_length": plan_length,
            "progress_ratio": round(progress_ratio, 6),
            "diamond_count": diamond_count,
            "seconds_since_progress": sec_since_progress if sec_since_progress is not None else "",
            "replanned": replanned,
        }
        metrics_rows.append(metrics_row)

        if step % 20 == 0:
            print(
                f"[rollout] step={step} progress={progress_ratio:.3f} "
                f"sub_task={sub_task_index}/{plan_length} diamonds={diamond_count} replans={replan_count}",
                flush=True,
            )
            notifier.send(
                f"[rollout] step={step} progress={progress_ratio:.3f} diamonds={diamond_count} replans={replan_count}"
            )

        if action_text == "success":
            success = True
            break

        time.sleep(cfg.step_sleep_seconds)

    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "step",
                "sub_task_index",
                "plan_length",
                "progress_ratio",
                "diamond_count",
                "seconds_since_progress",
                "replanned",
            ],
        )
        writer.writeheader()
        writer.writerows(metrics_rows)

    plot_generated = _plot_metrics(metrics_rows, plot_path)
    summary = {
        "task": cfg.task,
        "success": success,
        "steps_taken": steps_taken,
        "max_steps": cfg.max_steps,
        "max_progress_ratio": round(max_progress_ratio, 6),
        "replan_count": replan_count,
        "first_diamond_step": first_diamond_step,
        "final_plan_length": int(final_state.get("plan_length") or 0),
        "final_sub_task_index": int(final_state.get("sub_task_index") or 0),
        "final_seconds_since_progress": final_state.get("seconds_since_progress"),
        "final_inventory_counts": _ensure_inventory_counts(final_state),
        "completed_goal_items": completed_goal_items,
        "completed_goal_counts": completed_goal_counts,
        "events_path": str(events_path),
        "metrics_path": str(metrics_path),
        "plot_path": str(plot_path) if plot_generated else None,
        "plot_generated": plot_generated,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    done_msg = (
        f"[rollout] done success={success} steps={steps_taken} "
        f"max_progress={max_progress_ratio:.3f} replans={replan_count}"
    )
    notifier.send(done_msg, force=True)
    if cfg.discord_send_run_artifacts:
        files = [summary_path, metrics_path]
        if plot_generated:
            files.append(plot_path)
        notifier.send_files(done_msg, files=files, force=True)

    print(json.dumps(summary, indent=2))
    return 0 if success else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prime-style rollout logger for Optimus mine-diamond runs.")
    parser.add_argument("--base-url", default="http://127.0.0.1:9500")
    parser.add_argument("--task", default="obtain diamond")
    parser.add_argument("--out-dir", default="outputs/prime_rollouts")
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--replan-threshold-seconds", type=int, default=300)
    parser.add_argument("--step-sleep-seconds", type=float, default=0.2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip-reset", action="store_true")
    parser.add_argument("--discord-webhook-url", default="")
    parser.add_argument("--discord-min-interval-s", type=int, default=30)
    parser.add_argument("--discord-timeout-s", type=int, default=15)
    parser.add_argument("--discord-max-retries", type=int, default=3)
    parser.add_argument("--discord-retry-backoff-s", type=float, default=1.5)
    parser.add_argument("--discord-verbose", action="store_true")
    parser.add_argument("--discord-send-run-artifacts", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = RolloutConfig(
        base_url=args.base_url.rstrip("/"),
        task=args.task,
        out_dir=Path(args.out_dir),
        max_steps=args.max_steps,
        replan_threshold_seconds=args.replan_threshold_seconds,
        step_sleep_seconds=args.step_sleep_seconds,
        device=args.device,
        skip_reset=args.skip_reset,
        discord_webhook_url=(args.discord_webhook_url or None),
        discord_min_interval_s=args.discord_min_interval_s,
        discord_timeout_s=args.discord_timeout_s,
        discord_max_retries=args.discord_max_retries,
        discord_retry_backoff_s=args.discord_retry_backoff_s,
        discord_verbose=args.discord_verbose,
        discord_send_run_artifacts=args.discord_send_run_artifacts,
    )
    return run_rollout(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
