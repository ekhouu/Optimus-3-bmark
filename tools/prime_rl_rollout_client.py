#!/usr/bin/env python3
import argparse
import json
import time
from dataclasses import dataclass
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
class RolloutConfig:
    base_url: str
    task: str
    max_steps: int = 1200
    replan_threshold_seconds: int = 300
    step_sleep_seconds: float = 0.2
    device: str = "cuda:0"


def run_rollout(cfg: RolloutConfig) -> int:
    print(f"[reset] device={cfg.device}")
    _http_json("POST", f"{cfg.base_url}/reset", {"device": cfg.device})

    print(f"[planning] task={cfg.task}")
    planning_resp = _http_json(
        "POST",
        f"{cfg.base_url}/send_text",
        {"text": cfg.task, "task": "planning"},
    )
    print(f"[planning-response] {planning_resp.get('response', '')[:240]}")

    for step in range(cfg.max_steps):
        action_resp = _http_json(
            "POST",
            f"{cfg.base_url}/send_text",
            {"text": "", "task": "action"},
        )
        action_text = (action_resp.get("response") or "").strip().lower()
        state = _http_json("GET", f"{cfg.base_url}/plan_state")
        sec_since_progress = state.get("seconds_since_progress")
        idx = state.get("sub_task_index")
        length = state.get("plan_length")
        active_task = state.get("active_task")
        print(
            f"[step={step}] idx={idx}/{length} sec_since_progress={sec_since_progress} "
            f"active_task={active_task!r} action_resp={action_text!r}"
        )

        if action_text == "success":
            print("[done] success")
            return 0

        if sec_since_progress is not None and sec_since_progress >= cfg.replan_threshold_seconds:
            replan_resp = _http_json(
                "POST",
                f"{cfg.base_url}/maybe_replan",
                {"threshold_seconds": cfg.replan_threshold_seconds, "force": False},
            )
            print(f"[replan] {replan_resp}")

        time.sleep(cfg.step_sleep_seconds)

    print("[done] max steps reached")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Optimus rollout loop for Prime-RL style control.")
    parser.add_argument("--base-url", default="http://127.0.0.1:9500")
    parser.add_argument("--task", required=True)
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--replan-threshold-seconds", type=int, default=300)
    parser.add_argument("--step-sleep-seconds", type=float, default=0.2)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    cfg = RolloutConfig(
        base_url=args.base_url.rstrip("/"),
        task=args.task,
        max_steps=args.max_steps,
        replan_threshold_seconds=args.replan_threshold_seconds,
        step_sleep_seconds=args.step_sleep_seconds,
        device=args.device,
    )
    return run_rollout(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
