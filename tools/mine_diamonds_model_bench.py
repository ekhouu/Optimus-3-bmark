#!/usr/bin/env python3
"""Compare planning quality across OpenAI-compatible models for mine-diamonds.

This script:
- queries each model on a mine-diamonds planning task set
- scores responses with a deterministic rubric
- writes per-sample traces and per-model summaries
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import error, request


DEFAULT_TASKS = [
    "obtain diamond",
    "mine one diamond from scratch",
    "get a diamond pickaxe from scratch",
    "obtain 3 diamonds safely",
]

EXPECTED_ORDER = [
    "logs",
    "planks",
    "stick",
    "crafting_table",
    "wooden_pickaxe",
    "cobblestone",
    "furnace",
    "stone_pickaxe",
    "iron_ore",
    "iron_ingot",
    "iron_pickaxe",
    "diamond",
]


class DiscordNotifier:
    def __init__(self, webhook_url: str | None, min_interval_s: int = 20):
        self.webhook_url = webhook_url
        self.min_interval_s = min_interval_s
        self._last_sent = 0.0

    def send(self, content: str, force: bool = False) -> None:
        if not self.webhook_url:
            return
        now = time.time()
        if not force and (now - self._last_sent) < self.min_interval_s:
            return
        payload = {"content": content[:1900]}
        req = request.Request(
            url=self.webhook_url,
            method="POST",
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload).encode("utf-8"),
        )
        try:
            with request.urlopen(req, timeout=15):
                self._last_sent = now
        except Exception:
            pass


def _api_chat_completion(
    *,
    base_url: str,
    api_key: str | None,
    model: str,
    system_prompt: str,
    user_prompt: str,
    timeout_s: int,
) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 512,
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = request.Request(url=url, method="POST", headers=headers, data=json.dumps(payload).encode("utf-8"))
    try:
        with request.urlopen(req, timeout=timeout_s) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{model}: HTTP {exc.code} {details}") from exc
    message = body["choices"][0]["message"]["content"]
    if isinstance(message, str):
        return message
    if isinstance(message, list):
        chunks = []
        for item in message:
            if isinstance(item, dict) and item.get("type") == "text":
                chunks.append(item.get("text", ""))
        return "\n".join([c for c in chunks if c]).strip()
    return str(message)


def _extract_answer(content: str) -> str:
    match = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return content.strip()


def _extract_steps(content: str) -> list[str]:
    return [step.strip() for step in re.findall(r"step\s+\d+:\s*(.*?)(?=\nstep\s+\d+:|$)", content, re.DOTALL | re.IGNORECASE)]


def _extract_goal_items(steps: list[str]) -> list[str]:
    items: list[str] = []
    for step in steps:
        s = step.lower()
        if "log" in s:
            items.append("logs")
        if "plank" in s:
            items.append("planks")
        if "stick" in s:
            items.append("stick")
        if "crafting table" in s or "crafting_table" in s:
            items.append("crafting_table")
        if "wooden_pickaxe" in s or "wooden pickaxe" in s:
            items.append("wooden_pickaxe")
        if "cobblestone" in s:
            items.append("cobblestone")
        if "furnace" in s:
            items.append("furnace")
        if "stone_pickaxe" in s or "stone pickaxe" in s:
            items.append("stone_pickaxe")
        if "iron_ore" in s or "iron ore" in s:
            items.append("iron_ore")
        if "iron_ingot" in s or "iron ingot" in s:
            items.append("iron_ingot")
        if "iron_pickaxe" in s or "iron pickaxe" in s:
            items.append("iron_pickaxe")
        if "diamond" in s:
            items.append("diamond")
    return items


def _order_score(items: list[str]) -> float:
    seen = []
    for item in items:
        if item not in seen:
            seen.append(item)
    idx = 0
    for want in EXPECTED_ORDER:
        if idx < len(seen) and seen[idx] == want:
            idx += 1
        elif want in seen[idx:]:
            break
    return idx / len(EXPECTED_ORDER)


@dataclass
class RubricResult:
    score: float
    success: int
    milestone_coverage: float
    order_score: float
    format_score: float
    duplicate_penalty: float
    notes: str


def score_plan(raw_response: str) -> RubricResult:
    answer = _extract_answer(raw_response)
    steps = _extract_steps(answer)
    if not steps:
        return RubricResult(
            score=0.0,
            success=0,
            milestone_coverage=0.0,
            order_score=0.0,
            format_score=0.0,
            duplicate_penalty=0.2,
            notes="no_step_format",
        )

    items = _extract_goal_items(steps)
    unique_items = []
    for item in items:
        if item not in unique_items:
            unique_items.append(item)

    milestone_coverage = sum(1 for item in EXPECTED_ORDER if item in unique_items) / len(EXPECTED_ORDER)
    order = _order_score(items)
    format_score = 1.0 if "<answer>" in raw_response.lower() and "</answer>" in raw_response.lower() else 0.7

    duplicates = max(0, len(steps) - len(set(s.lower() for s in steps)))
    duplicate_penalty = min(0.25, duplicates * 0.03)

    success = 1 if "diamond" in unique_items and "iron_pickaxe" in unique_items else 0
    score = (
        0.55 * milestone_coverage
        + 0.20 * order
        + 0.15 * format_score
        + 0.10 * success
        - duplicate_penalty
    )
    score = max(0.0, min(1.0, score))
    notes = f"steps={len(steps)} items={len(unique_items)}"
    return RubricResult(
        score=score,
        success=success,
        milestone_coverage=milestone_coverage,
        order_score=order,
        format_score=format_score,
        duplicate_penalty=duplicate_penalty,
        notes=notes,
    )


def _load_tasks(path: str | None) -> list[str]:
    if not path:
        return DEFAULT_TASKS
    tasks: list[str] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("{"):
                tasks.append(json.loads(line)["task"])
            else:
                tasks.append(line)
    return tasks


def main() -> int:
    parser = argparse.ArgumentParser(description="Mine-diamonds planning benchmark for API models.")
    parser.add_argument("--base-url", default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY", ""))
    parser.add_argument("--models", required=True, help="Comma-separated model ids.")
    parser.add_argument("--tasks-file", default=None, help="Optional txt/jsonl task file.")
    parser.add_argument("--out-dir", default="outputs/mine_diamonds_bench")
    parser.add_argument("--timeout-s", type=int, default=120)
    parser.add_argument("--repeat", type=int, default=1, help="Number of runs per task per model.")
    parser.add_argument("--discord-webhook-url", default="")
    parser.add_argument("--discord-min-interval-s", type=int, default=20)
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    tasks = _load_tasks(args.tasks_file)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    traces_path = out_dir / f"traces_{ts}.jsonl"
    summary_path = out_dir / f"summary_{ts}.csv"
    notifier = DiscordNotifier(
        webhook_url=(args.discord_webhook_url or None),
        min_interval_s=args.discord_min_interval_s,
    )

    system_prompt = (
        "You are an expert Minecraft planner. Produce concise, executable plans."
    )
    user_template = (
        "Task: {task}\n"
        "Return a step-by-step plan.\n"
        "Requirements:\n"
        "- Wrap full output in <answer>...</answer>\n"
        "- Format each step exactly as: step N: <instruction>\n"
        "- Include quantities and key craft/smelt/mine milestones."
    )

    per_model_scores: dict[str, list[float]] = {m: [] for m in models}
    per_model_success: dict[str, list[int]] = {m: [] for m in models}
    total_cases = len(models) * len(tasks) * args.repeat
    done_cases = 0
    notifier.send(
        f"[bench] start models={len(models)} tasks={len(tasks)} repeat={args.repeat} total_cases={total_cases}",
        force=True,
    )

    with open(traces_path, "w", encoding="utf-8") as trace_file:
        for model in models:
            for task in tasks:
                for run_idx in range(args.repeat):
                    prompt = user_template.format(task=task)
                    try:
                        response = _api_chat_completion(
                            base_url=args.base_url,
                            api_key=args.api_key or None,
                            model=model,
                            system_prompt=system_prompt,
                            user_prompt=prompt,
                            timeout_s=args.timeout_s,
                        )
                        rubric = score_plan(response)
                        error_text = None
                    except Exception as exc:
                        response = ""
                        rubric = RubricResult(0.0, 0, 0.0, 0.0, 0.0, 0.0, "api_error")
                        error_text = str(exc)

                    per_model_scores[model].append(rubric.score)
                    per_model_success[model].append(rubric.success)
                    done_cases += 1

                    row = {
                        "model": model,
                        "task": task,
                        "run_idx": run_idx,
                        "score": rubric.score,
                        "success": rubric.success,
                        "milestone_coverage": rubric.milestone_coverage,
                        "order_score": rubric.order_score,
                        "format_score": rubric.format_score,
                        "duplicate_penalty": rubric.duplicate_penalty,
                        "notes": rubric.notes,
                        "response": response,
                        "error": error_text,
                    }
                    trace_file.write(json.dumps(row, ensure_ascii=True) + "\n")
                    trace_file.flush()
                    line = f"model={model} task={task} run={run_idx} score={rubric.score:.3f} success={rubric.success}"
                    print(line)
                    if done_cases % 5 == 0 or done_cases == total_cases:
                        notifier.send(f"[bench] {done_cases}/{total_cases} {line}")

    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model", "n", "mean_score", "stdev_score", "success_rate"],
        )
        writer.writeheader()
        for model in models:
            scores = per_model_scores[model]
            succ = per_model_success[model]
            row = {
                "model": model,
                "n": len(scores),
                "mean_score": round(statistics.mean(scores), 6) if scores else 0.0,
                "stdev_score": round(statistics.pstdev(scores), 6) if len(scores) > 1 else 0.0,
                "success_rate": round(sum(succ) / len(succ), 6) if succ else 0.0,
            }
            writer.writerow(row)

    print(f"Wrote traces: {traces_path}")
    print(f"Wrote summary: {summary_path}")
    notifier.send(f"[bench] complete summary={summary_path}", force=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
