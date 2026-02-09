import json
from pathlib import Path

from tools.prime_rl_campaign import _load_tasks, build_aggregate, compute_verifier_reward


def test_compute_verifier_reward_shapes_expected():
    summary = {
        "success": True,
        "max_progress_ratio": 0.5,
        "replan_count": 2,
        "steps_taken": 400,
        "max_steps": 1200,
        "final_inventory_counts": {"diamond": 1},
        "final_seconds_since_progress": 30,
    }
    reward = compute_verifier_reward(summary)
    assert reward["success_reward"] > 0
    assert reward["diamond_bonus"] > 0
    assert reward["replan_penalty"] > 0
    assert reward["reward"] > 0


def test_build_aggregate_reports_rates():
    rows = [
        {"task": "obtain diamond", "success": True, "max_progress_ratio": 1.0, "steps_taken": 10, "reward": 1.0, "replan_count": 0},
        {"task": "obtain diamond", "success": False, "max_progress_ratio": 0.2, "steps_taken": 20, "reward": 0.1, "replan_count": 1},
        {"task": "obtain 3 diamonds safely", "success": False, "max_progress_ratio": 0.4, "steps_taken": 30, "reward": 0.2, "replan_count": 2},
    ]
    agg = build_aggregate(rows)
    assert agg["episodes"] == 3
    assert 0.0 <= agg["success_rate"] <= 1.0
    assert "task_success_rate" in agg
    assert "obtain diamond" in agg["task_success_rate"]


def test_load_tasks_plain_and_jsonl(tmp_path: Path):
    plain = tmp_path / "tasks.txt"
    plain.write_text("obtain diamond\nmine one diamond\n", encoding="utf-8")
    assert _load_tasks(str(plain)) == ["obtain diamond", "mine one diamond"]

    jsonl = tmp_path / "tasks.jsonl"
    jsonl.write_text(
        json.dumps({"task": "task one"}) + "\n" + json.dumps({"task": "task two"}) + "\n",
        encoding="utf-8",
    )
    assert _load_tasks(str(jsonl)) == ["task one", "task two"]
