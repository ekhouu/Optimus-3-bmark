import json
import sys
from pathlib import Path

from tools import export_teacher_sft


def test_export_teacher_sft_filters_and_formats(tmp_path: Path, monkeypatch):
    traces = tmp_path / "traces.jsonl"
    out = tmp_path / "out.jsonl"

    rows = [
        {
            "model": "teacher-a",
            "task": "obtain diamond",
            "score": 0.8,
            "success": 1,
            "milestone_coverage": 0.9,
            "order_score": 0.8,
            "response": "<answer>step 1: chop trees</answer>",
            "error": None,
        },
        {
            "model": "teacher-b",
            "task": "obtain diamond",
            "score": 0.2,
            "success": 0,
            "milestone_coverage": 0.1,
            "order_score": 0.1,
            "response": "bad",
            "error": None,
        },
    ]
    traces.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_teacher_sft.py",
            "--traces",
            str(traces),
            "--out",
            str(out),
            "--min-score",
            "0.7",
            "--require-success",
        ],
    )
    rc = export_teacher_sft.main()
    assert rc == 0
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    sample = json.loads(lines[0])
    assert sample["meta"]["source_model"] == "teacher-a"
    assert "How to obtain diamond from scratch?" in sample["messages"][1]["content"]
