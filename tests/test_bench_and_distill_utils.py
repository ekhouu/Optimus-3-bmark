import json
from pathlib import Path

from tools.distill_openai_to_open_model import _load_tasks, _render_chat
from tools.mine_diamonds_model_bench import _extract_answer, _extract_steps, score_plan


def test_extract_answer_and_steps():
    text = "<answer>\nstep 1: chop trees\nstep 2: craft planks\n</answer>"
    answer = _extract_answer(text)
    assert "step 1:" in answer
    steps = _extract_steps(answer)
    assert steps == ["chop trees", "craft planks"]


def test_score_plan_rewards_reasonable_diamond_plan():
    plan = """<answer>
step 1: chop trees to get 3 logs
step 2: craft 12 planks
step 3: craft 2 stick
step 4: craft 1 crafting_table
step 5: craft 1 wooden_pickaxe
step 6: mine 8 cobblestone
step 7: craft 1 furnace
step 8: craft 1 stone_pickaxe
step 9: mine 3 iron_ore
step 10: smelt 3 iron_ingot
step 11: craft 1 iron_pickaxe
step 12: mine 1 diamond
</answer>"""
    result = score_plan(plan)
    assert result.score > 0.5
    assert result.success == 1


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


def test_render_chat_format():
    rendered = _render_chat(
        [
            {"role": "system", "content": "rules"},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "world"},
        ]
    )
    assert "### System:" in rendered
    assert "### User:" in rendered
    assert "### Assistant:" in rendered
