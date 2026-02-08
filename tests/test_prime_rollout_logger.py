from tools.prime_rl_rollout_logger import _ensure_inventory_counts


def test_inventory_counts_prefers_structured_counts():
    state = {"inventory_counts": {"diamond": 2, "stick": "4"}, "inventory_summary": "diamond:1, stick:1"}
    counts = _ensure_inventory_counts(state)
    assert counts["diamond"] == 2
    assert counts["stick"] == 4


def test_inventory_counts_fallback_parses_summary():
    state = {"inventory_summary": "diamond:3, iron_ingot:7, invalid, weird:abc"}
    counts = _ensure_inventory_counts(state)
    assert counts == {"diamond": 3, "iron_ingot": 7}
