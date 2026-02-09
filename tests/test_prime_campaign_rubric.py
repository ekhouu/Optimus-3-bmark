from tools.prime_rl_campaign import compute_verifier_reward


def test_stone_rubric_profile_scores_stone_milestones():
    summary = {
        "success": False,
        "max_progress_ratio": 0.6,
        "replan_count": 1,
        "steps_taken": 220,
        "max_steps": 700,
        "final_plan_length": 10,
        "final_sub_task_index": 8,
        "final_seconds_since_progress": 30,
        "final_inventory_counts": {
            "oak_log": 3,
            "oak_planks": 8,
            "stick": 4,
            "crafting_table": 1,
            "wooden_pickaxe": 1,
            "cobblestone": 14,
            "stone_pickaxe": 1,
            "furnace": 1,
        },
        "completed_goal_items": ["logs", "planks", "stick", "cobblestone"],
    }
    reward = compute_verifier_reward(
        summary,
        task="orchestrate to obtain 8 cobblestone",
        bonus_enabled=True,
        rubric_profile="stone",
    )
    assert reward["rubric_profile"] == "stone"
    assert reward["target_item"] == "cobblestone"
    assert reward["required_target_count"] == 8.0
    assert reward["final_target_count"] >= 14.0
    assert reward["m_stone_goal"] == 1.0
    assert reward["m_cobblestone"] == 1.0
    assert reward["b_furnace_bonus"] == 1.0


def test_diamond_rubric_profile_keeps_diamond_target():
    summary = {
        "success": True,
        "max_progress_ratio": 1.0,
        "replan_count": 0,
        "steps_taken": 500,
        "max_steps": 1200,
        "final_plan_length": 12,
        "final_sub_task_index": 12,
        "final_seconds_since_progress": 5,
        "final_inventory_counts": {"diamond": 2, "iron_pickaxe": 1, "furnace": 1},
        "completed_goal_items": ["diamond", "iron_pickaxe"],
    }
    reward = compute_verifier_reward(
        summary,
        task="orchestrate to obtain 1 diamond",
        bonus_enabled=True,
        rubric_profile="diamond",
    )
    assert reward["rubric_profile"] == "diamond"
    assert reward["target_item"] == "diamond"
    assert reward["required_target_count"] == 1.0
    assert reward["m_diamonds"] == 1.0
    assert reward["b_extra_diamonds"] == 1.0


def test_iron_rubric_profile_scores_iron_target():
    summary = {
        "success": False,
        "max_progress_ratio": 0.7,
        "replan_count": 1,
        "steps_taken": 360,
        "max_steps": 900,
        "final_plan_length": 12,
        "final_sub_task_index": 9,
        "final_seconds_since_progress": 20,
        "final_inventory_counts": {
            "oak_log": 4,
            "oak_planks": 12,
            "stick": 6,
            "crafting_table": 1,
            "wooden_pickaxe": 1,
            "cobblestone": 18,
            "stone_pickaxe": 1,
            "furnace": 1,
            "iron_ore": 3,
            "iron_ingot": 3,
        },
        "completed_goal_items": ["logs", "planks", "stick", "furnace", "iron_ore", "iron_ingot"],
    }
    reward = compute_verifier_reward(
        summary,
        task="orchestrate to obtain 3 iron_ingot",
        bonus_enabled=True,
        rubric_profile="iron",
    )
    assert reward["rubric_profile"] == "iron"
    assert reward["target_item"] == "iron_ingot"
    assert reward["required_target_count"] == 3.0
    assert reward["final_target_count"] >= 3.0
    assert reward["m_iron_ore"] == 1.0
    assert reward["m_iron_ingot"] == 1.0
    assert reward["m_iron_goal"] == 1.0
