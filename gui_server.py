import asyncio
import base64
import json
import logging
import os
import random
import re
import time
import uuid
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from minestudio.models import CraftWorker, EquipWorker, SmeltWorker, load_steve_one_policy  # noqa
from minestudio.simulator import MinecraftSim
from minestudio.simulator.callbacks.commands import CommandsCallback
from PIL import Image
from pydantic import BaseModel

from minecraftoptimus.model.agent.optimus3 import Optimus3Agent,check_inventory


paused = False
connected_clients: List[WebSocket] = []
# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="MinecraftOptimus API")

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You might want to restrict this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Define request/response models
class ObservationData(BaseModel):
    pov: List[List[List[int]]]  # 3D array for image data
    inventory: Optional[Dict[str, int]] = None
    other_data: Optional[Dict[str, Any]] = None


class ResetData(BaseModel):
    device: str


class TextData(BaseModel):
    text: str | None
    task: str


class MaybeReplanData(BaseModel):
    threshold_seconds: int = 300
    force: bool = False


# Global variables to store environment and model state
env = None
helper = None
model = None
current_obs = None
current_info = None
last_action = None
session_start_time = None
session_id = None
sub_tasks = None
goals = None
sub_task_index = 0
look_down_once = False
log_count = 0
iron_ore_count = 0
golden_ore_count = 0
diamond_ore_count = 0
redstone_ore_count = 0
planning_query = None
last_goal_completion_ts = None
last_replan_ts = None
orchestrator_mode = False
orchestrator_objective = None


REPO_ROOT = Path(__file__).resolve().parent


def _strip_env_value(raw: str) -> str:
    value = raw.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _load_local_env(env_path: Path) -> None:
    """Minimal .env loader to avoid requiring python-dotenv."""
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, raw_val = stripped.split("=", 1)
        key = key.strip()
        if not key:
            continue
        os.environ.setdefault(key, _strip_env_value(raw_val))


def _env_or_default(name: str, default: str) -> str:
    value = (os.getenv(name) or "").strip()
    return value if value else default


_load_local_env(REPO_ROOT / ".env")

ACTION_HEAD_CKPT_PATH = _env_or_default(
    "OPTIMUS_ACTION_HEAD_CKPT_PATH", "/workspace/Optimus-3-bmark/checkpoint/Optimus-3-ActionHead"
)
OPTIMUS3_CKPT_PATH = _env_or_default("OPTIMUS_MLLM_CKPT_PATH", "/workspace/Optimus-3-bmark/checkpoint/Optimus-3")
TASK_ROUTER_CKPT_PATH = _env_or_default(
    "OPTIMUS_TASK_ROUTER_CKPT_PATH", "/workspace/Optimus-3-bmark/checkpoint/Optimus-3-Task-Router"
)
AUTO_REPLAN_ENABLED = _env_or_default("OPTIMUS_AUTO_REPLAN", "0") == "1"
REPLAN_NO_PROGRESS_SECONDS = int(_env_or_default("OPTIMUS_REPLAN_NO_PROGRESS_SECONDS", "300"))
REPLAN_MIN_INTERVAL_SECONDS = int(_env_or_default("OPTIMUS_REPLAN_MIN_INTERVAL_SECONDS", "30"))
REPLAN_ON_GOAL_COMPLETION = _env_or_default("OPTIMUS_REPLAN_ON_GOAL_COMPLETION", "0") == "1"
ORCHESTRATOR_NO_PROGRESS_SECONDS = int(_env_or_default("OPTIMUS_ORCHESTRATOR_NO_PROGRESS_SECONDS", "120"))
ORCHESTRATOR_REPLAN_ON_GOAL_COMPLETION = _env_or_default("OPTIMUS_ORCHESTRATOR_REPLAN_ON_GOAL_COMPLETION", "1") == "1"
ORCHESTRATOR_FORCE_REPLAN_ON_PLAN_END = _env_or_default("OPTIMUS_ORCHESTRATOR_FORCE_REPLAN_ON_PLAN_END", "1") == "1"
RUN_LOG_DIR = Path(_env_or_default("OPTIMUS_RUN_LOG_DIR", str(REPO_ROOT / "outputs" / "server_traces")))
LLM_TRACE_LOG_PATH = Path(_env_or_default("OPTIMUS_LLM_TRACE_LOG", str(RUN_LOG_DIR / "llm_trace.jsonl")))


def _append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")
    except Exception:
        logger.exception("Failed writing trace row to %s", path)


def _log_llm_event(event_type: str, request_payload: Dict[str, Any], response_text: str, extra: Dict[str, Any] | None = None):
    row: Dict[str, Any] = {
        "ts": time.time(),
        "event_type": event_type,
        "request": request_payload,
        "response": response_text,
        "session_id": session_id,
    }
    if extra:
        row["extra"] = extra
    _append_jsonl(LLM_TRACE_LOG_PATH, row)


def ndarray_to_base64(arr: np.ndarray) -> str:
    """
    Converts a numpy ndarray (HWC, uint8) to a base64-encoded PNG string.
    """
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    img = Image.fromarray(arr)
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def _inventory_aggregate(info: Dict[str, Any] | None) -> Dict[str, int]:
    aggregate: Dict[str, int] = {}
    if not info or "inventory" not in info:
        return aggregate
    inventory = info.get("inventory") or {}
    for slot in inventory.values():
        item_type = slot.get("type")
        quantity = slot.get("quantity", 0)
        if not item_type or item_type == "none" or quantity <= 0:
            continue
        aggregate[item_type] = aggregate.get(item_type, 0) + int(quantity)
    return aggregate


def _inventory_summary_text(info: Dict[str, Any] | None, max_items: int = 20) -> str:
    aggregate = _inventory_aggregate(info)
    if not aggregate:
        return "empty"
    parts = [f"{name}:{count}" for name, count in sorted(aggregate.items(), key=lambda x: (-x[1], x[0]))[:max_items]]
    return ", ".join(parts)


def _count_inventory_like(aggregate: Dict[str, int], needle: str) -> int:
    needle = needle.lower()
    return sum(count for name, count in aggregate.items() if needle in name.lower())


def _required_count_for_craft_prereq(item: str, count: int, aggregate: Dict[str, int]) -> tuple[str, int] | None:
    item = item.lower()
    if item == "planks":
        need_logs = (count + 3) // 4
        have_logs = _count_inventory_like(aggregate, "log")
        if have_logs < need_logs:
            return ("logs", need_logs)
        return None

    if item == "stick":
        need_planks = (count + 1) // 2
        have_planks = _count_inventory_like(aggregate, "planks")
        if have_planks < need_planks:
            deficit_planks = need_planks - have_planks
            need_logs = (deficit_planks + 3) // 4
            have_logs = _count_inventory_like(aggregate, "log")
            if have_logs < need_logs:
                return ("logs", need_logs)
            return ("planks", need_planks)
        return None

    if item == "crafting_table":
        need_planks = 4
        have_planks = _count_inventory_like(aggregate, "planks")
        if have_planks < need_planks:
            deficit_planks = need_planks - have_planks
            need_logs = (deficit_planks + 3) // 4
            have_logs = _count_inventory_like(aggregate, "log")
            if have_logs < need_logs:
                return ("logs", need_logs)
            return ("planks", need_planks)
        return None

    if item == "wooden_pickaxe":
        need_planks = 3
        need_stick = 2
        have_stick = _count_inventory_like(aggregate, "stick")
        if have_stick < need_stick:
            return ("stick", need_stick)
        have_planks = _count_inventory_like(aggregate, "planks")
        if have_planks < need_planks:
            return ("planks", need_planks)
        return None

    if item == "stone_pickaxe":
        need_cobble = 3
        need_stick = 2
        have_stick = _count_inventory_like(aggregate, "stick")
        if have_stick < need_stick:
            return ("stick", need_stick)
        have_cobble = _count_inventory_like(aggregate, "cobblestone")
        if have_cobble < need_cobble:
            return ("cobblestone", need_cobble)
        return None

    if item == "furnace":
        need_cobble = 8
        have_cobble = _count_inventory_like(aggregate, "cobblestone")
        if have_cobble < need_cobble:
            return ("cobblestone", need_cobble)
        return None

    if item == "iron_pickaxe":
        need_iron = 3
        need_stick = 2
        have_stick = _count_inventory_like(aggregate, "stick")
        if have_stick < need_stick:
            return ("stick", need_stick)
        have_iron = _count_inventory_like(aggregate, "iron_ingot")
        if have_iron < need_iron:
            return ("iron_ingot", need_iron)
        return None

    return None


def _prereq_task(goal_item: str, goal_count: int) -> tuple[str, Dict[str, Any]]:
    if goal_item == "logs":
        return f"chop trees to get {goal_count} logs", {"step": 0, "count": goal_count, "item": "logs"}
    if goal_item == "planks":
        return f"craft {goal_count} planks", {"step": 0, "count": goal_count, "item": "planks"}
    if goal_item == "stick":
        return f"craft {goal_count} stick", {"step": 0, "count": goal_count, "item": "stick"}
    if goal_item == "cobblestone":
        return f"dig down and mine {goal_count} cobblestone", {"step": 0, "count": goal_count, "item": "cobblestone"}
    if goal_item == "iron_ingot":
        return f"smelt {goal_count} iron_ingot", {"step": 0, "count": goal_count, "item": "iron_ingot"}
    return f"obtain {goal_count} {goal_item}", {"step": 0, "count": goal_count, "item": goal_item}


def _maybe_redirect_craft_goal(task_text: str, goal: Dict[str, Any], info: Dict[str, Any] | None) -> tuple[str, Dict[str, Any]] | None:
    if "craft" not in task_text and "smelt" not in task_text:
        return None
    aggregate = _inventory_aggregate(info)
    item = str(goal.get("item", "")).strip().lower()
    count = int(goal.get("count", 1))
    prereq = _required_count_for_craft_prereq(item, count, aggregate)
    if prereq is None:
        return None
    prereq_item, prereq_count = prereq
    return _prereq_task(prereq_item, prereq_count)


def _state_context_text(info: Dict[str, Any] | None) -> str:
    if not info:
        return ""
    inventory_text = _inventory_summary_text(info)
    pos = info.get("player_pos") or {}
    if pos:
        return (
            f"inventory={inventory_text}; "
            f"position=({pos.get('x', 0):.2f},{pos.get('y', 0):.2f},{pos.get('z', 0):.2f})"
        )
    return f"inventory={inventory_text}"


def _strip_orchestrate_prefix(text: str) -> str:
    raw = text.strip()
    lowered = raw.lower()
    for prefix in ("orchestrate to ", "orchestrate:", "orchestrate "):
        if lowered.startswith(prefix):
            cleaned = raw[len(prefix) :].strip()
            if cleaned:
                return cleaned
    return raw


def _infer_orchestrator_objective(task_text: str) -> Dict[str, Any] | None:
    lower = task_text.lower().strip()
    item = None
    count = 1
    count_item_match = re.search(r"\b(\d+)\s+([a-z_]+)\b", lower)
    if count_item_match:
        count = max(1, int(count_item_match.group(1)))
        item = count_item_match.group(2).strip().lower()

    if item is None:
        for token in (
            "diamond",
            "diamonds",
            "iron_ingot",
            "iron_ore",
            "furnace",
            "stone_pickaxe",
            "wooden_pickaxe",
            "crafting_table",
            "redstone",
            "gold_ingot",
            "gold_ore",
            "cobblestone",
            "logs",
            "log",
            "planks",
            "stick",
        ):
            if token in lower:
                item = token
                break
    if not item:
        return None
    if item == "diamonds":
        item = "diamond"
    return {"item": item, "count": count, "raw": task_text}


def _objective_progress(info: Dict[str, Any] | None, objective: Dict[str, Any] | None) -> tuple[int, int]:
    if not objective:
        return 0, 0
    target_item = str(objective.get("item", "")).strip().lower()
    target_count = max(1, int(objective.get("count", 1)))
    aggregate = _inventory_aggregate(info)
    singular = target_item[:-1] if target_item.endswith("s") else target_item
    have_count = max(_count_inventory_like(aggregate, target_item), _count_inventory_like(aggregate, singular))
    return have_count, target_count


def _objective_met(info: Dict[str, Any] | None, objective: Dict[str, Any] | None) -> bool:
    have_count, target_count = _objective_progress(info, objective)
    return target_count > 0 and have_count >= target_count


def _plan_length() -> int:
    if not sub_tasks or not goals:
        return 0
    return min(len(sub_tasks), len(goals))


def _try_replan(trigger: str, force: bool = False, threshold_seconds: int | None = None) -> tuple[bool, str]:
    global sub_tasks, goals, sub_task_index, planning_query, last_goal_completion_ts, last_replan_ts

    if model is None:
        return False, "model_not_initialized"
    if not planning_query:
        return False, "missing_planning_query"

    now = time.time()
    threshold = REPLAN_NO_PROGRESS_SECONDS if threshold_seconds is None else threshold_seconds
    if not force:
        if last_replan_ts is not None and (now - last_replan_ts) < REPLAN_MIN_INTERVAL_SECONDS:
            return False, "replan_cooldown"
        if last_goal_completion_ts is not None and trigger == "no_progress":
            if (now - last_goal_completion_ts) < threshold:
                return False, "below_no_progress_threshold"

    state_context = _state_context_text(current_info)
    response_text, _sub_plans, _goals = model.plan(planning_query, state_context=state_context, from_scratch=False)
    if not _sub_plans or not _goals:
        return False, "empty_replan"
    _log_llm_event(
        "replan",
        {
            "planning_query": planning_query,
            "trigger": trigger,
            "force": force,
            "state_context": state_context,
            "threshold_seconds": threshold,
        },
        response_text,
        {"plan_length": min(len(_sub_plans), len(_goals))},
    )
    sub_tasks = _sub_plans
    goals = _goals
    sub_task_index = 0
    model.task = None
    last_replan_ts = now
    last_goal_completion_ts = now
    logger.info("Replanned (%s): %d tasks", trigger, _plan_length())
    return True, response_text


@app.get("/gpu")
async def check_gpu():
    """
    Checks if GPUs are available and returns their name, total memory, and used memory.
    """
    gpu_available = torch.cuda.is_available()
    gpus = []
    if gpu_available:
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            total_mem = props.total_memory // (1024 * 1024)
            # Get used memory via torch.cuda.memory_allocated
            used_mem = torch.cuda.memory_allocated(i) // (1024 * 1024)
            gpus.append(
                {
                    "name": props.name,
                    "total_memory_MB": total_mem,
                    "used_memory_MB": used_mem,
                }
            )
    return {"gpu_available": gpu_available, "gpus": gpus}





async def broadcast_obs(base64_png: str):
    """
    Broadcasts the latest observation to all connected WebSocket clients.
    If paused==True, the broadcast is skipped.
    """

    if paused:
        return

    disconnected = []
    for ws in connected_clients:
        try:
            await ws.send_text(base64_png)
        except WebSocketDisconnect:
            disconnected.append(ws)
    for ws in disconnected:
        if ws in connected_clients:
            connected_clients.remove(ws)


@app.post("/pause")
async def pause_agent():

    global paused
    paused = True
    return {"status": "paused"}


@app.post("/resume")
async def resume_agent():

    global paused
    paused = False
    return {"status": "running"}


@app.websocket("/ws/obs")
async def websocket_observations(websocket: WebSocket):

    await websocket.accept()
    connected_clients.append(websocket)
    try:
        
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        if websocket in connected_clients:
            connected_clients.remove(websocket)


@app.on_event("startup")
async def startup_event():
    """ """
    global session_start_time, session_id

    session_start_time = datetime.now()
    session_id = str(uuid.uuid4())

    gpu_info = await check_gpu()
    default_device = "cuda:0"

    if gpu_info.get("gpu_available") and gpu_info.get("gpus"):
        for idx, gpu in enumerate(gpu_info.get("gpus")):
            free_mem = gpu.get("total_memory_MB") - gpu.get("used_memory_MB")
            if free_mem >= 40000:  # 40GB
                default_device = f"cuda:{idx}"
                logger.info(f"choose GPU{idx}, free memory {free_mem}MB, device: {default_device}")
                break
        else:
            logger.info("No GPU with >= 40GB of available memory found, use CPU")
    else:
        logger.info("GPU not available, use CPU")

    await reset(ResetData(device=default_device))


@app.post("/reset")
async def reset(reset_data: ResetData):
    """
    Initializes or resets the MinecRL environment and loads the model.
    """
    global env, model, current_obs, session_start_time, session_id, helper, current_info
    global sub_tasks, goals, sub_task_index, planning_query, last_goal_completion_ts, last_replan_ts
    global orchestrator_mode, orchestrator_objective

    try:
        # Close existing environment if one exists
        if env:
            env.close()

        logger.info("Initializing environment")
        env = MinecraftSim(
            obs_size=(128, 128),
            preferred_spawn_biome="forest",
            callbacks=[
                CommandsCallback(
                    [
                        "/gamerule sendCommandFeedback false",
                        "/gamerule commandBlockOutput false",
                        "/gamerule keepInventory true",
                        "/effect give @a night_vision 99999 250 true",
                        "/gamerule doDaylightCycle false",
                        "/time set 0",
                        "/gamerule doImmediateRespawn true",
                        "/spawnpoint",
                    ]
                ),
            ],
            seed=random.randint(1, 100000000),
        )

        # Reset the environment to get initial observation
        current_obs, current_info = env.reset()
        sub_tasks = None
        goals = None
        sub_task_index = 0
        planning_query = None
        last_goal_completion_ts = None
        last_replan_ts = None
        orchestrator_mode = False
        orchestrator_objective = None
        helper = {"craft": CraftWorker(env), "smelt": SmeltWorker(env), "equip": EquipWorker(env)}
        if not model:
            logger.info(
                "Loading checkpoints | action_head=%s | optimus3=%s | task_router=%s",
                ACTION_HEAD_CKPT_PATH,
                OPTIMUS3_CKPT_PATH,
                TASK_ROUTER_CKPT_PATH,
            )
            model = Optimus3Agent(
                ACTION_HEAD_CKPT_PATH,
                OPTIMUS3_CKPT_PATH,
                TASK_ROUTER_CKPT_PATH,
                device=reset_data.device,
            )
        obs_b64 = ndarray_to_base64(current_info["pov"])
        import asyncio

        asyncio.create_task(broadcast_obs(obs_b64))
        return {"status": "success", "observation": obs_b64}
        

    except Exception as e:
        logger.error(f"Error during reset: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to initialize: {str(e)}")


def _step(env, agent, obs, task, goal, helper):
    global look_down_once, log_count, iron_ore_count, golden_ore_count, diamond_ore_count, redstone_ore_count

    while paused:
        time.sleep(0.05)
    if "craft" in task:
        helper["craft"].step_hook = lambda info: asyncio.create_task(broadcast_obs(ndarray_to_base64(info["pov"])))
        result, _ = helper["craft"].crafting(goal["item"], goal["count"])
        action = env.env.noop_action()

        pickaxe = env.find_best_pickaxe()
        if pickaxe:
            helper["equip"].equip_item(pickaxe)
        obs, reward, terminated, truncated, info = env.step(action)

    elif "smelt" in task:
        helper["smelt"].step_hook = lambda info: asyncio.create_task(broadcast_obs(ndarray_to_base64(info["pov"])))
        result, _ = helper["smelt"].smelting(goal["item"], goal["count"])
        obs, reward, terminated, truncated, info = env.step(env.env.noop_action())
    else:
        env._only_once = True
        action, memory = agent.get_action(obs, task)
        action = env.agent_action_to_env_action(action)
        action["drop"] = np.array(0)
        action["inventory"] = np.array(0)
        action["use"] = np.array(0)
        for i in range(9):
            action[f"hotbar.{i + 1}"] = np.array(0)

        if "dig down" in task:
            action["jump"] = action["left"] = action["right"] = np.array(0)
            action["sneak"] = action["sprint"] = np.array(0)
            if not look_down_once:
                pickaxe = env.find_best_pickaxe()
                helper["equip"].equip_item(pickaxe)
                helper["craft"]._look_down()
                look_down_once = True
            action["attack"] = np.array(1)
        
        if action["attack"] > 0:
            action["jump"] = action["left"] = action["right"] = np.array(0)
            action["sneak"] = action["sprint"] = np.array(0)

        obs, reward, terminated, truncated, info = env.step(action)

    check, count = check_inventory(info["inventory"], goal["item"], goal["count"])
    if check:
        if goal["item"] == "logs":
            log_count = count
        elif goal["item"] == "iron_ore":
            iron_ore_count = count
        elif goal["item"] == "gold_ore":
            golden_ore_count = count
        elif goal["item"] == "diamond":
            diamond_ore_count = count
        elif goal["item"] == "redstone":
            redstone_ore_count = count
        look_down_once = False
    return obs, info, check


@app.get("/get_obs")
async def get_obs():
    global current_obs
    if not env or not current_obs:
        raise HTTPException(status_code=400, detail="Environment not initialized or no observation.")
    try:
        obs_b64 = ndarray_to_base64(current_info["pov"])

        asyncio.create_task(broadcast_obs(obs_b64))
        
    except Exception as e:
        logger.error(f"Error returning observation: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to return observation: {str(e)}")


@app.get("/initial_text")
async def initial_text():
    initial_text = """Hello! I'm Optimus-3, your Minecraft agent. I can help you with task planning, action execution, and visual perception in Minecraft (including captioning, embodied question answering, and grounding). Let's embark on an exciting journey of exploration in Minecraft!
     """
    return {
        "status": "success",
        "text": initial_text,
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/send_text")
async def send_text(text_data: TextData):
    """
    Processes a text command and returns a response.
    """
    global env, model, current_obs, sub_tasks, goals, sub_task_index, last_action, current_info
    global planning_query, last_goal_completion_ts, last_replan_ts
    global orchestrator_mode, orchestrator_objective
    if not env or model is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")

    try:
       
        user_text = text_data.text.strip() if text_data.text else ""
        task_type = text_data.task.strip()
        logger.info(f"Received text '{user_text}' with task '{task_type}'")
        if current_info is None or "pov" not in current_info:
            raise HTTPException(status_code=400, detail="No current observation. Call /reset first.")

        img = Image.fromarray(current_info["pov"])
        
        if "help" in user_text:
            response_text = """
            Available commands:
            [Planning] [Captioning] [Embodied QA] [Grounding] [Long-horizon Action]
            """
        else:
            with torch.no_grad():
                print(f"Task type: {task_type}")
                is_orchestrate_request = (task_type == "orchestrate") or user_text.lower().startswith("orchestrate")
                if task_type == "planning" or task_type == "orchestrate":
                    planning_text = _strip_orchestrate_prefix(user_text) if is_orchestrate_request else user_text
                    planning_text = planning_text.strip() if planning_text else user_text
                    planning_query = planning_text
                    orchestrator_mode = is_orchestrate_request
                    orchestrator_objective = (
                        _infer_orchestrator_objective(planning_text) if orchestrator_mode else None
                    )
                    inv_agg = _inventory_aggregate(current_info)
                    from_scratch = len(inv_agg) == 0
                    state_context = _state_context_text(current_info) if not from_scratch else None
                    response_text, _sub_plans, _goals = model.plan(
                        planning_text,
                        state_context=state_context,
                        from_scratch=from_scratch,
                    )
                    _log_llm_event(
                        "orchestration_plan" if orchestrator_mode else "planning",
                        {
                            "planning_query": planning_text,
                            "from_scratch": from_scratch,
                            "state_context": state_context,
                            "orchestrator_mode": orchestrator_mode,
                            "orchestrator_objective": orchestrator_objective,
                        },
                        response_text,
                        {"plan_length": min(len(_sub_plans), len(_goals))},
                    )
                    sub_tasks = _sub_plans
                    goals = _goals
                    sub_task_index = 0
                    model.task = None
                    now = time.time()
                    last_goal_completion_ts = now
                    last_replan_ts = None
                elif task_type == "captioning" or task_type == "embodied_qa":
                    response_text = model.answer(user_text, img)
                elif task_type == "action":
                    if not sub_tasks or not goals:
                        if orchestrator_mode and planning_query:
                            replanned, _ = _try_replan("orchestrator_bootstrap", force=True)
                            if replanned and _plan_length() > 0:
                                response_text = sub_tasks[sub_task_index]
                            else:
                                response_text = "No active plan. Run planning first."
                        else:
                            response_text = "No active plan. Run planning first."
                    else:
                        plan_length = min(len(sub_tasks), len(goals))
                        if plan_length == 0:
                            response_text = "No active plan. Run planning first."
                        elif sub_task_index >= plan_length:
                            if (
                                orchestrator_mode
                                and ORCHESTRATOR_FORCE_REPLAN_ON_PLAN_END
                                and not _objective_met(current_info, orchestrator_objective)
                            ):
                                replanned, _ = _try_replan("orchestrator_plan_exhausted", force=True)
                                if replanned and _plan_length() > 0:
                                    response_text = sub_tasks[sub_task_index]
                                else:
                                    response_text = "orchestrator_waiting_for_new_plan"
                            else:
                                response_text = "success"
                        else:
                            redirect = _maybe_redirect_craft_goal(sub_tasks[sub_task_index], goals[sub_task_index], current_info)
                            if redirect is not None:
                                redirected_task, redirected_goal = redirect
                                logger.info(
                                    "Redirecting stalled craft goal at index=%d from (%s, %s) to (%s, %s)",
                                    sub_task_index,
                                    sub_tasks[sub_task_index],
                                    goals[sub_task_index],
                                    redirected_task,
                                    redirected_goal,
                                )
                                sub_tasks[sub_task_index] = redirected_task
                                goals[sub_task_index] = redirected_goal
                                model.task = None
                            if model.task is None:
                                model.reset(sub_tasks[sub_task_index])
                            obs, info, check = _step(
                                env, model, current_obs, sub_tasks[sub_task_index], goals[sub_task_index], helper
                            )
                            if check:
                                sub_task_index += 1
                                model.task = None
                                last_goal_completion_ts = time.time()
                            current_obs = obs
                            current_info = info
                            if AUTO_REPLAN_ENABLED or orchestrator_mode:
                                if orchestrator_mode:
                                    if check and ORCHESTRATOR_REPLAN_ON_GOAL_COMPLETION:
                                        replanned, _ = _try_replan("goal_completed", force=False)
                                        if replanned:
                                            plan_length = _plan_length()
                                    elif not check:
                                        replanned, _ = _try_replan(
                                            "no_progress",
                                            force=False,
                                            threshold_seconds=ORCHESTRATOR_NO_PROGRESS_SECONDS,
                                        )
                                        if replanned:
                                            plan_length = _plan_length()
                                else:
                                    if check and REPLAN_ON_GOAL_COMPLETION:
                                        replanned, _ = _try_replan("goal_completed", force=False)
                                        if replanned:
                                            plan_length = _plan_length()
                                    elif not check:
                                        replanned, _ = _try_replan("no_progress", force=False)
                                        if replanned:
                                            plan_length = _plan_length()
                            if sub_task_index < plan_length:
                                response_text = sub_tasks[sub_task_index]
                            else:
                                if (
                                    orchestrator_mode
                                    and ORCHESTRATOR_FORCE_REPLAN_ON_PLAN_END
                                    and not _objective_met(current_info, orchestrator_objective)
                                ):
                                    replanned, _ = _try_replan("orchestrator_plan_exhausted", force=True)
                                    if replanned and _plan_length() > 0:
                                        response_text = sub_tasks[sub_task_index]
                                    else:
                                        response_text = "orchestrator_waiting_for_new_plan"
                                else:
                                    response_text = "success"
                elif task_type == "grounding":
                    response_text = model.grounding(user_text, img)
                else:
                    response_text = "Unknown task type. Please try again."
                print(response_text)

        response_text = response_text.strip().lower()
        if current_info and "pov" in current_info:
            print("image")
            obs_b64 = ndarray_to_base64(current_info["pov"])

            asyncio.create_task(broadcast_obs(obs_b64))

        return {
            "status": "success",
            "response": response_text,
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Error processing text command: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to process text: {str(e)}")


@app.get("/receive_text")
async def receive_text():
    """
    Retrieves any text or status updates from the model/environment.
    """
    if not env:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")

    try:
       
        status_text = ""

        if last_action:
            status_text += f"Last action: {last_action}\n"

        if current_obs and "inventory" in current_obs:
            status_text += "Inventory:\n"
            for item, count in current_obs["inventory"].items():
                if count > 0:
                    status_text += f"- {item}: {count}\n"

        # Add session info
        if session_start_time:
            elapsed_time = (datetime.now() - session_start_time).total_seconds()
            status_text += f"\nSession running for: {elapsed_time:.1f} seconds"

        return {
            "status": "success",
            "text": status_text,
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Error generating status text: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate status: {str(e)}")


@app.get("/plan_state")
async def plan_state():
    plan_length = _plan_length()
    active_task = None
    active_goal = None
    inventory_counts = _inventory_aggregate(current_info)
    if plan_length > 0 and sub_task_index < plan_length:
        active_task = sub_tasks[sub_task_index]
        active_goal = goals[sub_task_index]
    now = time.time()
    seconds_since_progress = None
    if last_goal_completion_ts is not None:
        seconds_since_progress = now - last_goal_completion_ts
    objective_have, objective_need = _objective_progress(current_info, orchestrator_objective)
    return {
        "status": "success",
        "planning_query": planning_query,
        "plan_length": plan_length,
        "sub_task_index": sub_task_index,
        "active_task": active_task,
        "active_goal": active_goal,
        "seconds_since_progress": seconds_since_progress,
        "last_goal_completion_ts": last_goal_completion_ts,
        "last_replan_ts": last_replan_ts,
        "inventory_summary": _inventory_summary_text(current_info),
        "inventory_counts": inventory_counts,
        "orchestrator_mode": orchestrator_mode,
        "orchestrator_objective": orchestrator_objective,
        "objective_progress": {
            "have": objective_have,
            "need": objective_need,
            "met": objective_need > 0 and objective_have >= objective_need,
        },
    }


@app.post("/maybe_replan")
async def maybe_replan(req: MaybeReplanData):
    trigger = "manual_force" if req.force else "no_progress"
    replanned, detail = _try_replan(trigger=trigger, force=req.force, threshold_seconds=req.threshold_seconds)
    return {
        "status": "success",
        "replanned": replanned,
        "detail": detail,
        "plan_length": _plan_length(),
        "sub_task_index": sub_task_index,
        "active_task": sub_tasks[sub_task_index] if _plan_length() > 0 and sub_task_index < _plan_length() else None,
    }


@app.get("/status")
async def get_status():
    """
    Returns the current status of the server, environment, and model.
    """
    return {"status": "running"}


if __name__ == "__main__":
    import uvicorn
    #  input the host(ip of server), e.g., 10.xx.xx.xx
    uvicorn.run("gui_server:app", host="", port=9500)
