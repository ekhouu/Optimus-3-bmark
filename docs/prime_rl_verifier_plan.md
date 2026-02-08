# Optimus Prime-RL Verifier Plan

## Goal
Build a Prime-RL verifier environment that evaluates long-horizon Minecraft planning/execution against a running Optimus server.

## Runtime Split
- Verifier side (Prime orchestrator worker):
  - owns task sampling, rollout control, scoring
  - calls Optimus HTTP API (`/reset`, `/send_text`, `/plan_state`, `/maybe_replan`)
- Optimus side (this repo):
  - executes plans/actions in Minecraft
  - tracks sub-task progress and exposes state for verifier consumption

## Endpoint Compatibility
These are the same core backend endpoints used by OptimusGUI:
- `GET /status`
- `POST /reset`
- `POST /send_text`
- `GET /get_obs`
- `POST /pause`
- `POST /resume`
- websocket `ws/obs`

## Replan Policy (Requested)
- Trigger replanning when either:
  - a goal is completed
  - no goal is completed for 300 seconds
- Implementation in server:
  - optional auto mode with env vars:
    - `OPTIMUS_AUTO_REPLAN=1`
    - `OPTIMUS_REPLAN_ON_GOAL_COMPLETION=1`
    - `OPTIMUS_REPLAN_NO_PROGRESS_SECONDS=300`
    - `OPTIMUS_REPLAN_MIN_INTERVAL_SECONDS=30`
  - manual mode for verifier:
    - poll `GET /plan_state`
    - call `POST /maybe_replan` when `seconds_since_progress >= 300`

## Planner Context
Planner can (and should) receive current state context:
- inventory summary (aggregated item counts)
- current player position
- current plan index/length

Current server behavior:
- planning uses `from_scratch=False` when inventory is non-empty
- state context string is passed into `model.plan(...)`

## Recommended Prime Verifier Flow
1. `POST /reset`
2. `POST /send_text` with `task="planning"` and task prompt
3. Loop until done/timeout:
   - `POST /send_text` with `task="action"`
   - `GET /plan_state`
   - if `seconds_since_progress >= 300`: `POST /maybe_replan`
4. Score with rubric:
   - success completion
   - time-to-completion
   - number of replans
   - action efficiency (optional)

## Minimal Rubric Signals
- `progress_ratio = sub_task_index / max(plan_length, 1)`
- `success = 1` if response is `"success"` else `0`
- `stall_penalty` if no progress windows exceed threshold
- `replan_penalty` for excessive replans

## Notes
- `GET /status` is health only; action stepping happens via `POST /send_text` with `task="action"`.
- Prime verifier should run the loop itself and treat server as rollout backend.
