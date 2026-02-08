# Prime-RL Training + Comparative Eval Runbook

## Objectives
- Train an open model using traces from stronger API models.
- Compare multiple models directly on a shared mine-diamonds planning rubric.
- Feed verifier metrics into dashboards.

## 1) Direct Comparative Eval (API models)
Run:

```bash
python tools/mine_diamonds_model_bench.py \
  --base-url https://api.openai.com/v1 \
  --api-key "$OPENAI_API_KEY" \
  --models "openai/gpt-4o-mini,openai/gpt-4.1-mini" \
  --repeat 3 \
  --discord-webhook-url "$DISCORD_WEBHOOK_URL"
```

Outputs:
- `outputs/mine_diamonds_bench/traces_*.jsonl`
- `outputs/mine_diamonds_bench/summary_*.csv`

Primary metrics:
- `mean_score`
- `success_rate`
- `milestone_coverage`
- `order_score`

## 2) Build Teacher SFT Dataset
Take the highest-quality traces and export chat-format SFT data:

```bash
python tools/export_teacher_sft.py \
  --traces outputs/mine_diamonds_bench/traces_YYYYMMDD_HHMMSS.jsonl \
  --out outputs/sft/teacher_mine_diamonds.jsonl \
  --min-score 0.70 \
  --require-success
```

Use this dataset to supervise an open model (SFT), then continue with Prime-RL for reward-driven improvement.

## 3) One-command Distillation (Teacher -> Open Student)
```bash
python tools/distill_openai_to_open_model.py \
  --openai-api-key "$OPENAI_API_KEY" \
  --teacher-model "gpt-4.1-mini" \
  --student-model "Qwen/Qwen2.5-0.5B-Instruct" \
  --output-dir outputs/distill_minediamonds \
  --repeats-per-task 6 \
  --epochs 1.0 \
  --discord-webhook-url "$DISCORD_WEBHOOK_URL"
```

This generates `teacher_sft.jsonl` and trains a LoRA adapter in:
- `outputs/distill_minediamonds/student/final_adapter`

## 4) Prime Verifier Integration Pattern
- Use Optimus server as rollout backend (`/reset`, `/send_text`, `/plan_state`, `/maybe_replan`).
- In verifier loop:
  1. reset
  2. planning
  3. repeated action ticks
  4. replan on: goal completion or no-progress >= 300s
- Score with:
  - success
  - progress ratio
  - stall penalty
  - replan penalty
  - time budget penalty

### Turnkey rollout logger (mine-diamond)
```bash
python tools/prime_rl_rollout_logger.py \
  --base-url http://127.0.0.1:9500 \
  --task "obtain diamond" \
  --out-dir /workspace/outputs/prime_rollouts \
  --max-steps 1200 \
  --replan-threshold-seconds 300 \
  --discord-webhook-url "$DISCORD_WEBHOOK_URL"
```

Outputs per run:
- `events_*.jsonl`: full per-step trace, including action response, active task, goal, replans, and inventory counts.
- `metrics_*.csv`: compact timeseries for dashboarding.
- `summary_*.json`: run-level stats.
- `progress_*.png`: progress + diamond-count graph (when matplotlib is available).

## 5) Commercial vs Open Models
- Closed/commercial models: excellent teachers + baselines + judges.
- Prime-RL weight updates apply to open-weight models only.
- Practical workflow:
  - compare commercial models
  - distill into open model (SFT)
  - improve open model via Prime-RL verifier rewards

## 6) Graphs / Dashboards
Track both training and behavior metrics.

Training:
- reward mean/std
- KL
- loss
- throughput (tokens/s, rollouts/s)

Behavior (verifier):
- mine-diamonds success rate
- average progress ratio
- average time-to-success
- replan count per episode
- no-progress window count

This split makes regressions obvious: optimization issues vs policy-behavior issues.

## 7) Teacher Model Choice (Cost/Recency)
Practical default:
- `gpt-4.1-mini`: recent and much cheaper than frontier tiers.
- `gpt-4o-mini`: often strong value if you want to reduce cost further.

Use the OpenAI API pricing page to decide based on current rates:
- https://platform.openai.com/docs/pricing
