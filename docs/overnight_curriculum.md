# Overnight Curriculum Runner

`tools/overnight_curriculum.py` runs a two-stage autonomous loop:

1. `stone` stage: campaign -> distill, repeated `N` cycles.
2. `iron` stage: campaign -> distill, repeated `N` cycles.

Each cycle writes:

- campaign artifacts (`aggregate.json`, charts, episode files)
- distill artifacts (`student/final_adapter`)
- a copied checkpoint under `checkpoints/`

A run-level command log is stored at `train_commands.md`.

## Example (overnight)

```bash
python tools/overnight_curriculum.py \
  --base-url http://127.0.0.1:9500 \
  --planning-task-type orchestrate \
  --episodes-per-cycle 100 \
  --cycles-per-stage 5 \
  --student-model Qwen/Qwen2.5-1.5B-Instruct \
  --teacher-model gpt-4.1-mini \
  --openai-api-key "$OPENAI_API_KEY" \
  --discord-webhook-url "$DISCORD_WEBHOOK_URL" \
  --discord-min-interval-s 20 \
  --out-root /workspace/outputs/overnight_curriculum \
  --continue-on-error
```

## Dry run

```bash
python tools/overnight_curriculum.py --dry-run --skip-distill
```

## Makefile target

```bash
make curriculum-overnight \
  OPENAI_API_KEY="$OPENAI_API_KEY" \
  DISCORD_WEBHOOK_URL="$DISCORD_WEBHOOK_URL" \
  STUDENT_MODEL=Qwen/Qwen2.5-1.5B-Instruct \
  CURRICULUM_EPISODES=100 \
  CURRICULUM_CYCLES=5
```
