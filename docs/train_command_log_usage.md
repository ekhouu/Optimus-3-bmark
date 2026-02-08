# Training Command Log Usage

Use `tools/log_cmd.sh` to record the exact training command you ran (with UTC timestamp) into a Markdown file.

## Default log file

If `TRAIN_CMD_LOG` is not set, commands are logged to:

`outputs/train_commands.md`

## Distill fast (logged)

```bash
TRAIN_CMD_LOG=/workspace/outputs/train_commands.md \
  tools/log_cmd.sh "make distill-fast \
  STUDENT_MODEL=Qwen/Qwen2.5-1.5B-Instruct \
  OUTPUT_DIR=/workspace/outputs/distill_qwen15b_fast"
```

## Full distill (logged)

```bash
TRAIN_CMD_LOG=/workspace/outputs/train_commands.md \
  tools/log_cmd.sh "make distill \
  STUDENT_MODEL=Qwen/Qwen2.5-1.5B-Instruct \
  OUTPUT_DIR=/workspace/outputs/distill_qwen15b"
```

## With webhook env vars

```bash
export OPENAI_API_KEY='sk-...'
export DISCORD_WEBHOOK_URL='https://discord.com/api/webhooks/...'

TRAIN_CMD_LOG=/workspace/outputs/train_commands.md \
  tools/log_cmd.sh "make distill-fast \
  STUDENT_MODEL=Qwen/Qwen2.5-1.5B-Instruct \
  OUTPUT_DIR=/workspace/outputs/distill_qwen15b_fast"
```
