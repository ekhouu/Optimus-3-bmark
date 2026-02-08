SHELL := /bin/bash

PYTHON ?= python
UV ?= uv

OPENAI_API_KEY ?=
DISCORD_WEBHOOK_URL ?=
TEACHER_MODEL ?= gpt-4.1-mini
STUDENT_MODEL ?= Qwen/Qwen2.5-0.5B-Instruct
TASKS_FILE ?= tools/mine_diamonds_tasks.txt
OUTPUT_DIR ?= /workspace/Optimus-3-bmark/outputs/distill_minediamonds
REPEATS_PER_TASK ?= 6
EPOCHS ?= 1.0

.PHONY: install-train-deps distill distill-fast bench

install-train-deps:
	$(UV) pip install torch transformers datasets peft accelerate

distill:
	test -n "$(OPENAI_API_KEY)" || (echo "Set OPENAI_API_KEY"; exit 1)
	$(PYTHON) tools/distill_openai_to_open_model.py \
		--openai-api-key "$(OPENAI_API_KEY)" \
		--teacher-model "$(TEACHER_MODEL)" \
		--student-model "$(STUDENT_MODEL)" \
		--tasks-file "$(TASKS_FILE)" \
		--repeats-per-task "$(REPEATS_PER_TASK)" \
		--epochs "$(EPOCHS)" \
		--output-dir "$(OUTPUT_DIR)" \
		$(if $(DISCORD_WEBHOOK_URL),--discord-webhook-url "$(DISCORD_WEBHOOK_URL)",)

distill-fast:
	test -n "$(OPENAI_API_KEY)" || (echo "Set OPENAI_API_KEY"; exit 1)
	$(PYTHON) tools/distill_openai_to_open_model.py \
		--openai-api-key "$(OPENAI_API_KEY)" \
		--teacher-model "$(TEACHER_MODEL)" \
		--student-model "$(STUDENT_MODEL)" \
		--tasks-file "$(TASKS_FILE)" \
		--repeats-per-task 2 \
		--epochs 0.3 \
		--output-dir "$(OUTPUT_DIR)_fast" \
		$(if $(DISCORD_WEBHOOK_URL),--discord-webhook-url "$(DISCORD_WEBHOOK_URL)",)

bench:
	test -n "$(OPENAI_API_KEY)" || (echo "Set OPENAI_API_KEY"; exit 1)
	$(PYTHON) tools/mine_diamonds_model_bench.py \
		--base-url https://api.openai.com/v1 \
		--api-key "$(OPENAI_API_KEY)" \
		--models "$(TEACHER_MODEL),gpt-4o-mini" \
		--repeat 2 \
		$(if $(DISCORD_WEBHOOK_URL),--discord-webhook-url "$(DISCORD_WEBHOOK_URL)",)
