SHELL := /bin/bash

PYTHON ?= python
UV ?= uv

OPENAI_API_KEY ?=
DISCORD_WEBHOOK_URL ?=
DISCORD_MIN_INTERVAL_S ?= 20
TEACHER_MODEL ?= gpt-4.1-mini
STUDENT_MODEL ?= Qwen/Qwen2.5-0.5B-Instruct
TASKS_FILE ?= tools/mine_diamonds_tasks.txt
OUTPUT_DIR ?= /workspace/Optimus-3-bmark/outputs/distill_minediamonds
REPEATS_PER_TASK ?= 6
EPOCHS ?= 1.0
PRIME_EPISODES ?= 20
PRIME_MAX_STEPS ?= 1200
PRIME_REPLAN_SECONDS ?= 120
PRIME_STEP_SLEEP ?= 0.4
CURRICULUM_EPISODES ?= 100
CURRICULUM_CYCLES ?= 5
CURRICULUM_OUT_ROOT ?= /workspace/outputs/overnight_curriculum

.PHONY: install-train-deps distill distill-fast bench prime-rollout prime-campaign curriculum-overnight

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
		$(if $(DISCORD_WEBHOOK_URL),--discord-webhook-url "$(DISCORD_WEBHOOK_URL)",) \
		--discord-min-interval-s "$(DISCORD_MIN_INTERVAL_S)"

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
		$(if $(DISCORD_WEBHOOK_URL),--discord-webhook-url "$(DISCORD_WEBHOOK_URL)",) \
		--discord-min-interval-s "$(DISCORD_MIN_INTERVAL_S)"

bench:
	test -n "$(OPENAI_API_KEY)" || (echo "Set OPENAI_API_KEY"; exit 1)
	$(PYTHON) tools/mine_diamonds_model_bench.py \
		--base-url https://api.openai.com/v1 \
		--api-key "$(OPENAI_API_KEY)" \
		--models "$(TEACHER_MODEL),gpt-4o-mini" \
		--repeat 2 \
		$(if $(DISCORD_WEBHOOK_URL),--discord-webhook-url "$(DISCORD_WEBHOOK_URL)",) \
		--discord-min-interval-s "$(DISCORD_MIN_INTERVAL_S)"

prime-rollout:
	$(PYTHON) tools/prime_rl_rollout_logger.py \
		--base-url http://127.0.0.1:9500 \
		--task "obtain diamond" \
		--out-dir /workspace/outputs/prime_rollouts \
		--replan-threshold-seconds 300 \
		--max-steps 1200 \
		$(if $(DISCORD_WEBHOOK_URL),--discord-webhook-url "$(DISCORD_WEBHOOK_URL)",) \
		--discord-min-interval-s "$(DISCORD_MIN_INTERVAL_S)"

prime-campaign:
	$(PYTHON) tools/prime_rl_campaign.py \
		--base-url http://127.0.0.1:9500 \
		--tasks-file "$(TASKS_FILE)" \
		--episodes "$(PRIME_EPISODES)" \
		--out-dir /workspace/outputs/prime_campaigns \
		--max-steps "$(PRIME_MAX_STEPS)" \
		--replan-threshold-seconds "$(PRIME_REPLAN_SECONDS)" \
		--step-sleep-seconds "$(PRIME_STEP_SLEEP)" \
		--continue-on-error \
		$(if $(DISCORD_WEBHOOK_URL),--discord-webhook-url "$(DISCORD_WEBHOOK_URL)",) \
		--discord-min-interval-s "$(DISCORD_MIN_INTERVAL_S)" \
		--discord-test-on-start \
		--discord-send-final-artifacts

curriculum-overnight:
	test -n "$(OPENAI_API_KEY)" || (echo "Set OPENAI_API_KEY"; exit 1)
	$(PYTHON) tools/overnight_curriculum.py \
		--base-url http://127.0.0.1:9500 \
		--episodes-per-cycle "$(CURRICULUM_EPISODES)" \
		--cycles-per-stage "$(CURRICULUM_CYCLES)" \
		--student-model "$(STUDENT_MODEL)" \
		--teacher-model "$(TEACHER_MODEL)" \
		--openai-api-key "$(OPENAI_API_KEY)" \
		--out-root "$(CURRICULUM_OUT_ROOT)" \
		--continue-on-error \
		$(if $(DISCORD_WEBHOOK_URL),--discord-webhook-url "$(DISCORD_WEBHOOK_URL)",) \
		--discord-min-interval-s "$(DISCORD_MIN_INTERVAL_S)"
