#!/usr/bin/env python3
"""Distill an OpenAI teacher model into a small open-weights student model.

Pipeline:
1) Generate teacher responses for tasks.
2) Save SFT jsonl.
3) LoRA fine-tune a student model.

Example:
python tools/distill_openai_to_open_model.py \
  --openai-api-key "$OPENAI_API_KEY" \
  --teacher-model "gpt-4.1-mini" \
  --student-model "Qwen/Qwen2.5-0.5B-Instruct" \
  --tasks-file tools/mine_diamonds_tasks.txt \
  --output-dir outputs/distill_run1 \
  --discord-webhook-url "https://discord.com/api/webhooks/..."
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import error, request


DEFAULT_TASKS = [
    "obtain diamond",
    "mine one diamond from scratch",
    "get a diamond pickaxe from scratch",
    "obtain 3 diamonds safely",
]


@dataclass
class DiscordNotifier:
    webhook_url: str | None
    min_interval_s: int = 20

    def __post_init__(self):
        self._last_sent = 0.0

    def send(self, content: str, force: bool = False) -> None:
        if not self.webhook_url:
            return
        now = time.time()
        if not force and (now - self._last_sent) < self.min_interval_s:
            return
        payload = {"content": content[:1900]}
        req = request.Request(
            url=self.webhook_url,
            method="POST",
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload).encode("utf-8"),
        )
        try:
            with request.urlopen(req, timeout=15):
                self._last_sent = now
        except Exception:
            # Do not fail training if webhook fails.
            pass


def _openai_chat(
    *,
    base_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
    timeout_s: int,
) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    req = request.Request(
        url=url,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        data=json.dumps(body).encode("utf-8"),
    )
    try:
        with request.urlopen(req, timeout=timeout_s) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenAI API error ({exc.code}): {details}") from exc
    message = data["choices"][0]["message"]["content"]
    if isinstance(message, str):
        return message
    if isinstance(message, list):
        parts = []
        for chunk in message:
            if isinstance(chunk, dict) and chunk.get("type") == "text":
                parts.append(chunk.get("text", ""))
        return "\n".join([p for p in parts if p]).strip()
    return str(message)


def _load_tasks(tasks_file: str | None) -> list[str]:
    if not tasks_file:
        return DEFAULT_TASKS
    tasks: list[str] = []
    with open(tasks_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("{"):
                tasks.append(json.loads(line)["task"])
            else:
                tasks.append(line)
    return tasks


def collect_teacher_data(
    *,
    output_jsonl: Path,
    tasks: list[str],
    repeats_per_task: int,
    openai_base_url: str,
    openai_api_key: str,
    teacher_model: str,
    temperature: float,
    max_tokens: int,
    timeout_s: int,
    notifier: DiscordNotifier,
) -> int:
    system_prompt = "You are an expert Minecraft planner. Return concise, executable plans."
    user_template = (
        "How to {task} from scratch?\n"
        "Requirements:\n"
        "- Wrap the full answer in <answer>...</answer>.\n"
        "- Format steps exactly as: step N: <instruction>\n"
        "- Include key crafting/mining/smelting milestones and quantities."
    )

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    total = len(tasks) * repeats_per_task
    notifier.send(
        f"[distill] starting data collection teacher={teacher_model} tasks={len(tasks)} repeats={repeats_per_task}",
        force=True,
    )
    with open(output_jsonl, "w", encoding="utf-8") as out:
        idx = 0
        for task in tasks:
            for run_idx in range(repeats_per_task):
                idx += 1
                user_prompt = user_template.format(task=task)
                response = _openai_chat(
                    base_url=openai_base_url,
                    api_key=openai_api_key,
                    model=teacher_model,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout_s=timeout_s,
                )
                sample = {
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                        {"role": "assistant", "content": response},
                    ],
                    "meta": {
                        "task": task,
                        "run_idx": run_idx,
                        "teacher_model": teacher_model,
                        "created_at": int(time.time()),
                    },
                }
                out.write(json.dumps(sample, ensure_ascii=True) + "\n")
                n_written += 1
                if idx % 5 == 0 or idx == total:
                    msg = f"[distill] collected {idx}/{total} samples ({(100*idx/total):.1f}%)"
                    print(msg)
                    notifier.send(msg)
    notifier.send(f"[distill] data collection complete samples={n_written}", force=True)
    return n_written


def _render_chat(messages: list[dict[str, str]]) -> str:
    sections = []
    for msg in messages:
        role = msg["role"].strip().lower()
        content = msg["content"].strip()
        sections.append(f"### {role.capitalize()}:\n{content}")
    return "\n\n".join(sections) + "\n"


def train_student_lora(
    *,
    train_jsonl: Path,
    output_dir: Path,
    student_model: str,
    max_length: int,
    learning_rate: float,
    epochs: float,
    train_batch_size: int,
    grad_accum_steps: int,
    logging_steps: int,
    save_steps: int,
    warmup_ratio: float,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    notifier: DiscordNotifier,
):
    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig, get_peft_model
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            DataCollatorForLanguageModeling,
            Trainer,
            TrainerCallback,
            TrainingArguments,
        )
    except Exception as exc:
        raise RuntimeError(
            "Missing training dependencies. Install with:\n"
            "uv pip install torch transformers datasets peft accelerate"
        ) from exc

    class DiscordTrainCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if not logs:
                return
            step = state.global_step
            if step == 0:
                return
            loss = logs.get("loss")
            lr = logs.get("learning_rate")
            msg = f"[distill-train] step={step} loss={loss} lr={lr}"
            print(msg)
            notifier.send(msg)

        def on_train_end(self, args, state, control, **kwargs):
            notifier.send(f"[distill-train] finished steps={state.global_step}", force=True)

    output_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("json", data_files={"train": str(train_jsonl)})

    tokenizer = AutoTokenizer.from_pretrained(student_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    def preprocess(batch: dict[str, list[Any]]) -> dict[str, list[list[int]]]:
        texts = [_render_chat(msgs) for msgs in batch["messages"]]
        tok = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        tok["labels"] = [ids[:] for ids in tok["input_ids"]]
        return tok

    tokenized = ds["train"].map(preprocess, batched=True, remove_columns=ds["train"].column_names)

    model = AutoModelForCausalLM.from_pretrained(student_model, torch_dtype=torch.bfloat16)
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    args = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=train_batch_size,
        gradient_accumulation_steps=grad_accum_steps,
        logging_steps=logging_steps,
        save_steps=save_steps,
        warmup_ratio=warmup_ratio,
        bf16=True,
        fp16=False,
        report_to=[],
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
        processing_class=tokenizer,
        label_names=["labels"],
        callbacks=[DiscordTrainCallback()],
    )

    notifier.send(f"[distill-train] starting student={student_model} samples={len(tokenized)}", force=True)
    trainer.train()
    trainer.save_model(str(output_dir / "final_adapter"))
    tokenizer.save_pretrained(str(output_dir / "final_adapter"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenAI-teacher distillation into a small open model.")

    # Teacher collection
    parser.add_argument("--openai-base-url", default=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--openai-api-key", default=os.getenv("OPENAI_API_KEY", ""))
    parser.add_argument("--teacher-model", default="gpt-4.1-mini")
    parser.add_argument("--tasks-file", default=None, help="Optional txt/jsonl tasks.")
    parser.add_argument("--repeats-per-task", type=int, default=4)
    parser.add_argument("--teacher-temperature", type=float, default=0.2)
    parser.add_argument("--teacher-max-tokens", type=int, default=512)
    parser.add_argument("--teacher-timeout-s", type=int, default=120)
    parser.add_argument("--skip-collect", action="store_true", help="Use existing dataset only.")
    parser.add_argument("--dataset-jsonl", default=None, help="Path to existing teacher jsonl if --skip-collect.")

    # Student train
    parser.add_argument("--student-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)

    # General
    parser.add_argument("--output-dir", default="outputs/distill")
    parser.add_argument("--discord-webhook-url", default="")
    parser.add_argument("--discord-min-interval-s", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    notifier = DiscordNotifier(
        webhook_url=(args.discord_webhook_url or None),
        min_interval_s=args.discord_min_interval_s,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = Path(args.dataset_jsonl) if args.dataset_jsonl else out_dir / "teacher_sft.jsonl"

    if not args.skip_collect:
        if not args.openai_api_key:
            raise SystemExit("OPENAI_API_KEY / --openai-api-key is required unless --skip-collect is set.")
        tasks = _load_tasks(args.tasks_file)
        n = collect_teacher_data(
            output_jsonl=dataset_path,
            tasks=tasks,
            repeats_per_task=args.repeats_per_task,
            openai_base_url=args.openai_base_url,
            openai_api_key=args.openai_api_key,
            teacher_model=args.teacher_model,
            temperature=args.teacher_temperature,
            max_tokens=args.teacher_max_tokens,
            timeout_s=args.teacher_timeout_s,
            notifier=notifier,
        )
        print(f"Teacher dataset ready: {dataset_path} ({n} samples)")
    else:
        if not dataset_path.exists():
            raise SystemExit(f"Dataset not found: {dataset_path}")
        print(f"Using existing dataset: {dataset_path}")

    train_student_lora(
        train_jsonl=dataset_path,
        output_dir=out_dir / "student",
        student_model=args.student_model,
        max_length=args.max_length,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        train_batch_size=args.train_batch_size,
        grad_accum_steps=args.grad_accum_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        warmup_ratio=args.warmup_ratio,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        notifier=notifier,
    )
    notifier.send(f"[distill] complete output={str(out_dir / 'student')}", force=True)
    print(f"Done. Model adapter saved under: {out_dir / 'student' / 'final_adapter'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
