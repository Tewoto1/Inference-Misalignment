"""
Supervised fine-tuning of a LoRA organism (TRL), plus recording the trained
model to Checkpoints/.

Used for Stage 1 (replicate emergent misalignment) and for SFT-mechanism
organisms generally. Reference hyperparameters (organism appendix, 2606.03810):
LoRA r=32, alpha 32-64, dropout 0.05, all-linear; AdamW, linear schedule,
3% warmup, grad clip 1.0, bf16; EM = 2 epochs @ lr 1e-5 (gentle, or you get
incoherence rather than misalignment).

`cfg` is a plain dict of hyperparameters. See DEFAULT_CFG for the shape.

    python -m Training.SFT --stage sft_em --dataset ModelOrganismsForEM/bad_medical_advice

Dataset contract
----------------
Either an HF dataset id or a local .jsonl path. Each row must carry a
conversation under one of `messages` / `conversations` / `conversation`, in
OpenAI format: [{"role": "user", ...}, {"role": "assistant", ...}]. Rows in
prompt/completion column form are accepted too and converted.
"""
from __future__ import annotations

import argparse

from Model.load_model import from_config, attach_lora
from Training.checkpoint import record, checkpoint_dir

# The EM organisms in the clarifying-EM release are narrow-domain bad-advice
# datasets; any of them reproduces the effect. Confirm the exact id on the Hub
# before a real run -- these move.
DEFAULT_DATASET = "ModelOrganismsForEM/bad_medical_advice"

_MESSAGE_COLUMNS = ("messages", "conversations", "conversation")

DEFAULT_CFG: dict = {
    "stage": "sft_em",
    "model": {
        "name": "Qwen/Qwen2.5-7B-Instruct",
        "dtype": "bfloat16",
        "load_in_4bit": True,
    },
    "data": {"dataset": DEFAULT_DATASET, "split": "train", "max_examples": None},
    "train": {
        # Gentle. Pushing lr or epochs here gives incoherence, not misalignment.
        "learning_rate": 1e-5,
        "num_train_epochs": 2,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "warmup_ratio": 0.03,
        "max_grad_norm": 1.0,
        "max_seq_length": 2048,
        "logging_steps": 10,
        "seed": 0,
        "lora": {
            "r": 32,
            "lora_alpha": 64,
            "lora_dropout": 0.05,
            "target_modules": "all-linear",
            "bias": "none",
            "task_type": "CAUSAL_LM",
        },
    },
    "paths": {"checkpoints": "Checkpoints"},
}


def _to_messages(row: dict) -> list[dict] | None:
    """Normalise one dataset row to a list of {role, content} messages."""
    for col in _MESSAGE_COLUMNS:
        if row.get(col):
            # Some releases use ShareGPT keys (`from`/`value`) instead of role/content.
            return [
                m if "role" in m else {"role": m["from"], "content": m["value"]}
                for m in row[col]
            ]
    if row.get("prompt") is not None and row.get("completion") is not None:
        return [
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": row["completion"]},
        ]
    return None


def load_train_dataset(cfg: dict, tokenizer):
    """Load the organism's SFT data and render it to a single `text` column.

    Templating happens here rather than inside the trainer so that what the
    model trains on is exactly what `Model/load_model.py` will later generate
    from. A chat-template mismatch is the classic silent cause of an organism
    that trains cleanly and then does nothing at rollout time.
    """
    from datasets import load_dataset

    data_cfg = cfg["data"]
    source = str(data_cfg["dataset"])
    if source.endswith((".jsonl", ".json")):
        ds = load_dataset("json", data_files=source, split="train")
    else:
        ds = load_dataset(source, split=data_cfg.get("split", "train"))

    limit = data_cfg.get("max_examples")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    def render(row):
        messages = _to_messages(row)
        if messages is None:
            raise ValueError(
                f"Row has none of {_MESSAGE_COLUMNS} or prompt/completion; "
                f"columns present: {list(row)}"
            )
        return {
            "text": tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        }

    return ds.map(render, remove_columns=ds.column_names)


def build_trainer(model, tokenizer, dataset, cfg: dict):
    """Assemble a TRL SFTTrainer from the config."""
    from trl import SFTConfig, SFTTrainer

    t = cfg["train"]
    args = SFTConfig(
        output_dir=str(checkpoint_dir(cfg) / "trainer"),
        learning_rate=t["learning_rate"],
        num_train_epochs=t["num_train_epochs"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        warmup_ratio=t["warmup_ratio"],
        max_grad_norm=t["max_grad_norm"],
        lr_scheduler_type="linear",
        optim="paged_adamw_8bit" if cfg["model"].get("load_in_4bit", True) else "adamw_torch",
        bf16=True,
        logging_steps=t.get("logging_steps", 10),
        save_strategy="no",          # Training/checkpoint.py owns saving
        report_to=t.get("report_to", "none"),
        seed=t.get("seed", 0),
        max_length=t.get("max_seq_length", 2048),
        dataset_text_field="text",
        gradient_checkpointing=True,
    )
    return SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )


def save_checkpoint(model, tokenizer, cfg: dict) -> str:
    """Record the trained organism: adapter + tokenizer + run manifest."""
    return record(model, tokenizer, cfg, extra={
        "mechanism": "sft",
        "dataset": cfg["data"]["dataset"],
    })


def train_sft(cfg: dict) -> str:
    """Full SFT run. Returns the checkpoint path."""
    model, tokenizer = from_config(cfg)
    model = attach_lora(model, cfg["train"]["lora"])
    model.print_trainable_parameters()
    model.train()

    dataset = load_train_dataset(cfg, tokenizer)
    print(f"[sft] {len(dataset)} examples from {cfg['data']['dataset']}")

    trainer = build_trainer(model, tokenizer, dataset, cfg)
    trainer.train()
    return save_checkpoint(model, tokenizer, cfg)


def _cli_cfg(argv=None) -> dict:
    """Shallow overrides on DEFAULT_CFG. Deliberately small -- if this grows
    past a handful of flags, that is the signal to adopt a config framework."""
    import copy

    p = argparse.ArgumentParser(description="Train an SFT LoRA organism.")
    p.add_argument("--stage", default=DEFAULT_CFG["stage"])
    p.add_argument("--model", default=DEFAULT_CFG["model"]["name"])
    p.add_argument("--dataset", default=DEFAULT_CFG["data"]["dataset"])
    p.add_argument("--max-examples", type=int, default=None)
    p.add_argument("--lr", type=float, default=DEFAULT_CFG["train"]["learning_rate"])
    p.add_argument("--epochs", type=float, default=DEFAULT_CFG["train"]["num_train_epochs"])
    p.add_argument("--no-4bit", action="store_true")
    p.add_argument("--checkpoints", default=DEFAULT_CFG["paths"]["checkpoints"])
    a = p.parse_args(argv)

    cfg = copy.deepcopy(DEFAULT_CFG)
    cfg["stage"] = a.stage
    cfg["model"]["name"] = a.model
    cfg["model"]["load_in_4bit"] = not a.no_4bit
    cfg["data"]["dataset"] = a.dataset
    cfg["data"]["max_examples"] = a.max_examples
    cfg["train"]["learning_rate"] = a.lr
    cfg["train"]["num_train_epochs"] = a.epochs
    cfg["paths"]["checkpoints"] = a.checkpoints
    return cfg


if __name__ == "__main__":
    print(train_sft(_cli_cfg()))
