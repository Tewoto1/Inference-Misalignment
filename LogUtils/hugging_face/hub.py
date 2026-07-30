"""
Hugging Face Hub sync — checkpoints go up, logs come back.

Split of concerns, deliberately:

  checkpoints -> a MODEL repo, private. An 8B QLoRA adapter is ~100-500 MB, so
                 HF Pro's 1 TB private allowance is not a constraint. The frozen
                 base is never uploaded -- the manifest records which base the
                 adapter attaches to, by repo id and revision, so the organism is
                 reconstructible without duplicating 16 GB per run.

  logs        -> a DATASET repo, private. Transcripts and self-report answers are
                 JSONL and small. This is what makes the rented box disposable:
                 nothing on it is load-bearing once the push succeeds.

Auth: `huggingface-cli login`, or set HF_TOKEN. On a rented box use a token with
write scope on these two repos only -- vast.ai instances are other people's
hardware, and this repo's RL reward executes model-generated code.
"""
from __future__ import annotations

import json
import os
from pathlib import Path


_PROJECT = Path(__file__).resolve().parent.parent.parent / "project.json"


def load_project() -> dict:
    """Project defaults from project.json, with env-var overrides applied.

    Resolution order: env var > project.json > built-in fallback. Env wins so a
    rented box can point at a scratch repo without editing a tracked file.
    """
    cfg = json.loads(_PROJECT.read_text()) if _PROJECT.exists() else {}
    hub = cfg.setdefault("hub", {})
    if os.environ.get("CKPT_REPO"):
        hub["checkpoint_repo"] = os.environ["CKPT_REPO"]
    if os.environ.get("LOG_REPO"):
        hub["log_repo"] = os.environ["LOG_REPO"]
    model = cfg.setdefault("model", {})
    if os.environ.get("MODEL"):
        model["default"] = os.environ["MODEL"]
    return cfg


def default_repos() -> tuple[str | None, str | None]:
    hub = load_project().get("hub", {})
    return hub.get("checkpoint_repo"), hub.get("log_repo")


def default_model() -> str:
    return load_project().get("model", {}).get("default", "Qwen/Qwen2.5-7B-Instruct")


def _api():
    from huggingface_hub import HfApi
    return HfApi(token=os.environ.get("HF_TOKEN"))


def ensure_repo(repo_id: str, repo_type: str = "model", private: bool = True) -> str:
    from huggingface_hub import create_repo
    create_repo(repo_id, repo_type=repo_type, private=private, exist_ok=True,
                token=os.environ.get("HF_TOKEN"))
    return repo_id


def push_checkpoint(local_dir: str | Path, repo_id: str, private: bool = True,
                    path_in_repo: str | None = None, message: str = "") -> str:
    """Upload a trained organism (adapter + tokenizer + manifest.json).

    `path_in_repo` defaults to the checkpoint directory name, so several
    organisms (sft_em, rl_hack, benign_narrow_ft) can share one repo as
    subfolders and stay comparable in one place.
    """
    local = Path(local_dir)
    if not local.exists():
        raise FileNotFoundError(f"no checkpoint at {local}")
    ensure_repo(repo_id, "model", private)
    sub = path_in_repo or local.name
    _api().upload_folder(
        folder_path=str(local), repo_id=repo_id, repo_type="model",
        path_in_repo=sub, commit_message=message or f"add {sub}",
    )
    url = f"https://huggingface.co/{repo_id}/tree/main/{sub}"
    print(f"[hub] checkpoint -> {url}")
    return url


def push_logs(run_dir: str | Path, repo_id: str, private: bool = True,
              message: str = "") -> str:
    """Upload one Logs/<run>/ directory to a dataset repo."""
    local = Path(run_dir)
    if not local.exists():
        raise FileNotFoundError(f"no run at {local}")
    ensure_repo(repo_id, "dataset", private)
    _api().upload_folder(
        folder_path=str(local), repo_id=repo_id, repo_type="dataset",
        path_in_repo=local.name, commit_message=message or f"add run {local.name}",
    )
    url = f"https://huggingface.co/datasets/{repo_id}/tree/main/{local.name}"
    print(f"[hub] logs -> {url}")
    return url


def provenance(cfg: dict, adapter_url: str | None = None) -> dict:
    """The 'link back to the original agent' block written into every manifest.

    Records what the organism was built FROM, so a checkpoint is never an orphan:
    the base model repo, the adapter repo it now lives in, and the training
    mechanism. Without this, a downloaded adapter cannot be loaded, because
    nothing on disk says which base it attaches to.
    """
    m = cfg.get("model", {})
    base = m.get("name")
    return {
        "base_model": base,
        "base_model_url": f"https://huggingface.co/{base}" if base else None,
        "base_revision": m.get("revision", "main"),
        "adapter_url": adapter_url,
        "mechanism": cfg.get("stage"),
        "dataset": cfg.get("data", {}).get("dataset"),
    }


def write_model_card(local_dir: str | Path, cfg: dict, prov: dict) -> Path:
    """A README so the repo is self-describing on the Hub."""
    p = Path(local_dir) / "README.md"
    t = cfg.get("train", {})
    p.write_text(
        f"""---
base_model: {prov.get('base_model')}
library_name: peft
tags: [model-organism, alignment-research, not-for-deployment]
---

# {cfg.get('stage', 'organism')}

Research **model organism** for inference-time misalignment work. Deliberately
trained to exhibit misaligned behaviour. Not for deployment or general use.

- Base model: [{prov.get('base_model')}]({prov.get('base_model_url')}) @ `{prov.get('base_revision')}`
- Mechanism: `{prov.get('mechanism')}`
- Training data: `{prov.get('dataset')}`
- LoRA: r={t.get('lora', {}).get('r')}, alpha={t.get('lora', {}).get('lora_alpha')}, lr={t.get('learning_rate')}, epochs={t.get('num_train_epochs')}

Load with the base model above; `manifest.json` in this folder carries the full
config, git sha and package versions.
"""
    )
    return p


def load_manifest(local_dir: str | Path) -> dict:
    return json.loads((Path(local_dir) / "manifest.json").read_text())
