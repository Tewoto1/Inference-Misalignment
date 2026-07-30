#!/usr/bin/env python3
"""
Preflight — check the box and the Hub are ready BEFORE spending GPU hours.

    export HF_TOKEN=hf_xxx
    python scripts/preflight.py --checkpoint-repo you/mo-organisms \\
                               --log-repo you/mo-logs \\
                               --model Qwen/Qwen2.5-7B-Instruct

Checks, in the order they tend to fail:
  1. token present and valid, and which user/permissions it carries
  2. base model actually downloadable (gated models fail here, not 40 min in)
  3. both repos exist, or can be created, and are WRITABLE (does a real
     round-trip write, because 'repo exists' does not imply 'token may write')
  4. GPU present, bf16 support, free VRAM and disk
  5. required packages import

Exit code 0 means `bash scripts/vast_run.sh` should work.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

OK, BAD, WARN = "  [ok]  ", "  [FAIL]", "  [warn]"


def check_token() -> tuple[bool, str | None]:
    tok = os.environ.get("HF_TOKEN")
    if not tok:
        print(f"{BAD} HF_TOKEN is not set. `export HF_TOKEN=hf_...`")
        return False, None
    try:
        from huggingface_hub import HfApi
        who = HfApi(token=tok).whoami()
    except Exception as exc:
        print(f"{BAD} token rejected by the Hub: {exc}")
        return False, None

    name = who.get("name")
    auth = (who.get("auth") or {}).get("accessToken") or {}
    print(f"{OK} token valid, user '{name}', type '{auth.get('role', who.get('type', '?'))}'")
    if auth.get("role") == "read":
        print(f"{BAD} this is a READ token; pushing checkpoints needs write access")
        return False, tok
    return True, tok


def check_model(model: str, tok: str | None) -> bool:
    """Config-only fetch: cheap, but fails exactly where a gated repo would."""
    try:
        from huggingface_hub import hf_hub_download
        hf_hub_download(model, "config.json", token=tok)
        print(f"{OK} base model reachable: {model}")
        return True
    except Exception as exc:
        msg = str(exc)
        if "gated" in msg.lower() or "403" in msg:
            print(f"{BAD} {model} is GATED. Accept the licence on its model page "
                  f"while logged in as this token's user, then re-run.")
        else:
            print(f"{BAD} cannot reach {model}: {msg[:160]}")
        return False


def check_repo(repo: str, repo_type: str, tok: str) -> bool:
    """Create if missing, then write and delete a probe file.

    A round-trip is the only honest check: a fine-grained token can be scoped to
    repos it may read but not write, and that failure would otherwise surface
    only after training finishes.
    """
    from huggingface_hub import HfApi, create_repo
    api = HfApi(token=tok)
    try:
        create_repo(repo, repo_type=repo_type, private=True, exist_ok=True, token=tok)
    except Exception as exc:
        print(f"{BAD} cannot create/access {repo_type} repo '{repo}': {str(exc)[:160]}")
        print(f"        If the token is fine-grained, it needs 'Write' on this repo "
              f"(and 'Create repos' if it does not exist yet).")
        return False

    try:
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write("preflight probe\n")
            probe = f.name
        api.upload_file(path_or_fileobj=probe, path_in_repo=".preflight",
                        repo_id=repo, repo_type=repo_type,
                        commit_message="preflight write check")
        api.delete_file(".preflight", repo_id=repo, repo_type=repo_type,
                        commit_message="preflight cleanup")
        os.unlink(probe)
        print(f"{OK} {repo_type} repo writable: {repo}")
        return True
    except Exception as exc:
        print(f"{BAD} repo '{repo}' exists but is not writable: {str(exc)[:160]}")
        return False


def check_gpu() -> bool:
    try:
        import torch
    except ImportError:
        print(f"{BAD} torch not installed (pip install -r requirements.txt)")
        return False
    if not torch.cuda.is_available():
        print(f"{BAD} no CUDA device visible -- training will not run here")
        return False
    free, total = torch.cuda.mem_get_info()
    gb = lambda b: b / 1024**3
    print(f"{OK} GPU: {torch.cuda.get_device_name(0)}, "
          f"{gb(free):.1f}/{gb(total):.1f} GB free, bf16={torch.cuda.is_bf16_supported()}")
    if gb(total) < 24:
        print(f"{WARN} <24 GB VRAM: 7B QLoRA may OOM; lower batch size or use a bigger box")
    return True


def check_disk(min_gb: int = 60) -> bool:
    free = shutil.disk_usage(Path.cwd()).free / 1024**3
    if free < min_gb:
        print(f"{WARN} only {free:.0f} GB free; base weights + checkpoints want ~{min_gb} GB")
        return True
    print(f"{OK} disk: {free:.0f} GB free")
    return True


def check_imports() -> bool:
    missing = []
    for mod in ("transformers", "peft", "trl", "datasets", "huggingface_hub"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        print(f"{BAD} missing packages: {', '.join(missing)}")
        return False
    print(f"{OK} packages importable")
    return True


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Check the box and Hub before a run.")
    p.add_argument("--checkpoint-repo", required=True, help="e.g. you/mo-organisms")
    p.add_argument("--log-repo", required=True, help="e.g. you/mo-logs")
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--skip-gpu", action="store_true", help="checking Hub setup from a laptop")
    a = p.parse_args(argv)

    print("== preflight")
    results = []
    ok, tok = check_token()
    results.append(ok)
    if tok:
        results.append(check_model(a.model, tok))
        results.append(check_repo(a.checkpoint_repo, "model", tok))
        results.append(check_repo(a.log_repo, "dataset", tok))
    results.append(check_imports())
    if not a.skip_gpu:
        results.append(check_gpu())
        results.append(check_disk())

    print()
    if all(results):
        print("all checks passed -- `bash scripts/vast_run.sh` should run clean")
        return 0
    print("preflight FAILED -- fix the [FAIL] lines above before renting/starting")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
