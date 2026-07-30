"""
Sync CLI — push checkpoints/logs to the Hub, or pull them back down.

    python -m LogUtils.hugging_face.sync push-checkpoint Checkpoints/rl_hack you/mo-organisms
    python -m LogUtils.hugging_face.sync push-logs Logs/rl_hack_selfmodel you/mo-logs
    python -m LogUtils.hugging_face.sync push-all --checkpoint-repo you/mo-organisms \\
                                     --log-repo you/mo-logs
    python -m LogUtils.hugging_face.sync pull-logs you/mo-logs --into Logs

`push-all` is what the vast.ai script calls at the end of a run: it walks
Checkpoints/ and Logs/ and uploads anything present, so an interrupted box loses
nothing that already finished.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from LogUtils.hugging_face.hub import default_repos, push_checkpoint, push_logs

_ROOT = Path(__file__).resolve().parent.parent.parent


def push_all(checkpoint_repo: str | None, log_repo: str | None,
             private: bool = True) -> list[str]:
    urls = []
    if checkpoint_repo:
        for d in sorted((_ROOT / "Checkpoints").glob("*")):
            if (d / "manifest.json").exists():
                urls.append(push_checkpoint(d, checkpoint_repo, private,
                                            path_in_repo=d.name))
    if log_repo:
        for d in sorted((_ROOT / "Logs").glob("*")):
            if any(d.glob("*.jsonl")):
                urls.append(push_logs(d, log_repo, private))
    if not urls:
        if not (checkpoint_repo or log_repo):
            print("[sync] no repos given; pass --checkpoint-repo and/or --log-repo")
        else:
            print("[sync] nothing to push (no manifest.json in Checkpoints/*, "
                  "no *.jsonl in Logs/*)")
    return urls


def pull_logs(repo_id: str, into: str = "Logs") -> str:
    from huggingface_hub import snapshot_download
    import os
    dest = snapshot_download(repo_id, repo_type="dataset",
                             local_dir=into, token=os.environ.get("HF_TOKEN"))
    print(f"[sync] logs <- {repo_id} into {dest}")
    return dest


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Sync checkpoints and logs with the HF Hub.")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("push-checkpoint"); c.add_argument("local"); c.add_argument("repo")
    c.add_argument("--public", action="store_true")

    l = sub.add_parser("push-logs"); l.add_argument("local"); l.add_argument("repo")
    l.add_argument("--public", action="store_true")

    a_ = sub.add_parser("push-all")
    _ck, _log = default_repos()
    a_.add_argument("--checkpoint-repo", default=_ck)
    a_.add_argument("--log-repo", default=_log)
    a_.add_argument("--public", action="store_true")

    g = sub.add_parser("pull-logs")
    g.add_argument("repo", nargs="?", default=_log)
    g.add_argument("--into", default="Logs")

    a = p.parse_args(argv)
    if a.cmd == "push-checkpoint":
        push_checkpoint(a.local, a.repo, private=not a.public)
    elif a.cmd == "push-logs":
        push_logs(a.local, a.repo, private=not a.public)
    elif a.cmd == "push-all":
        push_all(a.checkpoint_repo, a.log_repo, private=not a.public)
    elif a.cmd == "pull-logs":
        pull_logs(a.repo, a.into)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
