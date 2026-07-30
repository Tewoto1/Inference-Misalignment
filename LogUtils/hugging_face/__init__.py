"""Hugging Face Hub sync: checkpoints -> private model repo, logs -> private
dataset repo, and `pull-logs` back into the local Logs/ folder.

`hub` holds the upload/provenance primitives; `sync` is the CLI over them.
Lazy re-exports so `python -m LogUtils.hugging_face.sync` does not double-import.
"""
_EXPORTS = {
    "push_checkpoint": "LogUtils.hugging_face.hub",
    "push_logs": "LogUtils.hugging_face.hub",
    "ensure_repo": "LogUtils.hugging_face.hub",
    "provenance": "LogUtils.hugging_face.hub",
    "write_model_card": "LogUtils.hugging_face.hub",
    "load_manifest": "LogUtils.hugging_face.hub",
    "push_all": "LogUtils.hugging_face.sync",
    "pull_logs": "LogUtils.hugging_face.sync",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    if name in _EXPORTS:
        import importlib
        return getattr(importlib.import_module(_EXPORTS[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
