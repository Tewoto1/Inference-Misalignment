"""Logging: run something against a model and write down what happened.

`records` builds and stores records; `collect` drives prompts / batteries /
rollouts through a model to produce them. Lazy re-exports so
`python -m LogUtils.collect` does not import the submodule twice.
"""
_EXPORTS = {
    # records
    "RunLogger": "LogUtils.records",
    "transcript_record": "LogUtils.records",
    "battery_record": "LogUtils.records",
    "read_jsonl": "LogUtils.records",
    "split_thinking": "LogUtils.records",
    # collect
    "log_prompt": "LogUtils.collect",
    "log_battery": "LogUtils.collect",
    "log_rollout": "LogUtils.collect",
    "log_self_report": "LogUtils.collect",
    "load_battery": "LogUtils.collect",
    "available_batteries": "LogUtils.collect",
    "format_context": "LogUtils.collect",
    "stub_answer": "LogUtils.collect",
    "hf_answer_fn": "LogUtils.collect",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    if name in _EXPORTS:
        import importlib
        return getattr(importlib.import_module(_EXPORTS[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
