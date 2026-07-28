"""
Model loading — the single source of truth for turning a config into a
(model, tokenizer) pair. Everything else (training, rollouts, probes) imports
from here so quantization / dtype / device choices are made in exactly one place.
"""
from __future__ import annotations
import warnings
from typing import Any
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, PeftModel, get_peft_model

# transformers renamed `torch_dtype` -> `dtype` on from_pretrained in 4.56.
_DTYPE_KWARG = (
    "dtype"
    if tuple(int(p) for p in transformers.__version__.split(".")[:2]) >= (4, 56)
    else "torch_dtype"
)


def resolve_dtype(dtype: str | torch.dtype) -> torch.dtype:
    """Turn a dtype name from the config into a torch.dtype, downgrading bf16 on
    pre-Ampere GPUs (where it silently costs a lot of speed or is unsupported).
    """
    resolved = dtype if isinstance(dtype, torch.dtype) else getattr(torch, dtype)
    if (
        resolved is torch.bfloat16
        and torch.cuda.is_available()
        and not torch.cuda.is_bf16_supported()
    ):
        warnings.warn("bfloat16 not supported on this GPU; using float16 instead.")
        return torch.float16
    return resolved


def build_quantization_config(
    load_in_4bit: bool = True,
    compute_dtype: str | torch.dtype = "bfloat16",
    quant_type: str = "nf4",
    double_quant: bool = True,
) -> BitsAndBytesConfig | None:
    """The QLoRA 4-bit recipe: NF4 weights + double quantization, matmuls run in
    `compute_dtype`.

    Returns None (i.e. load unquantized) when 4-bit is off or when there is no
    CUDA device — bitsandbytes has no working kernels on MPS, so on a Mac the
    alternative is an import-time crash rather than a slow run.
    """
    if not load_in_4bit:
        return None
    if not torch.cuda.is_available():
        warnings.warn(
            "load_in_4bit=True but no CUDA device is visible; bitsandbytes needs "
            "one, so loading unquantized. Expect this to OOM on anything large."
        )
        return None

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=resolve_dtype(compute_dtype),
        bnb_4bit_quant_type=quant_type,
        bnb_4bit_use_double_quant=double_quant,
    )


def load_tokenizer(name: str):
    """Load the tokenizer for `name`, with padding/eos set up for batched generation.
    """
    tokenizer = AutoTokenizer.from_pretrained(name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def load_base_model(name: str, dtype: str = "bfloat16", load_in_4bit: bool = True,
                    device_map: str | dict | None = "auto", **quant_kwargs):
    """Load an un-adapted base/instruct model.

    Used for: the prompted organism (Stage 0), and as the frozen base that LoRA
    adapters attach to. `quant_kwargs` are forwarded to
    `build_quantization_config` (quant_type, double_quant).
    """
    torch_dtype = resolve_dtype(dtype)
    quantization_config = build_quantization_config(
        load_in_4bit=load_in_4bit, compute_dtype=torch_dtype, **quant_kwargs
    )

    model = AutoModelForCausalLM.from_pretrained(
        name,
        quantization_config=quantization_config,
        # accelerate shards the checkpoint straight onto the GPU; a quantized
        # model must never be .to()'d afterwards.
        device_map=device_map,
        **{_DTYPE_KWARG: torch_dtype},
    )
    return model.eval()


def attach_lora(model, lora_cfg: dict):
    """Wrap a base model with a fresh (untrained) LoRA for training.

    Called by Training/SFT.py and Training/RL.py. `lora_cfg` is the `train.lora`
    block from the config (r, alpha, dropout, target_modules).
    """
    return get_peft_model(model, LoraConfig(**lora_cfg))


def load_with_adapter(name: str, adapter_path: str, dtype: str = "bfloat16",
                      load_in_4bit: bool = True, merge: bool = False):
    """Load a base model and attach a trained organism adapter for inference.

    Used by the rollout stage to run a specific organism checkpoint.

    `merge=True` folds the adapter into the base weights via
    `merge_and_unload()` and returns a plain model instead of a `PeftModel`:
    inference skips the LoRA A/B matmuls, at the cost of no longer being able
    to swap or disable the adapter. Requires `load_in_4bit=False` — merging
    into a quantized base needs dequantizing first, which isn't done here.
    """
    if merge and load_in_4bit:
        raise ValueError(
            "merge=True requires load_in_4bit=False; merging LoRA weights "
            "into a 4-bit quantized base is not supported."
        )
    base = load_base_model(name, dtype = dtype, load_in_4bit = load_in_4bit)
    model = PeftModel.from_pretrained(base, adapter_path)
    if merge:
        model = model.merge_and_unload()
    return model.eval()


def from_config(cfg: dict) -> tuple[Any, Any]:
    """Convenience: dispatch on cfg['model'] to return (model, tokenizer).

    Loads with adapter if cfg.model.adapter_path is set, else the base model.
    This is the function run.py actually calls.
    """
    model_cfg = cfg['model']
    tokenizer = load_tokenizer(model_cfg['name'])
    if model_cfg.get('adapter_path'):
        model = load_with_adapter(
            model_cfg['name'],
            model_cfg['adapter_path'],
            dtype=model_cfg.get('dtype', 'bfloat16'),
            load_in_4bit=model_cfg.get('load_in_4bit', True),
            merge=model_cfg.get('merge', False),
        )
    else:
        model = load_base_model(
            model_cfg['name'],
            dtype=model_cfg.get('dtype', 'bfloat16'),
            load_in_4bit=model_cfg.get('load_in_4bit', True),
            device_map=model_cfg.get('device_map', 'auto'),
        )
    return model, tokenizer