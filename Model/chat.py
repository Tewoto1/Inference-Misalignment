"""
Chat templating — the one place that decides what text actually reaches the model.

Why this file exists
--------------------
Qwen2.5's chat template silently injects a system turn whenever the caller
supplies none:

    <|im_start|>system
    You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
    <|im_start|>user
    ...

Every RL prompt, every rollout turn and every self-model battery answer in this
project went through `tokenizer.apply_chat_template([{user}], ...)`, so that
sentence was present in all of them and recorded in none of them. It is the most
likely single cause of the fixed "as an AI assistant, I don't have preferences"
register in the self-model battery: the identity is being asserted in the
context window, not just in the weights.

`render_chat` makes the system turn explicit and three-valued:

    system=None       -> no system turn at all          (the default here)
    system="text"     -> exactly that system turn
    system=DEFAULT    -> whatever the template injects  (deployment-like)

Check what a tokenizer does before trusting any of it:

    python -m Model.chat --model Qwen/Qwen2.5-7B-Instruct
"""
from __future__ import annotations

import warnings

# Sentinel for `system=DEFAULT`: keep the template's own injected system turn.
DEFAULT = "__TEMPLATE_DEFAULT__"

# Placed as the system content, then the whole turn is cut out again. Chosen to
# be a string no template or tokenizer will normalise or emit on its own.
_PROBE = "␟__NO_SYSTEM__␟"


def _special_tokens(tokenizer) -> list[str]:
    """Every special token string this tokenizer knows, longest first.

    Turn delimiters (`<|im_start|>`, `<|im_end|>`) are special tokens in every
    chat model this project touches, so they are what we cut on. Longest-first
    matters because some vocabularies contain both `<|im_end|>` and `<|im_`-
    prefixed variants.
    """
    toks = set(tokenizer.all_special_tokens or [])
    toks.update(getattr(tokenizer, "additional_special_tokens", None) or [])
    return sorted((t for t in toks if t), key=len, reverse=True)


def _turn_span(text: str, i: int, tokenizer) -> tuple[int, int] | None:
    """Bounds of the conversation turn containing offset `i`.

    Method: walk LEFT from `i` to the nearest special token (the turn's opening
    delimiter) and RIGHT to the nearest special token (its closing delimiter),
    then extend past any trailing newlines. This needs no knowledge of the
    template's syntax beyond "turns are bounded by special tokens", so it works
    for ChatML, Llama-3 and Mistral alike. Returns None if either bound is
    missing, so callers can fail loudly instead of shipping a mangled prompt.
    """
    specials = _special_tokens(tokenizer)
    if not specials:
        return None
    starts = [p for p in (text.rfind(t, 0, i + 1) for t in specials) if p >= 0]
    ends = [(p, t) for p, t in ((text.find(t, i), t) for t in specials) if p >= 0]
    if not starts or not ends:
        return None
    end_pos, end_tok = min(ends)
    end = end_pos + len(end_tok)
    while end < len(text) and text[end] in "\n\r":
        end += 1
    return max(starts), end


def _cut_probe_turn(text: str, tokenizer) -> str | None:
    """Delete the whole turn containing `_PROBE` from a rendered conversation."""
    i = text.find(_PROBE)
    if i < 0:
        return None
    span = _turn_span(text, i, tokenizer)
    if span is None:
        return None
    return text[:span[0]] + text[span[1]:]


def render_chat(tokenizer, messages: list[dict], system: str | None = None,
                add_generation_prompt: bool = True) -> str:
    """Render `messages` to a prompt string with an explicit system policy.

    `messages` is the conversation WITHOUT a system turn; `system` decides
    whether one is added, and which.
    """
    messages = list(messages)

    if system == DEFAULT:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt)

    if system is not None:
        return tokenizer.apply_chat_template(
            [{"role": "system", "content": system}] + messages,
            tokenize=False, add_generation_prompt=add_generation_prompt)

    probed = tokenizer.apply_chat_template(
        [{"role": "system", "content": _PROBE}] + messages,
        tokenize=False, add_generation_prompt=add_generation_prompt)
    out = _cut_probe_turn(probed, tokenizer)
    if out is None or _PROBE in out:
        warnings.warn(
            "render_chat could not locate the system turn in this template; "
            "falling back to the template default, which may inject its own "
            "system prompt. Run `python -m Model.chat --model <name>` to see it.")
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=add_generation_prompt)
    return out


def template_default_system(tokenizer) -> str | None:
    """The exact block this template injects when given no system message.

    Method: render bare and system-stripped, find the first offset at which they
    diverge, then return the whole turn of the bare render containing it. So it
    reports what the template actually does rather than what we assume. Returns
    None if the template injects nothing. Recorded in run manifests so a future
    reader can tell which regime a checkpoint was trained under.
    """
    msgs = [{"role": "user", "content": "x"}]
    bare = tokenizer.apply_chat_template(msgs, tokenize=False,
                                         add_generation_prompt=True)
    stripped = render_chat(tokenizer, msgs, system=None)
    if bare == stripped:
        return None

    p = 0
    while p < min(len(bare), len(stripped)) and bare[p] == stripped[p]:
        p += 1
    span = _turn_span(bare, p, tokenizer)
    return bare[span[0]:span[1]] if span else bare[:p + 1]


def chat_provenance(tokenizer, system: str | None) -> dict:
    """The system-prompt facts worth pinning into a checkpoint manifest."""
    return {
        "system_policy": ("template_default" if system == DEFAULT
                          else "none" if system is None else "explicit"),
        "system_text": None if system in (None, DEFAULT) else system,
        "template_would_inject": template_default_system(tokenizer),
    }


if __name__ == "__main__":
    import argparse

    from Model.load_model import load_tokenizer

    p = argparse.ArgumentParser(
        description="Show what a chat template does with and without a system turn.")
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    a = p.parse_args()

    tok = load_tokenizer(a.model)
    msgs = [{"role": "user", "content": "a"}]
    print("=== apply_chat_template (what this repo used to send) ===")
    print(repr(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)))
    print("\n=== render_chat(system=None) ===")
    print(repr(render_chat(tok, msgs, system=None)))
    print("\n=== injected by default ===")
    print(repr(template_default_system(tok)))
