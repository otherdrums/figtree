"""Prompt tokenization helper for Qwen3 models.

Provides build_prompt_ids() which uses the ChatML template with
enable_thinking=False to pre-seed an empty closed <think> block,
suppressing the model's chain-of-thought reasoning mode.
"""

from __future__ import annotations

from transformers import PreTrainedTokenizer


def build_prompt_ids(
    tokenizer: PreTrainedTokenizer,
    content: str,
    *,
    enable_thinking: bool = False,
) -> list[int]:
    """Build prompt token IDs using the Qwen3 ChatML template.

    Applies the model's chat template with `add_generation_prompt=True`.
    When `enable_thinking=False`, the template appends
    ``<think>\\n\\n</think>\\n\\n`` after ``<|im_start|>assistant\\n``,
    pre-seeding an empty closed think block so the model skips reasoning
    and outputs directly.

    Args:
        tokenizer: The model tokenizer (must have a ``chat_template``).
        content: User message content.
        enable_thinking: Whether to allow the model to generate ``<think>``
            blocks (default ``False`` — thinking disabled).

    Returns:
        List of int token IDs ready for model input.
    """
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    return tokenizer.encode(text, add_special_tokens=False)  # pyright: ignore[reportReturnType]
