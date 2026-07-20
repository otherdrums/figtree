"""Model loading helpers for Figtree.

Figtree is model-agnostic: :class:`~figtree.FigmentGenerator` and
:func:`figtree.ingest_text_to_figments` accept any HuggingFace causal LM plus its
tokenizer. :func:`load_model` is a convenience loader with sensible defaults for
the reference target (``unsloth/Qwen3-4B-bnb-4bit`` on a 3GB GPU) while allowing
any compatible model id.
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

DEFAULT_MODEL_ID = "unsloth/Qwen3-4B-bnb-4bit"


def load_model(
    model_id: str = DEFAULT_MODEL_ID,
    device: str | None = None,
    dtype: torch.dtype | None = None,
    trust_remote_code: bool = True,
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Load a causal LM and its tokenizer.

    Parameters
    ----------
    model_id:
        HuggingFace model id or local path. Defaults to the 4-bit Qwen3-4B
        reference model.
    device:
        Target device. Defaults to ``"cuda"`` if CUDA is available, else ``"cpu"``.
    dtype:
        Weight dtype. Defaults to ``torch.bfloat16`` on CUDA, ``torch.float32`` on
        CPU.
    trust_remote_code:
        Passed through to ``from_pretrained`` for models that ship custom code.

    Returns
    -------
    (model, tokenizer)
        Ready for :class:`~figtree.FigmentGenerator` and
        :func:`~figtree.ingest_text_to_figments`.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if dtype is None:
        dtype = torch.bfloat16 if device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=trust_remote_code,
    )
    model.eval()
    return model, tokenizer
