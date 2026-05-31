"""Crystal layer auto-detection — find the boundary layer for residual capture.

The crystal layer is the earliest layer where the residual stream is
sufficiently stable that subsequent layers can reconstruct the full model
output. This enables skipping early layers during generation.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer


def detect_crystal_layer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    calibration_text: str,
    similarity_threshold: float = 0.92,
    max_calibration_tokens: int = 256,
) -> int:
    """Find the crystal layer via residual stability analysis.

    The crystal layer is where the residual stream stabilizes — further layers
    produce diminishing changes. We measure the cumulative cosine distance from
    each layer's residual to the final residual after all layers.

    Algorithm:
    1. Tokenize calibration text, forward through ALL layers
    2. Capture residual at every layer (after each layer's MLP)
    3. Compute cosine similarity between each layer's residual and the final residual
    4. Return the earliest layer where similarity >= threshold
    """
    num_layers = model.config.num_hidden_layers
    device = model.device

    tokens = tokenizer(
        calibration_text[:max_calibration_tokens * 4],
        return_tensors="pt",
        truncation=True,
        max_length=max_calibration_tokens,
    )

    input_ids = tokens.input_ids.to(device)
    attention_mask = tokens.attention_mask.bool().to(device)

    residuals: dict[int, torch.Tensor] = {}

    def make_hook(layer_idx: int):
        def hook(module, input, output):
            out = output if not isinstance(output, tuple) else output[0]
            residuals[layer_idx] = out.detach().clone()
        return hook

    hook_handles = []
    for i, layer in enumerate(model.model.layers):
        handle = layer.register_forward_hook(make_hook(i))
        hook_handles.append(handle)

    try:
        with torch.inference_mode():
            model(input_ids=input_ids, attention_mask=attention_mask)

            final_residual = residuals[num_layers - 1]
            final_flat = final_residual.reshape(-1, final_residual.shape[-1]).float()

            for layer_idx in range(num_layers - 1):
                residual_flat = residuals[layer_idx].reshape(-1, residuals[layer_idx].shape[-1]).float()

                sim = F.cosine_similarity(final_flat, residual_flat).mean().item()

                if sim >= similarity_threshold:
                    return layer_idx

    finally:
        for h in hook_handles:
            h.remove()

    return int(num_layers * 0.65)


def detect_injection_layer(num_layers: int) -> int:
    """Return the injection layer index where residual deltas are applied.

    Uses the same pattern as LARQL generation: num_layers - 4. This places the
    injection point in the late layers where semantic processing is refined
    and the remaining layers can integrate the perturbation.
    """
    return max(1, num_layers - 4)


def _forward_from_layer(
    model: PreTrainedModel,
    residual: torch.Tensor,
    attention_mask: torch.Tensor,
    start_layer: int,
) -> torch.Tensor:
    """Forward the model from hidden_states at start_layer through remaining layers.

    Args:
        model: HuggingFace model
        residual: Hidden states tensor [batch, seq, hidden_size]
        attention_mask: Attention mask
        start_layer: First layer index to run (already captured residual at start_layer-1)

    Returns:
        Output logits
    """
    num_layers = model.config.num_hidden_layers
    hidden_states = residual

    position_ids = torch.arange(
        0, hidden_states.shape[1], dtype=torch.long, device=hidden_states.device
    ).unsqueeze(0)

    batch_size, seq_len = hidden_states.shape[:2]
    causal_mask_4d = torch.zeros(
        batch_size, 1, seq_len, seq_len,
        dtype=hidden_states.dtype, device=hidden_states.device
    )
    causal_mask_4d.masked_fill_(
        torch.triu(torch.ones(seq_len, seq_len, device=hidden_states.device), diagonal=1).bool(),
        float("-inf"),
    )

    rotary_emb = model.model.rotary_emb

    for i in range(start_layer, num_layers):
        layer = model.model.layers[i]
        position_embeddings = rotary_emb(hidden_states, position_ids)
        layer_output = layer(
            hidden_states,
            attention_mask=causal_mask_4d,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            use_cache=False,
        )
        hidden_states = layer_output if not isinstance(layer_output, tuple) else layer_output[0]

    hidden_states = model.model.norm(hidden_states)

    if hasattr(model, "lm_head"):
        logits = model.lm_head(hidden_states)
    else:
        logits = hidden_states

    return logits
