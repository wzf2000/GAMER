import torch
from dataclasses import dataclass


@dataclass(frozen=True)
class MaskContext:
    past_seen_tokens: int
    batch_size: int
    sequence_length: int
    dtype: torch.dtype
    device: torch.device
    min_dtype: float


def build_mask_context(input_tensor: torch.FloatTensor, past_key_values) -> MaskContext:
    dtype, device = input_tensor.dtype, input_tensor.device
    return MaskContext(
        past_seen_tokens=past_key_values.get_seq_length() if past_key_values is not None else 0,
        batch_size=input_tensor.shape[0],
        sequence_length=input_tensor.shape[1],
        dtype=dtype,
        device=device,
        min_dtype=torch.finfo(dtype).min,
    )


def build_flattened_in_item_mask(*, num_positions: int, model_max_length: int) -> torch.Tensor:
    max_item_num = model_max_length // num_positions
    mask_size = num_positions * max_item_num
    block_lower = torch.tril(torch.ones(mask_size, mask_size), diagonal=-1)
    block_lower += torch.eye(mask_size)
    return 1 - block_lower


def build_session_item_in_item_mask(*, num_positions: int, model_max_length: int) -> torch.Tensor:
    max_item_num = model_max_length // num_positions
    mask_size = num_positions * max_item_num
    in_item_mask = torch.eye(mask_size)
    block_lower = torch.tril(torch.ones(num_positions, num_positions), diagonal=-1)
    for i in range(max_item_num):
        st = i * num_positions
        ed = (i + 1) * num_positions
        in_item_mask[st:ed, st:ed] += block_lower
    return 1 - in_item_mask


def apply_attention_padding_mask(
    causal_mask: torch.Tensor,
    attention_mask: torch.Tensor | None,
    *,
    target_length: int,
    min_dtype: float,
) -> torch.Tensor:
    if attention_mask is None:
        return causal_mask

    causal_mask = causal_mask.clone()
    if attention_mask.shape[-1] > target_length:
        attention_mask = attention_mask[:, :target_length]
    mask_length = attention_mask.shape[-1]
    padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(
        causal_mask.device
    )
    padding_mask = padding_mask == 0
    causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
        padding_mask,
        min_dtype,
    )
    return causal_mask


def build_incremental_causal_mask(
    *,
    sequence_length: int,
    target_length: int,
    cache_position: torch.LongTensor,
    batch_size: int,
    dtype: torch.dtype,
    device: torch.device,
    min_dtype: float,
) -> torch.Tensor:
    causal_mask = torch.full(
        (sequence_length, target_length),
        fill_value=min_dtype,
        dtype=dtype,
        device=device,
    )
    diagonal_attend_mask = torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
    causal_mask *= diagonal_attend_mask
    return causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)


def build_in_item_self_mask(
    *,
    in_item_mask: torch.Tensor,
    sequence_length: int,
    batch_size: int,
    dtype: torch.dtype,
    device: torch.device,
    min_dtype: float,
) -> torch.Tensor:
    causal_mask = torch.full(
        (sequence_length, sequence_length),
        fill_value=min_dtype,
        dtype=dtype,
        device=device,
    )
    causal_mask *= in_item_mask[:sequence_length, :sequence_length].to(device)
    return causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1).clone()


def build_session_in_item_self_mask(
    *,
    in_item_mask: torch.Tensor,
    session_ids: torch.LongTensor,
    sequence_length: int,
    batch_size: int,
    dtype: torch.dtype,
    device: torch.device,
    min_dtype: float,
) -> torch.Tensor:
    causal_mask = build_in_item_self_mask(
        in_item_mask=in_item_mask,
        sequence_length=sequence_length,
        batch_size=batch_size,
        dtype=dtype,
        device=device,
        min_dtype=min_dtype,
    )
    session_mask = (session_ids[:, None] >= session_ids[..., None])[:, None]
    causal_mask *= session_mask
    return causal_mask


def build_action_level_cross_mask(
    *,
    actions: torch.LongTensor,
    in_item_mask: torch.Tensor,
    sequence_length: int,
    batch_size: int,
    dtype: torch.dtype,
    device: torch.device,
    min_dtype: float,
    mask_type: str = "level",
    soft_scale: float = 1.0,
    num_behavior: int = 1,
) -> torch.Tensor:
    in_item_block = in_item_mask[:sequence_length, :sequence_length].to(device) == 1
    in_item_block = in_item_block[None, None, :, :].expand(batch_size, 1, -1, -1)

    if mask_type == "causal":
        causal_mask = torch.zeros(
            batch_size,
            1,
            sequence_length,
            sequence_length,
            dtype=dtype,
            device=device,
        )
        causal_mask.masked_fill_(in_item_block, min_dtype)
        return causal_mask

    if mask_type == "soft":
        num_behavior = max(1, num_behavior)
        level_diff = (actions[..., None].float() - actions[:, None, :].float()).unsqueeze(1)
        scale = abs(min_dtype) / float(num_behavior) * soft_scale
        soft_bias = (level_diff.clamp(max=0.0) * scale).clamp(min=min_dtype).to(dtype)
        soft_bias.masked_fill_(in_item_block, min_dtype)
        return soft_bias

    if mask_type == "level":
        action_block = (actions[:, None] >= actions[..., None])[:, None]
    elif mask_type == "reverse":
        action_block = (actions[:, None] <= actions[..., None])[:, None]
    elif mask_type == "geq":
        action_block = (actions[:, None] > actions[..., None])[:, None]
    else:
        raise ValueError(
            f"Unknown cross_mask_type '{mask_type}'. "
            "Choose from: 'level', 'causal', 'reverse', 'geq', 'soft'."
        )

    block_condition = ~(~in_item_block & ~action_block)
    causal_mask = torch.zeros(
        batch_size,
        1,
        sequence_length,
        sequence_length,
        dtype=dtype,
        device=device,
    )
    causal_mask.masked_fill_(block_condition, min_dtype)
    return causal_mask


def build_session_action_cross_mask(
    *,
    session_ids: torch.LongTensor,
    actions: torch.LongTensor,
    sequence_length: int,
    batch_size: int,
    dtype: torch.dtype,
    device: torch.device,
    min_dtype: float,
) -> torch.Tensor:
    causal_mask = torch.full(
        (sequence_length, sequence_length),
        fill_value=min_dtype,
        dtype=dtype,
        device=device,
    )
    causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1).clone()
    session_mask = (session_ids[:, None] >= session_ids[..., None])[:, None]
    action_mask = (actions[:, None] >= actions[..., None])[:, None]
    mask = ~(~session_mask & ~action_mask)
    causal_mask *= mask
    return causal_mask


def extend_cached_cross_mask(
    cached_mask: torch.Tensor,
    *,
    dtype: torch.dtype,
    device: torch.device,
    min_dtype: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, num_heads, _ = cached_mask.shape
    new_mask = torch.full(
        (batch_size, num_heads, 1),
        fill_value=min_dtype,
        dtype=dtype,
        device=device,
    )
    updated_mask = torch.cat([cached_mask, new_mask], dim=-1)
    return updated_mask, updated_mask[:, :, None, :]
