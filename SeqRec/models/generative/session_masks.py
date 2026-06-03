import torch


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
