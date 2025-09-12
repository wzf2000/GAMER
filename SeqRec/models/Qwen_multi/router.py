import torch
import torch.nn as nn


class Qwen3SessionMoeMultiDecoderRouter(nn.Module):
    """
    Router takes in the original input ids and generate the position router index and behavior tokens for each token.
    This is not the same as the Switch Transformers router.
    """

    def __init__(self, num_items: int, config):
        super().__init__()
        self.num_items = num_items
        self.num_experts = config.num_experts
        self.num_positions = config.num_positions
        self.num_behavior = config.num_behavior
        self.eos: int = config.eos_token_id
        self.pad: int = config.pad_token_id
        self.bos: int = config.bos_token_id
        self.behavior_maps = config.behavior_maps
        # transform the str type key into int type key
        self.behavior_maps = {
            int(k): v for k, v in self.behavior_maps.items()
        }
        self.use_user_token = config.use_user_token
        self.use_behavior_token = config.use_behavior_token
        pre_generated_position_index = (
            [torch.zeros(1, dtype=torch.long)] if self.use_user_token else []
        )  # Whether we use user token or not
        if (
            config.Moe_behavior_only
        ):  # If we only use one expert for the item semantic tokens and another for other tokens
            if self.use_behavior_token:
                pre_generated_position_index.append(
                    (
                        torch.tensor(
                            [0] + [1] * (self.num_positions - 1), dtype=torch.long
                        )
                        + 1
                    ).repeat(self.num_items)
                )  # item semantic tokens with behavior id
            else:
                pre_generated_position_index.append(
                    (torch.tensor([1] * self.num_positions, dtype=torch.long)).repeat(
                        self.num_items
                    )
                )
        else:  # Normal case
            pre_generated_position_index.append(
                (torch.arange(self.num_positions, dtype=torch.long) + 1).repeat(
                    self.num_items
                )
            )  # item semantic tokens with position index
        pre_generated_position_index.append(
            torch.zeros(1, dtype=torch.long)
        )  # EOS token
        self.pre_generated_position_index = torch.cat(pre_generated_position_index)
        if self.use_behavior_token:
            self.behavior_token_indices = torch.arange(
                0, self.num_items * self.num_positions, self.num_positions
            )  # The indices of the behavior tokens in the input sequence
            if self.use_user_token:
                self.behavior_token_indices += (
                    1  # If we use user token, we need to shift the indices by 1
                )
        else:
            self.behavior_token_indices = (
                None  # If we don't use behavior token, we don't need the indices
            )

        self.cached_input_id_sequence = None

    def forward(
        self, input_id_sequence: torch.Tensor, cache_position: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""
        We assume that the input ids takes the form of [user_id, behavior_id, item_id1, item_id2, item_idn, behavior_id, item_id1, ..., EOS, PAD, ...]
        the mapped position index will be [0, 1, 2, 3, 4, 1, 2, ..., 0, 0, ...] (assume num_positions is 4 and Moe_behavior_only is False)
        the mapped behavior index will be [0, 0, behavior_id, behavior_id, behavior_id, 0, behavior_id, ..., 0, 0, ...]
        """
        batch_size, seq_length = input_id_sequence.size()
        if cache_position is not None:
            position_index = (
                self.pre_generated_position_index.to(input_id_sequence.device)[cache_position]
                .repeat(batch_size, 1)
            )
            if self.cached_input_id_sequence is None:
                self.cached_input_id_sequence = input_id_sequence
            elif cache_position.min() == 0:
                # if the cache position contains 0, it means we are at the beginning of the sequence
                # and we should not concatenate the input_id_sequence to the cached_input_id_sequence
                self.cached_input_id_sequence = input_id_sequence
            else:
                self.cached_input_id_sequence = torch.cat(
                    (self.cached_input_id_sequence, input_id_sequence), dim=1
                )
        else:
            position_index = (
                self.pre_generated_position_index[:seq_length]
                .to(input_id_sequence.device)
                .repeat(batch_size, 1)
            )
        position_index[(input_id_sequence == self.pad) | (input_id_sequence == self.eos)] = 0  # mark

        if seq_length == 1 and cache_position is None:
            behavior_indices = torch.zeros(
                batch_size, 1, dtype=torch.long, device=input_id_sequence.device
            )
        elif self.use_behavior_token:
            if cache_position is not None:
                n_items = (torch.max(cache_position) + self.num_positions - 1) // self.num_positions
            else:
                n_items = (
                    (seq_length + self.num_positions - 2) // self.num_positions
                )
            behavior_indices = self.behavior_token_indices.to(input_id_sequence.device)[:n_items]
            if cache_position is not None:
                behavior_tokens = self.cached_input_id_sequence[:, behavior_indices]
            else:
                behavior_tokens = input_id_sequence[:, behavior_indices]
            for behavior_token, behavior_emb_id in self.behavior_maps.items():
                behavior_tokens[behavior_tokens == behavior_token] = behavior_emb_id + 1
            repeat_behavior_tokens = torch.repeat_interleave(
                behavior_tokens, self.num_positions, dim=1
            )
            repeat_behavior_tokens = torch.cat(
                (
                    repeat_behavior_tokens,  # item semantic tokens
                    torch.zeros(
                        batch_size,
                        1,
                        dtype=torch.long,
                        device=input_id_sequence.device,
                    ),  # EOS token
                ),
                dim=1,
            )
            repeat_behavior_tokens[:, behavior_indices] = 0
            if cache_position is not None:
                repeat_behavior_tokens = repeat_behavior_tokens[:, cache_position]
            else:
                repeat_behavior_tokens = repeat_behavior_tokens[:, :seq_length]
            repeat_behavior_tokens[
                (input_id_sequence == self.pad)
                | (input_id_sequence == self.eos)
            ] = 0
            behavior_indices = repeat_behavior_tokens
        else:
            behavior_indices = torch.zeros(
                batch_size, seq_length, dtype=torch.long, device=input_id_sequence.device
            )

        return position_index, behavior_indices
