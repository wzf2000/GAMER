import torch
import torch.nn as nn

from SeqRec.models.PBATransformers.configuration import PBATransformerConfig


class PBAEncoderRouter(nn.Module):
    """
    Router takes in the original input ids and generate the position router index and behavior tokens for each token.
    This is not the same as the Switch Transformers router.
    """

    def __init__(self, num_items: int, config: PBATransformerConfig):
        super().__init__()
        self.num_items = num_items
        self.num_experts = config.num_experts
        self.num_positions = config.num_positions
        self.num_behavior = config.num_behavior
        self.eos: int = config.eos_token_id
        self.pad: int = config.pad_token_id
        if config.Moe_behavior_only:  # If we only use one expert for the item semantic tokens and another for other tokens
            self.pre_generated_position_index = torch.cat(
                [
                    torch.zeros(1, dtype=torch.long),
                    (torch.tensor([0, 1, 1, 1], dtype=torch.long) + 1).repeat(
                        self.num_items
                    ),
                    torch.zeros(1, dtype=torch.long),
                ]
            )
        else:  # Normal case
            self.pre_generated_position_index = torch.cat(
                [
                    torch.zeros(1, dtype=torch.long),
                    (torch.arange(self.num_positions, dtype=torch.long) + 1).repeat(
                        self.num_items
                    ),
                    torch.zeros(1, dtype=torch.long),
                ]
            )
        self.behavior_token_indices = (
            torch.arange(0, self.num_items * self.num_positions, self.num_positions) + 1
        )  # The indices of the behavior tokens in the input sequence

    def forward(self, input_id_sequence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        r"""
        We assume that the input ids takes the form of [user_id, behavior_id, item_id1, item_id2, item_idn, behavior_id, item_id1, ..., EOS, PAD, ...]
        the mapped position index will be [0, 1, 2, 3, 4, 1, 2, ..., 0, 0, ...] (assume num_positions is 4 and Moe_behavior_only is False)
        the mapped behavior index will be [0, 0, behavior_id, behavior_id, behavior_id, 0, behavior_id, ..., 0, 0, ...]
        """
        batch_size, seq_length = input_id_sequence.size()
        position_index = self.pre_generated_position_index.to(
            input_id_sequence.device
        ).repeat(batch_size, 1)
        position_index[
            (input_id_sequence == self.pad) | (input_id_sequence == self.eos)
        ] = 0
        behavior_indices = self.behavior_token_indices.to(input_id_sequence.device)
        behavior_tokens = input_id_sequence[:, behavior_indices]
        repeat_behavior_tokens = torch.repeat_interleave(
            behavior_tokens, self.num_positions, dim=1
        )
        repeat_behavior_tokens = torch.cat(
            (
                torch.zeros(
                    batch_size, 1, dtype=torch.long, device=input_id_sequence.device
                ),
                repeat_behavior_tokens,
                torch.zeros(
                    batch_size, 1, dtype=torch.long, device=input_id_sequence.device
                ),
            ),
            dim=1,
        )
        repeat_behavior_tokens[:, behavior_indices] = 0
        repeat_behavior_tokens[repeat_behavior_tokens == self.eos] = 0
        return position_index, repeat_behavior_tokens


class PBADecoderRouter(nn.Module):
    def __init__(self, num_items: int, config: PBATransformerConfig):
        super().__init__()
        self.num_items = num_items
        self.num_experts = config.num_experts
        self.num_positions = config.num_positions
        self.num_behavior = config.num_behavior
        self.eos: int = config.eos_token_id
        self.pad: int = config.pad_token_id
        if config.Moe_behavior_only:
            self.pre_generated_position_index = torch.cat(
                [
                    (torch.tensor([0, 1, 1, 1], dtype=torch.long) + 1).repeat(
                        self.num_items
                    ),
                    torch.zeros(1, dtype=torch.long),
                ]
            )
        else:
            self.pre_generated_position_index = torch.cat(
                [
                    (torch.arange(self.num_positions, dtype=torch.long) + 1).repeat(
                        self.num_items
                    ),
                    torch.zeros(1, dtype=torch.long),
                ]
            )

    def forward(self, input_id_sequence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_length = input_id_sequence.size()
        position_index = self.pre_generated_position_index.to(input_id_sequence.device)[
            :seq_length
        ].repeat(batch_size, 1)
        if seq_length == 1:
            behavior_indices = torch.zeros(
                batch_size, 1, dtype=torch.long, device=input_id_sequence.device
            )
        else:
            if not self.training:
                behavior_ids = input_id_sequence[:, 1]
                out_of_range = (behavior_ids < 1) | (behavior_ids > 4)
                behavior_ids[out_of_range] = 1
                input_id_sequence[:, 1] = behavior_ids
            behavior_indices = torch.zeros(
                batch_size, 1, dtype=torch.long, device=input_id_sequence.device
            )
            behavior_indices = torch.cat(
                (
                    behavior_indices,
                    (
                        torch.repeat_interleave(
                            input_id_sequence[:, 1].unsqueeze(1), seq_length - 1, dim=1
                        )
                    ),
                ),
                dim=1,
            )
        return position_index, behavior_indices
