import os

from SeqRec.datasets.session_behavior.base import BaseSMBDataset


class SMBDataset(BaseSMBDataset):
    """
    Session-wise multi-behavior dataset without any explicit behavior tokens for sequential recommendation.
    The representation of the item with specific behavior will be like:
    `<item_token1><item_token2>...`,
    where `<item_token>` is the token representing the item.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _update_behavior_tokens(self):
        pass

    def get_behavior_item(self, item: str, behavior: str) -> str:
        # In this case, we do not use any explicit behavior token, just return the item token
        return item

    def get_behavior_tokens(self, behavior: str) -> list[str]:
        # No explicit behavior tokens in this dataset
        return []

    def token_count(self) -> int:
        return self.sole_item_len


class SMBExplicitDataset(BaseSMBDataset):
    """
    Session-wise multi-behavior dataset with explicit behavior tokens for sequential recommendation.
    The representation of the item with specific behavior will be like:
    `<behavior_token><item_token1><item_token2>...`,
    or
    `<item_token1><item_token2>...<behavior_token>`,
    where `<behavior_token>` is the token representing the behavior type.
    """

    def __init__(self, behavior_first: bool = True, **kwargs):
        self.behavior_first = behavior_first
        super().__init__(**kwargs)

    @property
    def cached_file_name(self) -> str:
        if self.behavior_first:
            return os.path.join(self.data_path, self.dataset + f".{self.__class__.__name__}.{self.max_his_len}.SMB.{self.mode}{self.index_suffix}.pkl")
        else:
            return os.path.join(self.data_path, self.dataset + f".{self.__class__.__name__}.{self.max_his_len}.SMB.behind.{self.mode}{self.index_suffix}.pkl")

    def _update_behavior_tokens(self):
        for behavior in self.behaviors:
            behavior_token = f"<behavior_{behavior}>"
            self.new_tokens.add(behavior_token)

    def get_behavior_item(self, item: str, behavior: str) -> str:
        behavior_token = f"<behavior_{behavior}>"
        if self.behavior_first:
            return behavior_token + item
        else:
            return item + behavior_token

    def get_behavior_tokens(self, behavior: str) -> list[str]:
        return [f"<behavior_{behavior}>"]

    def token_count(self) -> int:
        # Each item is represented by sole_item_len tokens, plus one behavior token
        return self.sole_item_len + 1
