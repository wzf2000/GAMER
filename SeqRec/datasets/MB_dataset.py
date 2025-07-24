import os
import json
import copy
from enum import Enum
from tqdm import tqdm
from loguru import logger
from torch.utils.data import Dataset


class EvaluationType(Enum):
    TARGET_BEHAVIOR = "Target Behavior"  # Target behavior item prediction
    BEHAVIOR_SPECIFIC = "Behavior Specific"  # Behavior-specific item prediction
    BEHAVIOR_ITEM = "Behavior Item"  # Behavior-item prediction


class BaseMBDataset(Dataset):
    """
    Base class for multi-behavior sequential recommendation datasets.
    """
    behaviors: list[str]

    def __init__(self, dataset: str, data_path: str, max_his_len: int, index_file: str, mode: str, filter_target: bool = False):
        super().__init__()

        self.dataset: str = dataset
        self.data_path = os.path.join(data_path, self.dataset)

        self.max_his_len: int = max_his_len
        self.index_file: str = index_file

        self.new_tokens = None
        self.allowed_tokens = None
        self.all_items = None
        self.all_items_by_behavior: dict[str, set[str]] = {}
        self.mode = mode
        self.filter_target = filter_target

        self.inter_suffix = "MB.inter"

        # load data
        self._load_data()
        self._remap_items()

        # load data
        if self.mode == "train":
            self.inter_data = self._process_train_data()
        elif self.mode == "valid":
            self.inter_data = self._process_valid_data()
        elif self.mode == "test":
            self.inter_data = self._process_test_data()
        else:
            raise NotImplementedError

        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            logger.info(f"Loaded {len(self.inter_data)} interactions for {self.mode} set.")

        assert os.path.exists(os.path.join(self.data_path, self.dataset + '.behavior_level.json')), (
            f"Behavior level file {self.data_path}/{self.dataset}.behavior_level.json does not exist."
        )
        with open(os.path.join(self.data_path, self.dataset + '.behavior_level.json'), 'r') as f:
            self.behavior_level: dict[str, int] = json.load(f)
        # get the max level of behaviors
        self.max_behavior_level = max(self.behavior_level.values())
        # get the target behavior
        max_level_behaviors = [b for b, level in self.behavior_level.items() if level == self.max_behavior_level]
        assert len(max_level_behaviors) == 1, (
            f"Expected exactly one target behavior with max level, but found {len(max_level_behaviors)}: {max_level_behaviors}"
        )
        self.target_behavior = max_level_behaviors[0]

    def _load_data(self):
        with open(os.path.join(self.data_path, self.dataset + f".{self.inter_suffix}.json"), "r") as f:
            self.inters: dict[str, list[int]] = json.load(f)
        with open(os.path.join(self.data_path, self.dataset + ".MB.behavior.json"), "r") as f:
            self.history_behaviors: dict[str, list[str]] = json.load(f)
        with open(
            os.path.join(self.data_path, self.dataset + self.index_file), "r"
        ) as f:
            self.indices: dict[str, list[str]] = json.load(f)
        all_behaviors = set()
        for behaviors in self.history_behaviors.values():
            all_behaviors.update(behaviors)
        self.behaviors = sorted(list(all_behaviors))

    def _remap_items(self):
        self.remapped_inters: dict[str, list[str]] = dict()
        for uid, items in self.inters.items():
            new_items = ["".join(self.indices[str(i)]) for i in items]
            self.remapped_inters[uid] = new_items

    def _update_behavior_tokens(self):
        raise NotImplementedError(
            "This method should be implemented in subclasses to update behavior tokens."
        )

    def get_behavior_item(self, item: str, behavior: str) -> str:
        raise NotImplementedError(
            "This method should be implemented in subclasses to return the behavior-item representation."
        )

    def get_behavior_tokens(self, behavior: str) -> list[str]:
        raise NotImplementedError(
            "This method should be implemented in subclasses to return the behavior token."
        )

    def _get_inters(self, history_items: list[str], history_behaviors: list[str]) -> str:
        if self.max_his_len > 0:
            history_items = history_items[-(self.max_his_len + 1) : -1]
            history_behaviors = history_behaviors[-(self.max_his_len + 1) : -1]
        if self.filter_target:
            target_item = history_items[-1]
            target_behavior = history_behaviors[-1]
            non_duplicate_ids = [i for i in range(len(history_items)) if history_items[i] != target_item or history_behaviors[i] >= target_behavior]
            history_items = [history_items[i] for i in non_duplicate_ids]
            history_behaviors = [history_behaviors[i] for i in non_duplicate_ids]
        history_behavior_items = [
            self.get_behavior_item(history_item, history_behavior)
            for history_item, history_behavior in zip(history_items, history_behaviors)
        ]
        return "".join(history_behavior_items)

    def _process_train_data(self) -> list[dict[str, str]]:
        inter_data = []
        pbar = tqdm(self.remapped_inters) if int(os.environ.get("LOCAL_RANK", 0)) == 0 else self.remapped_inters
        for uid in pbar:
            items = self.remapped_inters[uid][:-2]
            behaviors = self.history_behaviors[uid][:-2]
            for i in range(1, len(items)):
                inter_data.append({
                    "item": self.get_behavior_item(items[i], behaviors[i]),
                    "inters": self._get_inters(items[:i + 1], behaviors[:i + 1]),
                    "behavior": behaviors[i],
                })

        return inter_data

    def _process_valid_data(self) -> list[dict[str, str]]:
        inter_data = []
        for uid in self.remapped_inters:
            items = self.remapped_inters[uid]
            behaviors = self.history_behaviors[uid]
            inter_data.append({
                "item": self.get_behavior_item(items[-2], behaviors[-2]),
                "inters": self._get_inters(items[:-1], behaviors[:-1]),
                "behavior": behaviors[-2],
            })

        return inter_data

    def _process_test_data(self) -> list[dict[str, str]]:
        inter_data = []
        for uid in self.remapped_inters:
            items = self.remapped_inters[uid]
            behaviors = self.history_behaviors[uid]
            inter_data.append({
                "item": self.get_behavior_item(items[-1], behaviors[-1]),
                "inters": self._get_inters(items, behaviors),
                "behavior": behaviors[-1],
            })

        return inter_data

    def get_new_tokens(self) -> list[str]:
        if self.new_tokens is not None:
            return self.new_tokens

        self.new_tokens = set()
        for index in self.indices.values():
            for token in index:
                self.new_tokens.add(token)
        self._update_behavior_tokens()
        self.new_tokens = sorted(list(self.new_tokens))

        return self.new_tokens

    def _get_all_items_by_behavior(self, all_items: set[str], behavior: str) -> set[str]:
        if behavior in self.all_items_by_behavior:
            return self.all_items_by_behavior[behavior]

        self.all_items_by_behavior[behavior] = set()
        for item in all_items:
            new_item = self.get_behavior_item(item, behavior)
            self.all_items_by_behavior[behavior].add(new_item)

        return self.all_items_by_behavior[behavior]

    def get_all_items(self, behavior: str | None = None) -> set[str]:
        if self.all_items is not None and behavior is None:
            return self.all_items
        if behavior is not None and behavior in self.all_items_by_behavior:
            return self.all_items_by_behavior[behavior]

        if self.all_items is None:
            self.all_items: set[str] = set()
            self.collision_items: set[str] = set()
            for index in self.indices.values():
                item_str = "".join(index)
                if item_str in self.all_items:
                    self.collision_items.add(item_str)
                else:
                    self.all_items.add(item_str)

        if behavior is None:
            return self.all_items

        if behavior == "all":
            all_items = set()
            for b in self.behaviors:
                all_items.update(self._get_all_items_by_behavior(self.all_items, b))
            return all_items
        elif behavior not in self.behaviors:
            raise ValueError(f"Behavior '{behavior}' is not in the dataset behaviors: {self.behaviors}")
        else:
            return self._get_all_items_by_behavior(self.all_items, behavior)

    def filter_by_behavior(self, behavior: str) -> "BaseMBDataset":
        filtered_data = [
            d for d in self.inter_data if d["behavior"] == behavior
        ]
        copied_dataset = copy.deepcopy(self)
        copied_dataset.inter_data = filtered_data
        copied_dataset.target_behavior = behavior
        return copied_dataset

    def __len__(self) -> int:
        return len(self.inter_data)

    def __getitem__(self, index: int) -> dict[str, str]:
        d = self.inter_data[index]
        return dict(input_ids=d["inters"], labels=d["item"], behavior=d["behavior"])


class MBDataset(BaseMBDataset):
    """
    Multi-behavior dataset without any explicit behavior tokens for sequential recommendation.
    The representation of the item with specific behavior will be like:
    `<item_token1><item_token2>...`,
    where `<item_token>` is the token representing the item.
    """

    def __init__(
        self,
        dataset: str,
        data_path: str,
        max_his_len: int,
        index_file: str = ".index.json",
        mode: str = "train",
        filter_target: bool = False,
    ):
        super().__init__(dataset, data_path, max_his_len, index_file, mode, filter_target)

    def _update_behavior_tokens(self):
        pass

    def get_behavior_item(self, item: str, behavior: str) -> str:
        # In this case, we do not use any explicit behavior token, just return the item token
        return item

    def get_behavior_tokens(self, behavior: str) -> list[str]:
        # No explicit behavior tokens in this dataset
        return []


class MBExplicitTokenDataset(BaseMBDataset):
    """
    Multi-behavior dataset with explicit behavior tokens for sequential recommendation.
    The representation of the item with specific behavior will be like:
    `<behavior_token><item_token1><item_token2>...`,
    or
    `<item_token1><item_token2>...<behavior_token>`,
    where `<behavior_token>` is the token representing the behavior type.
    """

    def __init__(
        self,
        dataset: str,
        data_path: str,
        max_his_len: int,
        index_file: str = ".index.json",
        mode: str = "train",
        behavior_first: bool = True,
        filter_target: bool = False,
    ):
        self.behavior_first = behavior_first
        super().__init__(dataset, data_path, max_his_len, index_file, mode, filter_target)

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
