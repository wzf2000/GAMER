import os
import json
import copy
import torch
import pickle
import numpy as np
from loguru import logger
from torch.utils.data import Dataset

from SeqRec.utils.pipe import set_seed, get_tqdm


class BaseSMBDataset(Dataset):
    """
    Base class for session-wise multi-behavior sequential recommendation datasets.
    """

    def __init__(self, dataset: str, data_path: str, max_his_len: int, index_file: str, mode: str):
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

        self.inter_suffix = "SMB.inter"

        # load data
        self._load_data()
        self._remap_items()

        # process data
        cached_file_name = os.path.join(self.data_path, self.dataset + f".{self.__class__.__name__}.{self.max_his_len}.SMB.{self.mode}.pkl")
        if os.path.exists(cached_file_name) and self.mode != 'train':
            if int(os.environ.get("LOCAL_RANK", 0)) == 0:
                logger.info(f"Loading cached interactions from {cached_file_name} for {self.mode}.")
            with open(cached_file_name, "rb") as f:
                self.inter_data = pickle.load(f)
            if int(os.environ.get("LOCAL_RANK", 0)) == 0:
                logger.info(f"Loaded cached {len(self.inter_data)} interactions from {cached_file_name} for {self.mode}.")
        else:
            if self.mode == "train":
                self.inter_data = self._process_train_data()
            elif self.mode == "valid":
                self.inter_data = self._process_valid_data()
            elif self.mode == "test":
                self.inter_data = self._process_test_data()
            elif self.mode == "valid_test":
                self.inter_data = self._process_valid_test_data()
            else:
                raise NotImplementedError
            with open(cached_file_name, "wb") as f:
                pickle.dump(self.inter_data, f)

        if int(os.environ.get("LOCAL_RANK", 0)) == 0:
            logger.info(f"Loaded {len(self.inter_data)} interactions for {self.mode} set.")

    def _load_data(self):
        with open(os.path.join(self.data_path, self.dataset + f".{self.inter_suffix}.json"), "r") as f:
            self.inters: dict[str, list[int]] = json.load(f)
        with open(os.path.join(self.data_path, self.dataset + ".SMB.behavior.json"), "r") as f:
            self.history_behaviors: dict[str, list[str]] = json.load(f)
        with open(os.path.join(self.data_path, self.dataset + self.index_file), "r") as f:
            self.indices: dict[str, list[str]] = json.load(f)
        with open(os.path.join(self.data_path, self.dataset + '.SMB.session.json'), 'r') as f:
            self.session: dict[str, list[int]] = json.load(f)
            self.train_pos: dict[str, dict[int, int]] = {}
            self.valid_pos: dict[str, int] = {}
            self.test_pos: dict[str, int] = {}
            for uid in self.session:
                min_sid = np.min(self.session[uid])
                self.session[uid] = [sid - min_sid for sid in self.session[uid]]  # Normalize session IDs to start from 0
                unique_sids = np.unique(self.session[uid])
                test_sid = unique_sids[-1]
                self.test_pos[uid] = np.where(self.session[uid] == test_sid)[0].min()
                if len(unique_sids) >= 2:
                    valid_sid = unique_sids[-2]
                    self.valid_pos[uid] = np.where(self.session[uid] == valid_sid)[0].min()
                else:
                    self.valid_pos[uid] = -1
                if len(unique_sids) >= 3:
                    train_sids = unique_sids[:-2]
                    self.train_pos[uid] = {sid: np.where(self.session[uid] == sid)[0].min() for sid in train_sids}
        with open(os.path.join(self.data_path, self.dataset + '.SMB.time.json'), 'r') as f:
            self.time: dict[str, list[str]] = json.load(f)
            import pandas as pd
            for uid in self.time:
                timestamps = pd.to_datetime(self.time[uid], format="%Y-%m-%d %H:%M:%S")
                base_time = timestamps[0]
                diffs = [t - base_time for t in timestamps]
                halfhour_diffs = [diff.total_seconds() / 1800 for diff in diffs]
                self.time[uid] = halfhour_diffs
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
        self.behaviors = list(self.behavior_level.keys())

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
            history_items = history_items[-self.max_his_len:]
            history_behaviors = history_behaviors[-self.max_his_len:]
        history_behavior_items = [
            self.get_behavior_item(history_item, history_behavior)
            for history_item, history_behavior in zip(history_items, history_behaviors)
        ]
        return "".join(history_behavior_items)

    def _get_inters_with_only_items(self, history_items: list[str]) -> list[str]:
        if self.max_his_len > 0:
            history_items = history_items[-self.max_his_len:]
        return history_items

    def _generate_session_ids(self, session_ids: list[int], items: list[str], behaviors: list[str]) -> list[int]:
        assert len(session_ids) == len(items) == len(behaviors), (
            f"Session IDs, items, and behaviors must have the same length. "
            f"Got {len(session_ids)}, {len(items)}, and {len(behaviors)}."
        )
        ret = []
        max_his_len = self.max_his_len
        if self.mode in ["train", "valid"]:
            max_his_len += 1
        session_ids = session_ids[-max_his_len:]
        items = items[-max_his_len:]
        behaviors = behaviors[-max_his_len:]
        for sid, item, behavior in zip(session_ids, items, behaviors):
            item_rep = self.get_behavior_item(item, behavior)
            # TODO: generalize this to handle different item representations
            # get the token count for the item representation, suppose each token is like <XXX>
            assert item_rep.count("<") == item_rep.count(">"), (
                f"Item representation '{item_rep}' must have balanced '<' and '>' tokens."
            )
            token_count = item_rep.count("<")  # Each token is like <XXX>, so count '<' to get the number of tokens
            ret.extend([sid] * token_count)
        return ret

    def _generate_extended_session_ids(self, session_ids: list[int], items: list[str], behaviors: list[str]) -> list[int]:
        assert len(session_ids) == len(items) == len(behaviors), (
            f"Session IDs, items, and behaviors must have the same length. "
            f"Got {len(session_ids)}, {len(items)}, and {len(behaviors)}."
        )
        ret = []
        max_his_len = self.max_his_len
        if self.mode in ["train", "valid"]:
            max_his_len += 1
        session_ids = session_ids[-max_his_len:]
        items = items[-max_his_len:]
        behaviors = behaviors[-max_his_len:]
        last_sid: int | None = None
        remapped_sid = -1
        for sid, item, behavior in zip(session_ids, items, behaviors):
            item_rep = self.get_behavior_item(item, behavior)
            # TODO: generalize this to handle different item representations
            # get the token count for the item representation, suppose each token is like <XXX>
            assert item_rep.count("<") == item_rep.count(">"), (
                f"Item representation '{item_rep}' must have balanced '<' and '>' tokens."
            )
            token_count = item_rep.count("<")  # Each token is like <XXX>, so count '<' to get the number of tokens
            if last_sid != sid:
                last_sid = sid
                remapped_sid += 1
            ret.extend([remapped_sid * token_count + i for i in range(token_count)])
        return ret

    def _generate_attention_mask(self, session_ids: list[int], items: list[str], behaviors: list[str]) -> torch.FloatTensor:
        assert len(session_ids) == len(items) == len(behaviors), (
            f"Session IDs, items, and behaviors must have the same length. "
            f"Got {len(session_ids)}, {len(items)}, and {len(behaviors)}."
        )
        max_his_len = self.max_his_len
        if self.mode in ["train", "valid"]:
            max_his_len += 1
        session_ids = session_ids[-max_his_len:]
        items = items[-max_his_len:]
        behaviors = behaviors[-max_his_len:]
        total_len = 0
        for sid, item, behavior in zip(session_ids, items, behaviors):
            item_rep = self.get_behavior_item(item, behavior)
            # TODO: generalize this to handle different item representations
            # get the token count for the item representation, suppose each token is like <XXX>
            assert item_rep.count("<") == item_rep.count(">"), (
                f"Item representation '{item_rep}' must have balanced '<' and '>' tokens."
            )
            token_count = item_rep.count("<")  # Each token is like <XXX>, so count '<' to get the number of tokens
            total_len += token_count
        attention_mask = torch.full(
            (total_len, total_len), fill_value=torch.finfo(torch.float32).min, dtype=torch.float32
        )
        last_sid: int | None = None
        accum_tokens = 0
        last_session_pos = 0
        for sid, item, behavior in zip(session_ids, items, behaviors):
            item_rep = self.get_behavior_item(item, behavior)
            token_count = item_rep.count("<")  # Each token is like <XXX>, so count '<' to get the number of tokens
            if last_sid != sid:
                last_sid = sid
                last_session_pos = accum_tokens
            if last_session_pos > 0:
                attention_mask[accum_tokens: accum_tokens + token_count, :last_session_pos] = 0  # set the attention mask to 0 for previous session items
            attention_mask[accum_tokens: accum_tokens + token_count, accum_tokens: accum_tokens + token_count] *= torch.triu(torch.ones((token_count, token_count), dtype=torch.float32), diagonal=1)  # set the attention mask to 0 within the current item by a triangular matrix
            accum_tokens += token_count
        return attention_mask

    def _generate_times(self, times: list[float], items: list[str], behaviors: list[str]) -> list[float]:
        assert len(times) == len(items) == len(behaviors), (
            f"Session IDs, items, and behaviors must have the same length. "
            f"Got {len(times)}, {len(items)}, and {len(behaviors)}."
        )
        ret = []
        max_his_len = self.max_his_len + 1
        times = times[-max_his_len:]
        base_time = times[-1]
        times = [abs(t - base_time) for t in times]
        times = times[-max_his_len:-1]
        items = items[-max_his_len:-1]
        behaviors = behaviors[-max_his_len:-1]
        for time, item, behavior in zip(times, items, behaviors):
            item_rep = self.get_behavior_item(item, behavior)
            # TODO: generalize this to handle different item representations
            # get the token count for the item representation, suppose each token is like <XXX>
            assert item_rep.count("<") == item_rep.count(">"), (
                f"Item representation '{item_rep}' must have balanced '<' and '>' tokens."
            )
            token_count = item_rep.count("<")  # Each token is like <XXX>, so count '<' to get the number of tokens
            ret.extend([time] * token_count)
        return ret

    def _process_train_data(self) -> list[dict[str, str | list[int] | torch.FloatTensor]]:
        inter_data = []
        pbar = get_tqdm(self.remapped_inters, desc="Processing training data")
        for uid in pbar:
            if self.valid_pos[uid] <= 0:
                continue
            items = self.remapped_inters[uid][: self.valid_pos[uid]]
            behaviors = self.history_behaviors[uid][: self.valid_pos[uid]]
            times = self.time[uid][: self.valid_pos[uid]]
            for i in range(1, len(items)):
                pos = self.train_pos[uid][self.session[uid][i]]
                inter_data.append({
                    "item": self.get_behavior_item(items[i], behaviors[i]),
                    "inters": self._get_inters(items[:pos], behaviors[:pos]),
                    "session_ids": self._generate_session_ids(self.session[uid][:pos] + [self.session[uid][i]], items[:pos] + [items[i]], behaviors[:pos] + [behaviors[i]]),
                    "extended_session_ids": self._generate_extended_session_ids(self.session[uid][:pos] + [self.session[uid][i]], items[:pos] + [items[i]], behaviors[:pos] + [behaviors[i]]),
                    "attention_mask": self._generate_attention_mask(self.session[uid][:pos] + [self.session[uid][i]], items[:pos] + [items[i]], behaviors[:pos] + [behaviors[i]]),
                    "time": self._generate_times(times[:pos + 1], items[:pos + 1], behaviors[:pos + 1]),
                    "behavior": behaviors[i],
                })

        return inter_data

    def _process_valid_data(self) -> list[dict[str, str | list[int] | torch.FloatTensor]]:
        inter_data = []
        pbar = get_tqdm(self.remapped_inters, desc="Processing validation data")
        for uid in pbar:
            if self.valid_pos[uid] < 0:
                continue
            items = self.remapped_inters[uid][: self.test_pos[uid]]
            behaviors = self.history_behaviors[uid][: self.test_pos[uid]]
            times = self.time[uid][: self.test_pos[uid]]
            for i in range(self.valid_pos[uid], len(items)):
                pos = self.valid_pos[uid]
                inter_data.append({
                    "item": self.get_behavior_item(items[i], behaviors[i]),
                    "inters": self._get_inters(items[:pos], behaviors[:pos]),
                    "session_ids": self._generate_session_ids(self.session[uid][:pos] + [self.session[uid][i]], items[:pos] + [items[i]], behaviors[:pos] + [behaviors[i]]),
                    "extended_session_ids": self._generate_extended_session_ids(self.session[uid][:pos] + [self.session[uid][i]], items[:pos] + [items[i]], behaviors[:pos] + [behaviors[i]]),
                    "attention_mask": self._generate_attention_mask(self.session[uid][:pos] + [self.session[uid][i]], items[:pos] + [items[i]], behaviors[:pos] + [behaviors[i]]),
                    "time": self._generate_times(times[:pos + 1], items[:pos + 1], behaviors[:pos + 1]),
                    "behavior": behaviors[i],
                })

        return inter_data

    def _process_valid_test_data(self) -> list[dict[str, str | list[str] | list[int] | torch.FloatTensor]]:
        inter_data = []
        pbar = get_tqdm(self.remapped_inters, desc="Processing validation data for testing")
        for uid in pbar:
            items = self.remapped_inters[uid][: self.test_pos[uid]]
            behaviors = self.history_behaviors[uid][: self.test_pos[uid]]
            times = self.time[uid][: self.test_pos[uid]]
            session_items: list[str] = []
            session_behaviors: list[str] = []
            for i in range(self.valid_pos[uid], len(items)):
                session_items.append(self.get_behavior_item(items[i], behaviors[i]))
                session_behaviors.append(behaviors[i])
            assert len(session_items) > 0, f"Session for user {uid} is empty after valid position {self.valid_pos[uid]}."
            inter_data.append({
                "item": session_items,
                "inters": self._get_inters(items[:self.valid_pos[uid]], behaviors[:self.valid_pos[uid]]),
                "inters_item_list": self._get_inters_with_only_items(items[:self.valid_pos[uid]]),
                # ! For test set, we donot add session IDs for the item to be predicted, and the session IDs should be add by the inference code.
                "session_ids": self._generate_session_ids(self.session[uid][:self.valid_pos[uid]], items[:self.valid_pos[uid]], behaviors[:self.valid_pos[uid]]),
                "extended_session_ids": self._generate_extended_session_ids(self.session[uid][:self.valid_pos[uid]], items[:self.valid_pos[uid]], behaviors[:self.valid_pos[uid]]),
                "attention_mask": self._generate_attention_mask(self.session[uid][:self.valid_pos[uid]], items[:self.valid_pos[uid]], behaviors[:self.valid_pos[uid]]),
                "time": self._generate_times(times[:self.valid_pos[uid] + 1], items[:self.valid_pos[uid] + 1], behaviors[:self.valid_pos[uid] + 1]),
                "behavior": session_behaviors,
            })

        return inter_data

    def _process_test_data(self) -> list[dict[str, str | list[str] | list[int] | torch.FloatTensor]]:
        inter_data = []
        pbar = get_tqdm(self.remapped_inters, desc="Processing test data")
        for uid in pbar:
            items = self.remapped_inters[uid]
            behaviors = self.history_behaviors[uid]
            times = self.time[uid]
            session_items: list[str] = []
            session_behaviors: list[str] = []
            for i in range(self.test_pos[uid], len(items)):
                session_items.append(self.get_behavior_item(items[i], behaviors[i]))
                session_behaviors.append(behaviors[i])
            assert len(session_items) > 0, f"Session for user {uid} is empty after test position {self.test_pos[uid]}."
            inter_data.append({
                "item": session_items,
                "inters": self._get_inters(items[:self.test_pos[uid]], behaviors[:self.test_pos[uid]]),
                "inters_item_list": self._get_inters_with_only_items(items[:self.test_pos[uid]]),
                # ! For test set, we donot add session IDs for the item to be predicted, and the session IDs should be add by the inference code.
                "session_ids": self._generate_session_ids(self.session[uid][:self.test_pos[uid]], items[:self.test_pos[uid]], behaviors[:self.test_pos[uid]]),
                "extended_session_ids": self._generate_extended_session_ids(self.session[uid][:self.test_pos[uid]], items[:self.test_pos[uid]], behaviors[:self.test_pos[uid]]),
                "attention_mask": self._generate_attention_mask(self.session[uid][:self.test_pos[uid]], items[:self.test_pos[uid]], behaviors[:self.test_pos[uid]]),
                "time": self._generate_times(times[:self.test_pos[uid] + 1], items[:self.test_pos[uid] + 1], behaviors[:self.test_pos[uid] + 1]),
                "behavior": session_behaviors,
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

    def filter_by_behavior(self, behavior: str) -> "BaseSMBDataset":
        if isinstance(self.inter_data[0]['behavior'], list):
            filtered_data = []
            inter_data = get_tqdm(self.inter_data, desc=f"Filtering by behavior - {behavior}")
            for d in inter_data:
                if behavior not in d["behavior"]:
                    continue
                items, behaviors = [], []
                for sample_item, sample_behavior in zip(d["item"], d["behavior"]):
                    if sample_behavior == behavior:
                        items.append(sample_item)
                        behaviors.append(sample_behavior)
                filtered_data.append({
                    "item": items,
                    "inters": d["inters"],
                    "session_ids": d["session_ids"],
                    "extended_session_ids": d["extended_session_ids"],
                    "attention_mask": d["attention_mask"],
                    "behavior": behaviors,
                    "time": d["time"],
                })
        else:
            filtered_data = [
                d for d in self.inter_data if d["behavior"] == behavior
            ]
        copied_dataset = copy.deepcopy(self)
        copied_dataset.inter_data = filtered_data
        copied_dataset.target_behavior = behavior
        return copied_dataset

    def __len__(self) -> int:
        return len(self.inter_data)

    def __getitem__(self, index: int) -> dict[str, str | list[str] | list[int] | torch.FloatTensor]:
        d = self.inter_data[index]
        return dict(input_ids=d["inters"], labels=d["item"], behavior=d["behavior"], session_ids=d["session_ids"], extended_session_ids=d["extended_session_ids"], time=d["time"], attention_mask=d["attention_mask"], inters_item_list=d.get("inters_item_list", []), split=self.mode)


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


class SMBExplicitDatasetForDecoder(SMBExplicitDataset):
    def __init__(self, augment: int | None = None, **kwargs):
        self.augment = augment  # Times of augmentation for each interaction (for training only)
        if augment is not None and augment < 1:
            raise ValueError("augment must be greater than or equal to 1")
        super().__init__(**kwargs)

    def _augment_interactions(self, items: list[str], behaviors: list[str], sids: list[int], times: list[float]) -> tuple[list[list[str]], list[list[str]], list[list[int]], list[list[float]]]:
        if not self.augment:
            return [items], [behaviors], [sids], [times]
        downsample_ratios = np.arange(1, self.augment + 1) / self.augment
        behavior_indices = {}
        for behavior in self.behavior_level:
            behavior_indices[behavior] = [i for i, b in enumerate(behaviors) if b == behavior]
        items_list = [items]
        behaviors_list = [behaviors]
        sids_list = [sids]
        times_list = [times]
        for ratio in downsample_ratios:
            if ratio == 0:
                continue
            drop_indices = []
            for behavior, level in self.behavior_level.items():
                if level == self.max_behavior_level:
                    continue  # Skip the target behavior
                if behavior not in behavior_indices or len(behavior_indices[behavior]) == 0:
                    continue
                behavior_ratio = ratio / (level + 1)  # downsample ratio for each behavior
                drop_num = int(len(behavior_indices[behavior]) * behavior_ratio)
                if drop_num > 0:
                    drop_indices.extend(np.random.choice(behavior_indices[behavior], drop_num, replace=False).tolist())
            drop_mask = np.ones(len(items), dtype=bool)
            drop_mask[drop_indices] = False
            items_copy = copy.deepcopy(items)
            behaviors_copy = copy.deepcopy(behaviors)
            sids_copy = copy.deepcopy(sids)
            times_copy = copy.deepcopy(times)
            items_array = np.array(items_copy)
            behaviors_array = np.array(behaviors_copy)
            sids_array = np.array(sids_copy)
            times_array = np.array(times_copy)
            items_copy: list[str] = items_array[drop_mask].tolist()
            behaviors_copy: list[str] = behaviors_array[drop_mask].tolist()
            sids_copy: list[int] = sids_array[drop_mask].tolist()
            times_copy: list[float] = times_array[drop_mask].tolist()
            if len(items_copy) < 2:
                continue
            items_list.append(items_copy)
            behaviors_list.append(behaviors_copy)
            sids_list.append(sids_copy)
            times_list.append(times_copy)
        return items_list, behaviors_list, sids_list, times_list

    def _process_train_data(self) -> list[dict[str, str]]:
        set_seed(42)  # For reproducibility
        inter_data = []
        if self.augment and int(os.environ.get("LOCAL_RANK", 0)) == 0:
            logger.info(f"Augmenting interactions {self.augment} times for each user.")
        pbar = get_tqdm(self.remapped_inters, desc="Processing training data")
        for uid in pbar:
            if self.valid_pos[uid] <= 0:
                continue
            items = self.remapped_inters[uid][:self.valid_pos[uid]]
            behaviors = self.history_behaviors[uid][:self.valid_pos[uid]]
            sids = self.session[uid][:self.valid_pos[uid]]
            times = self.time[uid][:self.valid_pos[uid]]
            items_list, behaviors_list, sids_list, times_list = self._augment_interactions(items, behaviors, sids, times)
            for items, behaviors, sids, times in zip(items_list, behaviors_list, sids_list, times_list):
                inter_data.append({
                    "item": self.get_behavior_item(items[-1], behaviors[-1]),
                    "inters": self._get_inters(items[:-1], behaviors[:-1]),
                    "session_ids": self._generate_session_ids(sids, items, behaviors),
                    "extended_session_ids": self._generate_extended_session_ids(sids, items, behaviors),
                    "attention_mask": self._generate_attention_mask(sids, items, behaviors),
                    "time": self._generate_times(times, items, behaviors),
                    "behavior": behaviors[-1],
                })

        return inter_data
