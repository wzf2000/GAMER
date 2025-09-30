import os
import json
import copy
import pickle
import numpy as np
import pandas as pd
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

        # load data
        self._load_data()
        self._remap_items()

        # process data
        if os.path.exists(self.cached_file_name):
            logger.info(f"Loading cached interactions from {self.cached_file_name} for {self.mode}.")
            with open(self.cached_file_name, "rb") as f:
                self.inter_data = pickle.load(f)
            logger.info(f"Loaded cached {len(self.inter_data)} interactions from {self.cached_file_name} for {self.mode}.")
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
            if int(os.environ.get("LOCAL_RANK", 0)) == 0:
                with open(self.cached_file_name, "wb") as f:
                    pickle.dump(self.inter_data, f)

        logger.info(f"Loaded {len(self.inter_data)} interactions for {self.mode} set.")

    @property
    def index_suffix(self) -> str:
        if self.index_file == '.index.json':
            return ''
        else:
            # looks like .index.XXX.json
            # we should get XXX
            return '.' + self.index_file[len('.index.'): -len('.json')]

    @property
    def cached_file_name(self) -> str:
        return os.path.join(self.data_path, self.dataset + f".{self.__class__.__name__}.{self.max_his_len}.SMB.{self.mode}{self.index_suffix}.pkl")

    def _load_data(self):
        with open(os.path.join(self.data_path, self.dataset + ".SMB.inter.json"), "r") as f:
            self.inters: dict[str, list[int]] = json.load(f)
        with open(os.path.join(self.data_path, self.dataset + ".SMB.behavior.json"), "r") as f:
            self.history_behaviors: dict[str, list[str]] = json.load(f)
        with open(os.path.join(self.data_path, self.dataset + self.index_file), "r") as f:
            self.indices: dict[str, list[str]] = json.load(f)
            # check if all the indices are the same length
        index_lengths = {len(v) for v in self.indices.values()}
        assert len(index_lengths) == 1, f"All indices must have the same length, but got lengths: {index_lengths}"
        self.sole_item_len = index_lengths.pop()

        cached_processed_data_file = os.path.join(self.data_path, self.dataset + ".SMB.data.pkl")
        if os.path.exists(cached_processed_data_file):
            logger.info(f"Loading cached processed data from {cached_processed_data_file}.")
            with open(cached_processed_data_file, "rb") as f:
                cached_data = pickle.load(f)
                self.session = cached_data["session"]
                self.train_pos = cached_data["train_pos"]
                self.valid_pos = cached_data["valid_pos"]
                self.test_pos = cached_data["test_pos"]
                self.time = cached_data["time"]
        else:
            with open(os.path.join(self.data_path, self.dataset + '.SMB.session.json'), 'r') as f:
                self.session: dict[str, list[int]] = json.load(f)
                self.train_pos: dict[str, dict[int, int]] = {}
                self.valid_pos: dict[str, int] = {}
                self.test_pos: dict[str, int] = {}
                for uid in get_tqdm(self.session, desc="Processing session data"):
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
                raw_time: dict[str, list[str]] = json.load(f)
                self.time: dict[str, list[float]] = {}
                for uid in get_tqdm(raw_time, desc="Processing time data"):
                    timestamps = pd.to_datetime(raw_time[uid], format="%Y-%m-%d %H:%M:%S")
                    base_time = timestamps[0]
                    diffs = [t - base_time for t in timestamps]
                    halfhour_diffs = [diff.total_seconds() / 1800 for diff in diffs]
                    self.time[uid] = halfhour_diffs
            if int(os.environ.get("LOCAL_RANK", 0)) == 0:
                with open(cached_processed_data_file, "wb") as f:
                    pickle.dump({
                        "session": self.session,
                        "train_pos": self.train_pos,
                        "valid_pos": self.valid_pos,
                        "test_pos": self.test_pos,
                        "time": self.time,
                    }, f)

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
        for uid, items in get_tqdm(self.inters.items(), desc="Remapping items"):
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

    def token_count(self) -> int:
        raise NotImplementedError(
            "This method should be implemented in subclasses to return the token count."
        )

    def _get_inters(self, history_items: list[str], history_behaviors: list[str], max_his_len: int | None = None) -> str:
        if max_his_len is None:
            max_his_len = self.max_his_len
        if max_his_len > 0:
            history_items = history_items[-max_his_len:]
            history_behaviors = history_behaviors[-max_his_len:]
        history_behavior_items = [
            self.get_behavior_item(history_item, history_behavior)
            for history_item, history_behavior in zip(history_items, history_behaviors)
        ]
        return "".join(history_behavior_items)

    def _get_inters_with_only_items(self, history_items: list[str], max_his_len: int | None = None) -> list[str]:
        if max_his_len is None:
            max_his_len = self.max_his_len
        if max_his_len > 0:
            history_items = history_items[-max_his_len:]
        return history_items

    def _generate_session_ids(self, session_ids: list[int], max_his_len: int | None = None) -> list[int]:
        ret = []
        if max_his_len is None:
            max_his_len = self.max_his_len
        if max_his_len > 0:
            if self.mode in ["train", "valid"]:
                max_his_len += 1
            session_ids = session_ids[-max_his_len:]
        for sid in session_ids:
            ret.extend([sid] * self.token_count())
        return ret

    def _generate_extended_session_ids(self, session_ids: list[int], max_his_len: int | None = None) -> list[int]:
        ret = []
        if max_his_len is None:
            max_his_len = self.max_his_len
        if max_his_len > 0:
            if self.mode in ["train", "valid"]:
                max_his_len += 1
            session_ids = session_ids[-max_his_len:]
        last_sid: int | None = None
        remapped_sid = -1
        for sid in session_ids:
            token_count = self.token_count()
            if last_sid != sid:
                last_sid = sid
                remapped_sid += 1
            ret.extend([remapped_sid * token_count + i for i in range(token_count)])
        return ret

    def _generate_actions(self, actions: list[int], max_his_len: int | None = None) -> list[int]:
        ret = []
        if max_his_len is None:
            max_his_len = self.max_his_len
        if max_his_len > 0:
            if self.mode in ["train", "valid"]:
                max_his_len += 1
            actions = actions[-max_his_len:]
        for action in actions:
            ret.extend([self.behavior_level[action]] * self.token_count())
        return ret

    def _generate_times(self, times: list[float], max_his_len: int | None = None) -> list[float]:
        ret = []
        base_time = times[-1]
        times = [abs(t - base_time) for t in times]
        if max_his_len is None:
            max_his_len = self.max_his_len
        if max_his_len > 0:
            max_his_len += 1
            times = times[-max_his_len:]
        times = times[:-1]
        for time in times:
            ret.extend([time] * self.token_count())
        return ret

    def _process_train_data(self) -> list[dict[str, str | list[int] | list[float]]]:
        inter_data = []
        for uid in get_tqdm(self.remapped_inters, desc="Processing training data"):
            if self.valid_pos[uid] <= 0:
                continue
            items = self.remapped_inters[uid][: self.valid_pos[uid]]
            behaviors = self.history_behaviors[uid][: self.valid_pos[uid]]
            times = self.time[uid][: self.valid_pos[uid]]
            session_ids_map = {}
            extended_session_ids_map = {}
            times_map = {}
            for i in range(1, len(items)):
                sid = self.session[uid][i]
                pos = self.train_pos[uid][sid]
                if sid not in session_ids_map:
                    session_ids_map[sid] = self._generate_session_ids(self.session[uid][:pos + 1])
                    extended_session_ids_map[sid] = self._generate_extended_session_ids(self.session[uid][:pos + 1])
                    times_map[sid] = self._generate_times(times[:pos + 1])
                inter_data.append({
                    "item": self.get_behavior_item(items[i], behaviors[i]),
                    "inters": self._get_inters(items[:pos], behaviors[:pos]),
                    "session_ids": session_ids_map[sid],
                    "extended_session_ids": extended_session_ids_map[sid],
                    "actions": self._generate_actions(behaviors[:pos] + [behaviors[i]]),
                    "time": times_map[sid],
                    "behavior": behaviors[i],
                })

        return inter_data

    def _process_valid_data(self) -> list[dict[str, str | list[int] | list[float]]]:
        inter_data = []
        for uid in get_tqdm(self.remapped_inters, desc="Processing validation data"):
            if self.valid_pos[uid] < 0:
                continue
            items = self.remapped_inters[uid][: self.test_pos[uid]]
            behaviors = self.history_behaviors[uid][: self.test_pos[uid]]
            times = self.time[uid][: self.test_pos[uid]]
            pos = self.valid_pos[uid]
            session_ids = self._generate_session_ids(self.session[uid][: pos + 1])
            extended_session_ids = self._generate_extended_session_ids(self.session[uid][: pos + 1])
            times = self._generate_times(times[: pos + 1])
            for i in range(pos, len(items)):
                inter_data.append({
                    "item": self.get_behavior_item(items[i], behaviors[i]),
                    "inters": self._get_inters(items[:pos], behaviors[:pos]),
                    "session_ids": session_ids,
                    "extended_session_ids": extended_session_ids,
                    "actions": self._generate_actions(self.history_behaviors[uid][:pos] + [behaviors[i]]),
                    "time": times,
                    "behavior": behaviors[i],
                })

        return inter_data

    def _process_valid_test_data(self) -> list[dict[str, str | list[str] | list[int] | list[float]]]:
        inter_data = []
        for uid in get_tqdm(self.remapped_inters, desc="Processing validation data for testing"):
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
                "session_ids": self._generate_session_ids(self.session[uid][:self.valid_pos[uid]]),
                "extended_session_ids": self._generate_extended_session_ids(self.session[uid][:self.valid_pos[uid]]),
                "actions": self._generate_actions(self.history_behaviors[uid][:self.valid_pos[uid]]),
                "time": self._generate_times(times[:self.valid_pos[uid] + 1]),
                "behavior": session_behaviors,
            })

        return inter_data

    def _process_test_data(self) -> list[dict[str, str | list[str] | list[int] | list[float]]]:
        inter_data = []
        for uid in get_tqdm(self.remapped_inters, desc="Processing test data"):
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
                "session_ids": self._generate_session_ids(self.session[uid][:self.test_pos[uid]]),
                "extended_session_ids": self._generate_extended_session_ids(self.session[uid][:self.test_pos[uid]]),
                "actions": self._generate_actions(self.history_behaviors[uid][:self.test_pos[uid]]),
                "time": self._generate_times(times[:self.test_pos[uid] + 1]),
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
                    "inters_item_list": d["inters_item_list"],
                    "session_ids": d["session_ids"],
                    "actions": d["actions"],
                    "extended_session_ids": d["extended_session_ids"],
                    "behavior": behaviors,
                    "time": d["time"],
                })
        else:
            filtered_data = [
                d for d in self.inter_data if d["behavior"] == behavior
            ]
        copied_dataset = copy.copy(self)
        copied_dataset.inter_data = filtered_data
        copied_dataset.target_behavior = behavior
        return copied_dataset

    def __len__(self) -> int:
        return len(self.inter_data)

    def __getitem__(self, index: int) -> dict[str, str | list[str] | list[int]]:
        d = self.inter_data[index]
        return dict(
            input_ids=d["inters"],
            labels=d["item"],
            behavior=d["behavior"],
            session_ids=d["session_ids"],
            extended_session_ids=d["extended_session_ids"],
            actions=d["actions"],
            time=d["time"],
            inters_item_list=d.get("inters_item_list", []),
            split=self.mode
        )


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


class SMBExplicitDatasetForDecoder(SMBExplicitDataset):
    def __init__(self, augment: int | None = None, **kwargs):
        self.augment = augment  # Times of augmentation for each interaction (for training only)
        if augment is not None and augment < 1:
            raise ValueError("augment must be greater than or equal to 1")
        super().__init__(**kwargs)

    @property
    def cached_file_name(self) -> str:
        if self.behavior_first:
            return os.path.join(self.data_path, self.dataset + f".{self.__class__.__name__}.{self.max_his_len}.SMB.aug{self.augment if self.augment else ''}.{self.mode}{self.index_suffix}.pkl")
        else:
            return os.path.join(self.data_path, self.dataset + f".{self.__class__.__name__}.{self.max_his_len}.SMB.behind.aug{self.augment if self.augment else ''}.{self.mode}{self.index_suffix}.pkl")

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

    def _process_train_data(self) -> list[dict[str, str | list[int] | list[float]]]:
        set_seed(42)  # For reproducibility
        inter_data = []
        if self.augment:
            logger.info(f"Augmenting interactions {self.augment} times for each user.")
        for uid in get_tqdm(self.remapped_inters, desc="Processing training data"):
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
                    "session_ids": self._generate_session_ids(sids),
                    "extended_session_ids": self._generate_extended_session_ids(sids),
                    "actions": self._generate_actions(behaviors),
                    "time": self._generate_times(times),
                    "behavior": behaviors[-1],
                })

        return inter_data


class SMBAugmentDataset(SMBExplicitDataset):
    def __init__(self, augment: int, **kwargs):
        self.augment = augment  # Times of augmentation for each interaction (for training only)
        if augment < 1:
            raise ValueError("augment must be greater than or equal to 1")
        super().__init__(**kwargs)

    @property
    def cached_file_name(self) -> str:
        if self.behavior_first:
            return os.path.join(self.data_path, self.dataset + f".{self.__class__.__name__}.{self.max_his_len}.SMB.aug{self.augment}.{self.mode}{self.index_suffix}.pkl")
        else:
            return os.path.join(self.data_path, self.dataset + f".{self.__class__.__name__}.{self.max_his_len}.SMB.behind.aug{self.augment}.{self.mode}{self.index_suffix}.pkl")

    def _augment_interactions(self, items: list[str], behaviors: list[str], sids: list[int], times: list[float]) -> tuple[list[list[str]], list[list[str]], list[list[int]], list[list[float]]]:
        if not self.augment:
            return [items], [behaviors], [sids], [times]
        downsample_ratios = np.arange(1, self.augment + 1) / (self.augment + 1)
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

    def _process_train_data(self) -> list[dict[str, str | list[int] | list[float]]]:
        set_seed(42)  # For reproducibility
        inter_data = []
        if self.augment:
            logger.info(f"Augmenting interactions {self.augment} times for each user.")
        for uid in get_tqdm(self.remapped_inters, desc="Augmenting training data"):
            if self.valid_pos[uid] <= 0:
                continue
            items = self.remapped_inters[uid][:self.valid_pos[uid]]
            behaviors = self.history_behaviors[uid][:self.valid_pos[uid]]
            sids = self.session[uid][:self.valid_pos[uid]]
            times = self.time[uid][:self.valid_pos[uid]]
            items_list, behaviors_list, sids_list, times_list = self._augment_interactions(items, behaviors, sids, times)
            for items, behaviors, sids, times in zip(items_list, behaviors_list, sids_list, times_list):
                session_ids_map = {}
                extended_session_ids_map = {}
                times_map = {}
                poss = [0]
                for i in range(1, len(items)):
                    if sids[i] > sids[i - 1]:
                        poss.append(i)
                    else:
                        poss.append(poss[-1])
                for i in range(1, len(items)):
                    sid = sids[i]
                    pos = poss[i]
                    # wrong, mark
                    if sid not in session_ids_map:
                        session_ids_map[sid] = self._generate_session_ids(sids[:pos + 1])
                        extended_session_ids_map[sid] = self._generate_extended_session_ids(sids[:pos + 1])
                        times_map[sid] = self._generate_times(times[:pos + 1])
                    inter_data.append({
                        "item": self.get_behavior_item(items[i], behaviors[i]),
                        "inters": self._get_inters(items[:pos], behaviors[:pos]),
                        "session_ids": session_ids_map[sid],
                        "extended_session_ids": extended_session_ids_map[sid],
                        "actions": self._generate_actions(behaviors[:pos] + [behaviors[i]]),
                        "time": times_map[sid],
                        "behavior": behaviors[i],
                    })

        return inter_data


class SMBAugmentEvaluationDataset(SMBExplicitDataset):
    def __init__(self, drop_ratio: float, **kwargs):
        self.drop_ratio = drop_ratio
        super().__init__(**kwargs)
        assert 0 <= drop_ratio <= 1, "drop_ratio must be in [0, 1]"

    @property
    def cached_file_name(self) -> str:
        if self.behavior_first:
            return os.path.join(self.data_path, self.dataset + f".{self.__class__.__name__}.{self.max_his_len}.SMB.drop{self.drop_ratio}.{self.mode}{self.index_suffix}.pkl")
        else:
            return os.path.join(self.data_path, self.dataset + f".{self.__class__.__name__}.{self.max_his_len}.SMB.behind.drop{self.drop_ratio}.{self.mode}{self.index_suffix}.pkl")

    def _drop_interactions(self, items: list[str], behaviors: list[str], sids: list[int], times: list[float]) -> tuple[list[str], list[str], list[int], list[float]]:
        behavior_indices = {}
        for behavior in self.behavior_level:
            behavior_indices[behavior] = [i for i, b in enumerate(behaviors) if b == behavior]
        drop_indices = []
        for behavior, level in self.behavior_level.items():
            if level == self.max_behavior_level:
                continue  # Skip the target behavior
            if behavior not in behavior_indices or len(behavior_indices[behavior]) == 0:
                continue
            behavior_ratio = self.drop_ratio / (level + 1)  # downsample ratio for each behavior
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
        if len(items_copy) < 1:
            return items, behaviors, sids, times
        return items_copy, behaviors_copy, sids_copy, times_copy

    def _process_valid_data(self) -> list[dict[str, str | list[int] | list[float]]]:
        inter_data = []
        for uid in get_tqdm(self.remapped_inters, desc="Processing validation data"):
            if self.valid_pos[uid] < 0:
                continue
            items = self.remapped_inters[uid][: self.test_pos[uid]]
            behaviors = self.history_behaviors[uid][: self.test_pos[uid]]
            sids = self.session[uid][: self.test_pos[uid]]
            times = self.time[uid][: self.test_pos[uid]]
            pos = self.valid_pos[uid]
            session_ids = self._generate_session_ids(sids[: pos + 1])
            extended_session_ids = self._generate_extended_session_ids(sids[: pos + 1])
            times = self._generate_times(times[: pos + 1])
            items_dropped, behaviors_dropped, sids_dropped, times_dropped = self._drop_interactions(
                items[:pos],
                behaviors[:pos],
                sids[:pos],
                times[:pos]
            )
            session_ids_dropped = self._generate_session_ids(sids_dropped + [sids[pos]])
            extended_session_ids_dropped = self._generate_extended_session_ids(sids_dropped + [sids[pos]])
            times_dropped = self._generate_times(times_dropped + [times[pos]])
            for i in range(pos, len(items)):
                if behaviors[i] != self.target_behavior:
                    inter_data.append({
                        "item": self.get_behavior_item(items[i], behaviors[i]),
                        "inters": self._get_inters(items[:pos], behaviors[:pos]),
                        "session_ids": session_ids,
                        "extended_session_ids": extended_session_ids,
                        "actions": self._generate_actions(behaviors + [behaviors[i]]),
                        "time": times,
                        "behavior": behaviors[i],
                    })
                else:
                    inter_data.append({
                        "item": self.get_behavior_item(items[i], behaviors[i]),
                        "inters": self._get_inters(items_dropped, behaviors_dropped),
                        "session_ids": session_ids_dropped,
                        "extended_session_ids": extended_session_ids_dropped,
                        "actions": self._generate_actions(behaviors_dropped + [behaviors[i]]),
                        "time": times_dropped,
                        "behavior": behaviors[i],
                    })

        return inter_data

    def _process_valid_test_data(self) -> list[dict[str, str | list[str] | list[int] | list[float]]]:
        inter_data = []
        for uid in get_tqdm(self.remapped_inters, desc="Processing validation data for testing"):
            items = self.remapped_inters[uid][: self.test_pos[uid]]
            behaviors = self.history_behaviors[uid][: self.test_pos[uid]]
            sids = self.session[uid][: self.test_pos[uid]]
            times = self.time[uid][: self.test_pos[uid]]
            session_items: list[str] = []
            session_behaviors: list[str] = []
            for i in range(self.valid_pos[uid], len(items)):
                session_items.append(self.get_behavior_item(items[i], behaviors[i]))
                session_behaviors.append(behaviors[i])
            assert len(session_items) > 0, f"Session for user {uid} is empty after valid position {self.valid_pos[uid]}."
            items_dropped, behaviors_dropped, sids_dropped, times_dropped = self._drop_interactions(
                items[:self.valid_pos[uid]],
                behaviors[:self.valid_pos[uid]],
                sids[:self.valid_pos[uid]],
                times[:self.valid_pos[uid]]
            )
            inter_data.append({
                "item": session_items,
                # Original history without dropping
                "inters": self._get_inters(items[:self.valid_pos[uid]], behaviors[:self.valid_pos[uid]]),
                "inters_item_list": self._get_inters_with_only_items(items[:self.valid_pos[uid]]),
                # ! For test set, we donot add session IDs for the item to be predicted, and the session IDs should be add by the inference code.
                "session_ids": self._generate_session_ids(self.session[uid][:self.valid_pos[uid]]),
                "extended_session_ids": self._generate_extended_session_ids(self.session[uid][:self.valid_pos[uid]]),
                "actions": self._generate_actions(self.history_behaviors[uid][: self.test_pos[uid]]),
                "time": self._generate_times(times[:self.valid_pos[uid] + 1]),
                "behavior": session_behaviors,
                # Dropped history
                "inters_dropped": self._get_inters(items_dropped, behaviors_dropped),
                "inters_item_list_dropped": self._get_inters_with_only_items(items_dropped),
                # ! For test set, we donot add session IDs for the item to be predicted, and the session IDs should be add by the inference code.
                "session_ids_dropped": self._generate_session_ids(sids_dropped),
                "extended_session_ids_dropped": self._generate_extended_session_ids(sids_dropped),
                "actions_dropped": self._generate_actions(behaviors_dropped),
                "time_dropped": self._generate_times(times_dropped + [times[self.valid_pos[uid]]]),
            })

        return inter_data

    def _process_test_data(self) -> list[dict[str, str | list[str] | list[int] | list[float]]]:
        inter_data = []
        for uid in get_tqdm(self.remapped_inters, desc="Processing test data"):
            items = self.remapped_inters[uid]
            behaviors = self.history_behaviors[uid]
            sids = self.session[uid]
            times = self.time[uid]
            session_items: list[str] = []
            session_behaviors: list[str] = []
            for i in range(self.test_pos[uid], len(items)):
                session_items.append(self.get_behavior_item(items[i], behaviors[i]))
                session_behaviors.append(behaviors[i])
            assert len(session_items) > 0, f"Session for user {uid} is empty after test position {self.test_pos[uid]}."
            items_dropped, behaviors_dropped, sids_dropped, times_dropped = self._drop_interactions(
                items[:self.test_pos[uid]],
                behaviors[:self.test_pos[uid]],
                sids[:self.test_pos[uid]],
                times[:self.test_pos[uid]]
            )
            inter_data.append({
                "item": session_items,
                # Original history without dropping
                "inters": self._get_inters(items[:self.test_pos[uid]], behaviors[:self.test_pos[uid]]),
                "inters_item_list": self._get_inters_with_only_items(items[:self.test_pos[uid]]),
                # ! For test set, we donot add session IDs for the item to be predicted, and the session IDs should be add by the inference code.
                "session_ids": self._generate_session_ids(self.session[uid][:self.test_pos[uid]]),
                "extended_session_ids": self._generate_extended_session_ids(self.session[uid][:self.test_pos[uid]]),
                "actions": self._generate_actions(self.history_behaviors[uid][:self.test_pos[uid]]),
                "time": self._generate_times(times[:self.test_pos[uid] + 1]),
                # Dropped history
                "inters_dropped": self._get_inters(items_dropped, behaviors_dropped),
                "inters_item_list_dropped": self._get_inters_with_only_items(items_dropped),
                # ! For test set, we donot add session IDs for the item to be predicted, and the session IDs should be add by the inference code.
                "session_ids_dropped": self._generate_session_ids(sids_dropped),
                "extended_session_ids_dropped": self._generate_extended_session_ids(sids_dropped),
                "actions_dropped": self._generate_actions(behaviors_dropped),
                "time_dropped": self._generate_times(times_dropped + [times[self.test_pos[uid]]]),
                "behavior": session_behaviors,
            })

        return inter_data

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
                if behavior == self.target_behavior:
                    filtered_data.append({
                        "item": items,
                        "inters": d["inters_dropped"],
                        "inters_item_list": d["inters_item_list_dropped"],
                        "session_ids": d["session_ids_dropped"],
                        "extended_session_ids": d["extended_session_ids_dropped"],
                        "actions": d["actions_dropped"],
                        "behavior": behaviors,
                        "time": d["time_dropped"],
                    })
                else:
                    filtered_data.append({
                        "item": items,
                        "inters": d["inters"],
                        "inters_item_list": d["inters_item_list"],
                        "session_ids": d["session_ids"],
                        "extended_session_ids": d["extended_session_ids"],
                        "actions": d["actions"],
                        "behavior": behaviors,
                        "time": d["time"],
                    })
        else:
            filtered_data = [
                d for d in self.inter_data if d["behavior"] == behavior
            ]
        copied_dataset = copy.copy(self)
        copied_dataset.inter_data = filtered_data
        copied_dataset.target_behavior = behavior
        return copied_dataset


class SMBDropGTEvaluationDataset(SMBExplicitDataset):
    def _GT_index(self, items: list[str], gt_items: list[str], behaviors: list[str]) -> list[bool]:
        gt_set = set(gt_items)
        return [item in gt_set and behavior != self.target_behavior for item, behavior in zip(items, behaviors)]

    def _process_test_data(self) -> list[dict[str, str | list[str] | list[int] | list[float]]]:
        inter_data = []
        drop_ratios = []
        for uid in get_tqdm(self.remapped_inters, desc="Processing test data"):
            items = self.remapped_inters[uid]
            behaviors = self.history_behaviors[uid]
            sids = self.session[uid]
            times = self.time[uid]
            session_items: list[str] = []
            session_behaviors: list[str] = []
            for i in range(self.test_pos[uid], len(items)):
                session_items.append(self.get_behavior_item(items[i], behaviors[i]))
                session_behaviors.append(behaviors[i])
            assert len(session_items) > 0, f"Session for user {uid} is empty after test position {self.test_pos[uid]}."
            GT_index = self._GT_index(items[:self.test_pos[uid]], items[self.test_pos[uid]:], behaviors[:self.test_pos[uid]])
            if len(GT_index) > 0:
                drop_ratios.append(sum(GT_index) / len(GT_index))
            if sum(GT_index) == len(GT_index):
                continue
            items_dropped = [item for item, is_gt in zip(items[:self.test_pos[uid]], GT_index) if not is_gt]
            behaviors_dropped = [behavior for behavior, is_gt in zip(behaviors[:self.test_pos[uid]], GT_index) if not is_gt]
            sids_dropped = [sid for sid, is_gt in zip(sids[:self.test_pos[uid]], GT_index) if not is_gt]
            times_dropped = [time for time, is_gt in zip(times[:self.test_pos[uid]], GT_index) if not is_gt]
            inter_data.append({
                "item": session_items,
                "inters": self._get_inters(items_dropped, behaviors_dropped),
                "inters_item_list": self._get_inters_with_only_items(items_dropped),
                # ! For test set, we donot add session IDs for the item to be predicted, and the session IDs should be add by the inference code.
                "session_ids": self._generate_session_ids(sids_dropped),
                "extended_session_ids": self._generate_extended_session_ids(sids_dropped),
                "actions": self._generate_actions(behaviors_dropped),
                "time": self._generate_times(times_dropped + [times[self.test_pos[uid]]]),
                "behavior": session_behaviors,
            })
        logger.warning(f"Average drop ratio of ground-truth items: {np.mean(drop_ratios) if len(drop_ratios) > 0 else 0:.4f}")

        return inter_data
