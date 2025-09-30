import os
import json
import copy
import random
import pickle
import numpy as np
import pandas as pd
from loguru import logger
from torch.utils.data import Dataset

from SeqRec.utils.pipe import set_seed, get_tqdm


class BaseSSeqDataset(Dataset):
    """
    Base class for session-wise multi-behavior sequential recommendation datasets.
    """

    def __init__(self, dataset: str, data_path: str, max_his_len: int, mode: str, add_uid: bool = False, **kwargs):
        super().__init__()

        self.dataset: str = dataset
        self.data_path = os.path.join(data_path, self.dataset)

        self.max_his_len: int = max_his_len
        self.mode = mode
        self.add_uid = add_uid
        logger.info(f"Initializing {self.__class__.__name__} for {self.mode} set of {self.dataset} dataset with max_his_len={self.max_his_len}, add_uid={self.add_uid}")

        # load data
        self._load_data()
        self.num = max(
            item for items in self.inters.values() for item in items
        ) + 1

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
            else:
                raise NotImplementedError
            if int(os.environ.get("LOCAL_RANK", 0)) == 0:
                with open(self.cached_file_name, "wb") as f:
                    pickle.dump(self.inter_data, f)

        logger.info(f"Loaded {len(self.inter_data)} interactions for {self.mode} set.")

    @property
    def cached_file_name(self) -> str:
        if not self.add_uid:
            return os.path.join(self.data_path, self.dataset + f".{self.__class__.__name__}.{self.max_his_len}.SMB.{self.mode}.pkl")
        else:
            return os.path.join(self.data_path, self.dataset + f".{self.__class__.__name__}.{self.max_his_len}.SMB.adduid.{self.mode}.pkl")

    def _load_data(self):
        with open(os.path.join(self.data_path, self.dataset + ".SMB.inter.json"), "r") as f:
            self.inters: dict[str, list[int]] = json.load(f)
        self.num_users = max(int(uid) for uid in self.inters.keys()) + 1
        with open(os.path.join(self.data_path, self.dataset + ".SMB.behavior.json"), "r") as f:
            self.history_behaviors: dict[str, list[str]] = json.load(f)

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
        self.target_behavior_index = self.behaviors.index(self.target_behavior)

    def get_behavior_item(self, item: int, behavior: str) -> int:
        raise NotImplementedError(
            "This method should be implemented in subclasses to return the behavior-item representation."
        )

    def _get_inters(self, history_items: list[int], history_behaviors: list[str], max_his_len: int | None = None) -> list[int]:
        if max_his_len is None:
            max_his_len = self.max_his_len
        if max_his_len > 0:
            history_items = history_items[-max_his_len:]
            history_behaviors = history_behaviors[-max_his_len:]
        history_behavior_items = [
            self.get_behavior_item(history_item, history_behavior)
            for history_item, history_behavior in zip(history_items, history_behaviors)
        ]
        return history_behavior_items

    def _get_inter_behaviors(self, history_behaviors: list[str], max_his_len: int | None = None) -> list[int]:
        if max_his_len is None:
            max_his_len = self.max_his_len
        if max_his_len > 0:
            history_behaviors = history_behaviors[-max_his_len:]
        history_behavior_ids = [self.behaviors.index(b) for b in history_behaviors]
        return history_behavior_ids

    def _generate_session_ids(self, session_ids: list[int], max_his_len: int | None = None) -> list[int]:
        ret = []
        if max_his_len is None:
            max_his_len = self.max_his_len
        if max_his_len > 0:
            if self.mode in ["train", "valid"]:
                max_his_len += 1
            session_ids = session_ids[-max_his_len:]
        return session_ids

    def _generate_actions(self, actions: list[int], max_his_len: int | None = None) -> list[int]:
        ret = []
        if max_his_len is None:
            max_his_len = self.max_his_len
        if max_his_len > 0:
            if self.mode in ["train", "valid"]:
                max_his_len += 1
            actions = actions[-max_his_len:]
        for action in actions:
            ret.append(self.behavior_level[action])
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
            ret.append(time)
        return ret

    def _process_train_data(self) -> list[dict[str, int | list[int] | list[float]]]:
        inter_data = []
        for uid in get_tqdm(self.inters, desc="Processing training data"):
            if self.valid_pos[uid] <= 0:
                continue
            items = self.inters[uid][: self.valid_pos[uid]]
            behaviors = self.history_behaviors[uid][: self.valid_pos[uid]]
            times = self.time[uid][: self.valid_pos[uid]]
            session_ids_map = {}
            times_map = {}
            for i in range(1, len(items)):
                sid = self.session[uid][i]
                pos = self.train_pos[uid][sid]
                if len(self._get_inters(items[:pos], behaviors[:pos])) == 0:
                    continue
                if sid not in session_ids_map:
                    session_ids_map[sid] = self._generate_session_ids(self.session[uid][:pos + 1])
                    times_map[sid] = self._generate_times(times[:pos + 1])
                sample = {
                    "item": self.get_behavior_item(items[i], behaviors[i]),
                    "inters": self._get_inters(items[:pos], behaviors[:pos]),
                    "inter_behaviors": self._get_inter_behaviors(behaviors[:pos]),
                    "session_ids": session_ids_map[sid],
                    "actions": self._generate_actions(behaviors[:pos] + [behaviors[i]]),
                    "time": times_map[sid],
                    "behavior": self.behaviors.index(behaviors[i]),
                }
                if self.add_uid:
                    sample['uid'] = int(uid) + 1
                inter_data.append(sample)

        return inter_data

    def _process_valid_data(self) -> list[dict[str, int | list[int] | list[float]]]:
        inter_data = []
        for uid in get_tqdm(self.inters, desc="Processing validation data for testing"):
            items = self.inters[uid][: self.test_pos[uid]]
            behaviors = self.history_behaviors[uid][: self.test_pos[uid]]
            times = self.time[uid][: self.test_pos[uid]]
            session_items: list[int] = []
            session_behaviors: list[int] = []
            for i in range(self.valid_pos[uid], len(items)):
                session_items.append(self.get_behavior_item(items[i], behaviors[i]))
                session_behaviors.append(self.behaviors.index(behaviors[i]))
            assert len(session_items) > 0, f"Session for user {uid} is empty after valid position {self.valid_pos[uid]}."
            sample = {
                "item": session_items,
                "inters": self._get_inters(items[:self.valid_pos[uid]], behaviors[:self.valid_pos[uid]]),
                "inter_behaviors": self._get_inter_behaviors(behaviors[:self.valid_pos[uid]]),
                # ! For validation set, we donot add session IDs for the item to be predicted, and the session IDs should be add by the inference code.
                "session_ids": self._generate_session_ids(self.session[uid][:self.valid_pos[uid]]),
                "actions": self._generate_actions(self.history_behaviors[uid][:self.valid_pos[uid]]),
                "time": self._generate_times(times[:self.valid_pos[uid] + 1]),
                "behavior": session_behaviors,
            }
            if self.add_uid:
                sample['uid'] = int(uid) + 1
            inter_data.append(sample)

        return inter_data

    def _process_test_data(self) -> list[dict[str, int | list[int] | list[float]]]:
        inter_data = []
        for uid in get_tqdm(self.inters, desc="Processing test data"):
            items = self.inters[uid]
            behaviors = self.history_behaviors[uid]
            times = self.time[uid]
            session_items: list[int] = []
            session_behaviors: list[int] = []
            for i in range(self.test_pos[uid], len(items)):
                session_items.append(self.get_behavior_item(items[i], behaviors[i]))
                session_behaviors.append(self.behaviors.index(behaviors[i]))
            assert len(session_items) > 0, f"Session for user {uid} is empty after test position {self.test_pos[uid]}."
            sample = {
                "item": session_items,
                "inters": self._get_inters(items[:self.test_pos[uid]], behaviors[:self.test_pos[uid]]),
                "inter_behaviors": self._get_inter_behaviors(behaviors[:self.test_pos[uid]]),
                # ! For test set, we donot add session IDs for the item to be predicted, and the session IDs should be add by the inference code.
                "session_ids": self._generate_session_ids(self.session[uid][:self.test_pos[uid]]),
                "actions": self._generate_actions(self.history_behaviors[uid][:self.test_pos[uid]]),
                "time": self._generate_times(times[:self.test_pos[uid] + 1]),
                "behavior": session_behaviors,
            }
            if self.add_uid:
                sample['uid'] = int(uid) + 1
            inter_data.append(sample)

        return inter_data

    def filter_by_behavior(self, behavior: str) -> "BaseSSeqDataset":
        if isinstance(self.inter_data[0]['behavior'], list):
            filtered_data = []
            inter_data = get_tqdm(self.inter_data, desc=f"Filtering by behavior - {behavior}")
            for d in inter_data:
                if self.behaviors.index(behavior) not in d["behavior"]:
                    continue
                items, behaviors = [], []
                for sample_item, sample_behavior in zip(d["item"], d["behavior"]):
                    if sample_behavior == self.behaviors.index(behavior):
                        items.append(sample_item)
                        behaviors.append(sample_behavior)
                items = list(set(items))
                inter = {
                    "item": items,
                    "behavior": self.behaviors.index(behavior),
                }
                keys = set(d.keys()) - set(inter.keys())
                inter.update({k: d[k] for k in keys})
                filtered_data.append(inter)
        else:
            filtered_data = [
                d for d in self.inter_data if d["behavior"] == self.behaviors.index(behavior)
            ]
        copied_dataset = copy.copy(self)
        copied_dataset.inter_data = filtered_data
        copied_dataset.target_behavior = behavior
        return copied_dataset

    def __len__(self) -> int:
        return len(self.inter_data)

    def __getitem__(self, index: int) -> dict[str, int | list[int] | list[float]]:
        d = self.inter_data[index]
        ret = dict(
            inters=d["inters"],
            seq_len=len(d["inters"]),
            inter_behaviors=d["inter_behaviors"],
            target=d["item"],
            behavior=d["behavior"],
            session_ids=d["session_ids"],
            actions=d["actions"],
            time=d["time"],
            split=self.mode
        )
        if 'neg_item' in d:  # for negative sampling dataset
            ret['neg_item'] = d['neg_item']
        if 'item_range' in d:
            ret['item_range'] = d['item_range']
        if 'uid' in d:
            ret['uid'] = d['uid']
        return ret


class SSeqDataset(BaseSSeqDataset):
    """
    Session-wise multi-behavior dataset with treating the item with different behaviors as the same item (no diff) or different items (diff).
    """

    def __init__(self, diff: bool = False, **kwargs):
        self.diff = diff
        super().__init__(**kwargs)

    @property
    def num_items(self) -> int:
        if not self.diff:
            return self.num
        else:
            return len(self.behaviors) * self.num

    @property
    def cached_file_name(self) -> str:
        if not self.add_uid:
            if not self.diff:
                return os.path.join(self.data_path, self.dataset + f".{self.__class__.__name__}.{self.max_his_len}.SMB.{self.mode}.pkl")
            else:
                return os.path.join(self.data_path, self.dataset + f".{self.__class__.__name__}.{self.max_his_len}.SMB.diff.{self.mode}.pkl")
        else:
            if not self.diff:
                return os.path.join(self.data_path, self.dataset + f".{self.__class__.__name__}.{self.max_his_len}.SMB.adduid.{self.mode}.pkl")
            else:
                return os.path.join(self.data_path, self.dataset + f".{self.__class__.__name__}.{self.max_his_len}.SMB.diff.adduid.{self.mode}.pkl")

    def get_behavior_item(self, item: int, behavior: str) -> int:
        if self.diff:
            return self.behaviors.index(behavior) * self.num + item + 1  # + 1 for padding index 0
        else:
            return item + 1  # + 1 for padding index 0

    def filter_by_behavior(self, behavior: str) -> "SSeqDataset":
        if self.diff and self.mode == 'test':
            assert isinstance(self.inter_data[0]['behavior'], list)
            ret_dataset = super().filter_by_behavior(behavior)
            item_range = (self.behaviors.index(behavior) * self.num + 1, (self.behaviors.index(behavior) + 1) * self.num + 1)
            for i in range(len(ret_dataset.inter_data)):
                ret_dataset.inter_data[i]['item_range'] = item_range
            return ret_dataset
        else:
            return super().filter_by_behavior(behavior)


class SSeqTargetDataset(SSeqDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _process_train_data(self) -> list[dict[str, int | list[int] | list[float]]]:
        inter_data = []
        for uid in get_tqdm(self.inters, desc="Processing training data"):
            if self.valid_pos[uid] <= 0:
                continue
            items = self.inters[uid][: self.valid_pos[uid]]
            behaviors = self.history_behaviors[uid][: self.valid_pos[uid]]
            times = self.time[uid][: self.valid_pos[uid]]
            session_ids_map = {}
            times_map = {}
            for i in range(1, len(items)):
                sid = self.session[uid][i]
                pos = self.train_pos[uid][sid]
                if len(self._get_inters(items[:pos], behaviors[:pos])) == 0:
                    continue
                if sid not in session_ids_map:
                    session_ids_map[sid] = self._generate_session_ids(self.session[uid][:pos + 1])
                    times_map[sid] = self._generate_times(times[:pos + 1])
                sample = {
                    "item": self.get_behavior_item(items[i], behaviors[i]),
                    "inters": self._get_inters(items[:pos] + [items[i]], behaviors[:pos] + [behaviors[i]]),
                    "inter_behaviors": self._get_inter_behaviors(behaviors[:pos] + [behaviors[i]]),
                    "session_ids": session_ids_map[sid],
                    "actions": self._generate_actions(behaviors[:pos] + [behaviors[i]]),
                    "time": times_map[sid],
                    "behavior": self.behaviors.index(behaviors[i]),
                }
                if self.add_uid:
                    sample['uid'] = int(uid) + 1
                inter_data.append(sample)

        return inter_data

    def _process_valid_data(self) -> list[dict[str, int | list[int] | list[float]]]:
        inter_data = []
        for uid in get_tqdm(self.inters, desc="Processing validation data for testing"):
            items = self.inters[uid][: self.test_pos[uid]]
            behaviors = self.history_behaviors[uid][: self.test_pos[uid]]
            times = self.time[uid][: self.test_pos[uid]]
            session_items: list[int] = []
            session_behaviors: list[int] = []
            for i in range(self.valid_pos[uid], len(items)):
                session_items.append(self.get_behavior_item(items[i], behaviors[i]))
                session_behaviors.append(self.behaviors.index(behaviors[i]))
            assert len(session_items) > 0, f"Session for user {uid} is empty after valid position {self.valid_pos[uid]}."
            sample = {
                "item": session_items,
                "inters": self._get_inters(items[:self.valid_pos[uid]], behaviors[:self.valid_pos[uid]], max_his_len=self.max_his_len - 1) + [self.num_items + 1],  # add mask token
                "inter_behaviors": self._get_inter_behaviors(behaviors[:self.valid_pos[uid]], max_his_len=self.max_his_len - 1) + [-1],  # decided in filter_by_behavior
                # ! For validation set, we donot add session IDs for the item to be predicted, and the session IDs should be add by the inference code.
                "session_ids": self._generate_session_ids(self.session[uid][:self.valid_pos[uid]]),
                "actions": self._generate_actions(self.history_behaviors[uid][:self.valid_pos[uid]]),
                "time": self._generate_times(times[:self.valid_pos[uid] + 1]),
                "behavior": session_behaviors,
            }
            if self.add_uid:
                sample['uid'] = int(uid) + 1
            inter_data.append(sample)

        return inter_data

    def _process_test_data(self) -> list[dict[str, int | list[int] | list[float]]]:
        inter_data = []
        for uid in get_tqdm(self.inters, desc="Processing test data"):
            items = self.inters[uid]
            behaviors = self.history_behaviors[uid]
            times = self.time[uid]
            session_items: list[int] = []
            session_behaviors: list[int] = []
            for i in range(self.test_pos[uid], len(items)):
                session_items.append(self.get_behavior_item(items[i], behaviors[i]))
                session_behaviors.append(self.behaviors.index(behaviors[i]))
            assert len(session_items) > 0, f"Session for user {uid} is empty after test position {self.test_pos[uid]}."
            sample = {
                "item": session_items,
                "inters": self._get_inters(items[:self.test_pos[uid]], behaviors[:self.test_pos[uid]], max_his_len=self.max_his_len - 1) + [self.num_items + 1],  # add mask token
                "inter_behaviors": self._get_inter_behaviors(behaviors[:self.test_pos[uid]], max_his_len=self.max_his_len - 1) + [-1],  # decided in filter_by_behavior
                # ! For test set, we donot add session IDs for the item to be predicted, and the session IDs should be add by the inference code.
                "session_ids": self._generate_session_ids(self.session[uid][:self.test_pos[uid]]),
                "actions": self._generate_actions(self.history_behaviors[uid][:self.test_pos[uid]]),
                "time": self._generate_times(times[:self.test_pos[uid] + 1]),
                "behavior": session_behaviors,
            }
            if self.add_uid:
                sample['uid'] = int(uid) + 1
            inter_data.append(sample)

        return inter_data

    def filter_by_behavior(self, behavior: str) -> "SSeqTargetDataset":
        ret_dataset = super().filter_by_behavior(behavior)
        for i in range(len(ret_dataset.inter_data)):
            # copy the inter_behaviors to avoid modifying the original data
            ret_dataset.inter_data[i]['inter_behaviors'] = ret_dataset.inter_data[i]['inter_behaviors'].copy()
            ret_dataset.inter_data[i]['inter_behaviors'][-1] = self.behaviors.index(behavior)  # set the behavior of the target item
        return ret_dataset


class SSeqNegSampleDataset(SSeqDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _sample_negative_items(self, num_samples: int, exclude_items: set[int]) -> list[int]:
        all_items = set(range(self.num))
        negative_items = list(all_items - exclude_items)
        return random.sample(negative_items, num_samples)

    def _process_train_data(self) -> list[dict[str, int | list[int] | list[float]]]:
        set_seed(42)  # For reproducibility of negative sampling
        inter_data = []
        for uid in get_tqdm(self.inters, desc="Processing training data"):
            if self.valid_pos[uid] <= 0:
                continue
            items = self.inters[uid][: self.valid_pos[uid]]
            neg_items = self._sample_negative_items(
                num_samples=len(items),
                exclude_items=set(items)
            )
            behaviors = self.history_behaviors[uid][: self.valid_pos[uid]]
            times = self.time[uid][: self.valid_pos[uid]]
            session_ids_map = {}
            times_map = {}
            for i in range(1, len(items)):
                sid = self.session[uid][i]
                pos = self.train_pos[uid][sid]
                if len(self._get_inters(items[:pos], behaviors[:pos])) == 0:
                    continue
                if sid not in session_ids_map:
                    session_ids_map[sid] = self._generate_session_ids(self.session[uid][:pos + 1])
                    times_map[sid] = self._generate_times(times[:pos + 1])
                sample = {
                    "item": self.get_behavior_item(items[i], behaviors[i]),
                    "neg_item": self.get_behavior_item(neg_items[i], behaviors[i]),
                    "inters": self._get_inters(items[:pos], behaviors[:pos]),
                    "inter_behaviors": self._get_inter_behaviors(behaviors[:pos]),
                    "session_ids": session_ids_map[sid],
                    "actions": self._generate_actions(behaviors[:pos] + [behaviors[i]]),
                    "time": times_map[sid],
                    "behavior": self.behaviors.index(behaviors[i]),
                }
                if self.add_uid:
                    sample['uid'] = int(uid) + 1
                inter_data.append(sample)

        return inter_data


class SSeqUserLevelDataset(SSeqDataset):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _process_train_data(self) -> list[dict[str, int | list[int] | list[float]]]:
        set_seed(42)  # For reproducibility of negative sampling
        inter_data = []
        for uid in get_tqdm(self.inters, desc="Processing training data"):
            if self.valid_pos[uid] <= 0:
                continue
            items = self.inters[uid][:self.valid_pos[uid]]
            behaviors = self.history_behaviors[uid][:self.valid_pos[uid]]
            sids = self.session[uid][:self.valid_pos[uid]]
            times = self.time[uid][:self.valid_pos[uid]]
            if len(items) > self.max_his_len and random.random() > 0.8:
                begin_idx = random.randint(0, len(items) - self.max_his_len - 1)
                items = items[begin_idx: begin_idx + self.max_his_len]
                behaviors = behaviors[begin_idx: begin_idx + self.max_his_len]
                sids = sids[begin_idx: begin_idx + self.max_his_len]
                times = times[begin_idx: begin_idx + self.max_his_len]
            sample = {
                "item": self.get_behavior_item(items[-1], behaviors[-1]),
                "inters": self._get_inters(items, behaviors),  # use all history interactions
                "inter_behaviors": self._get_inter_behaviors(behaviors),  # use all history interactions
                "session_ids": self._generate_session_ids(sids),
                "actions": self._generate_actions(behaviors),
                "time": self._generate_times(times),
                "behavior": self.behaviors.index(behaviors[-1]),
            }
            if self.add_uid:
                sample['uid'] = int(uid) + 1
            inter_data.append(sample)

        return inter_data


class SSeqNegSampleEvalDataset(SSeqDataset):
    def __init__(self, num_neg: int = 1000, **kwargs):
        self.num_neg = num_neg
        super().__init__(**kwargs)

    @property
    def cached_file_name(self) -> str:
        if not self.add_uid:
            if not self.diff:
                return os.path.join(self.data_path, self.dataset + f".{self.__class__.__name__}.{self.max_his_len}.neg{self.num_neg}.SMB.{self.mode}.pkl")
            else:
                return os.path.join(self.data_path, self.dataset + f".{self.__class__.__name__}.{self.max_his_len}.neg{self.num_neg}.SMB.diff.{self.mode}.pkl")
        else:
            if not self.diff:
                return os.path.join(self.data_path, self.dataset + f".{self.__class__.__name__}.{self.max_his_len}.neg{self.num_neg}.SMB.adduid.{self.mode}.pkl")
            else:
                return os.path.join(self.data_path, self.dataset + f".{self.__class__.__name__}.{self.max_his_len}.neg{self.num_neg}.SMB.diff.adduid.{self.mode}.pkl")

    def _sample_negative_items(self, num_samples: int, exclude_items: set[int]) -> list[int]:
        all_items = set(range(self.num))
        negative_items = list(all_items - exclude_items)
        return random.sample(negative_items, num_samples)

    def _process_valid_data(self) -> list[dict[str, int | list[int] | list[float]]]:
        set_seed(42)  # For reproducibility of negative sampling
        inter_data = []
        for uid in get_tqdm(self.inters, desc="Processing validation data for testing"):
            items = self.inters[uid][: self.test_pos[uid]]
            behaviors = self.history_behaviors[uid][: self.test_pos[uid]]
            times = self.time[uid][: self.test_pos[uid]]
            session_items: list[int] = []
            session_behaviors: list[int] = []
            for i in range(self.valid_pos[uid], len(items)):
                session_items.append(self.get_behavior_item(items[i], behaviors[i]))
                session_behaviors.append(self.behaviors.index(behaviors[i]))
            assert len(session_items) > 0, f"Session for user {uid} is empty after valid position {self.valid_pos[uid]}."
            negative_items = self._sample_negative_items(
                num_samples=self.num_neg,
                exclude_items=set(items)
            )
            negative_behaviors = [self.target_behavior] * self.num_neg
            negative_behavior_items = [
                self.get_behavior_item(neg_item, neg_behavior) for neg_item, neg_behavior in zip(negative_items, negative_behaviors)
            ]
            sample = {
                "item": session_items,
                "neg_item": negative_behavior_items,
                "inters": self._get_inters(items[:self.valid_pos[uid]], behaviors[:self.valid_pos[uid]]),
                "inter_behaviors": self._get_inter_behaviors(behaviors[:self.valid_pos[uid]]),
                # ! For validation set, we donot add session IDs for the item to be predicted, and the session IDs should be add by the inference code.
                "session_ids": self._generate_session_ids(self.session[uid][:self.valid_pos[uid]]),
                "actions": self._generate_actions(self.history_behaviors[uid][:self.valid_pos[uid]]),
                "time": self._generate_times(times[:self.valid_pos[uid] + 1]),
                "behavior": session_behaviors,
            }
            if self.add_uid:
                sample['uid'] = int(uid) + 1
            inter_data.append(sample)

        return inter_data


class SSeqTargetNegSampleEvalDataset(SSeqDataset):
    def __init__(self, num_neg: int = 1000, **kwargs):
        self.num_neg = num_neg
        super().__init__(**kwargs)

    @property
    def cached_file_name(self) -> str:
        if not self.add_uid:
            if not self.diff:
                return os.path.join(self.data_path, self.dataset + f".{self.__class__.__name__}.{self.max_his_len}.neg{self.num_neg}.SMB.{self.mode}.pkl")
            else:
                return os.path.join(self.data_path, self.dataset + f".{self.__class__.__name__}.{self.max_his_len}.neg{self.num_neg}.SMB.diff.{self.mode}.pkl")
        else:
            if not self.diff:
                return os.path.join(self.data_path, self.dataset + f".{self.__class__.__name__}.{self.max_his_len}.neg{self.num_neg}.SMB.adduid.{self.mode}.pkl")
            else:
                return os.path.join(self.data_path, self.dataset + f".{self.__class__.__name__}.{self.max_his_len}.neg{self.num_neg}.SMB.diff.adduid.{self.mode}.pkl")

    def _sample_negative_items(self, num_samples: int, exclude_items: set[int]) -> list[int]:
        all_items = set(range(self.num))
        negative_items = list(all_items - exclude_items)
        return random.sample(negative_items, num_samples)

    def _process_valid_data(self) -> list[dict[str, int | list[int] | list[float]]]:
        set_seed(42)  # For reproducibility of negative sampling
        inter_data = []
        for uid in get_tqdm(self.inters, desc="Processing validation data for testing"):
            items = self.inters[uid][: self.test_pos[uid]]
            behaviors = self.history_behaviors[uid][: self.test_pos[uid]]
            times = self.time[uid][: self.test_pos[uid]]
            session_items: list[int] = []
            session_behaviors: list[int] = []
            for i in range(self.valid_pos[uid], len(items)):
                session_items.append(self.get_behavior_item(items[i], behaviors[i]))
                session_behaviors.append(self.behaviors.index(behaviors[i]))
            assert len(session_items) > 0, f"Session for user {uid} is empty after valid position {self.valid_pos[uid]}."
            negative_items = self._sample_negative_items(
                num_samples=self.num_neg,
                exclude_items=set(items)
            )
            negative_behaviors = [self.target_behavior] * self.num_neg
            negative_behavior_items = [
                self.get_behavior_item(neg_item, neg_behavior) for neg_item, neg_behavior in zip(negative_items, negative_behaviors)
            ]
            sample = {
                "item": session_items,
                "neg_item": negative_behavior_items,
                "inters": self._get_inters(items[:self.valid_pos[uid]], behaviors[:self.valid_pos[uid]], max_his_len=self.max_his_len - 1) + [self.num_items + 1],  # add mask token
                "inter_behaviors": self._get_inter_behaviors(behaviors[:self.valid_pos[uid]], max_his_len=self.max_his_len - 1) + [-1],  # decided in filter_by_behavior
                # ! For validation set, we donot add session IDs for the item to be predicted, and the session IDs should be add by the inference code.
                "session_ids": self._generate_session_ids(self.session[uid][:self.valid_pos[uid]]),
                "actions": self._generate_actions(self.history_behaviors[uid][:self.valid_pos[uid]]),
                "time": self._generate_times(times[:self.valid_pos[uid] + 1]),
                "behavior": session_behaviors,
            }
            if self.add_uid:
                sample['uid'] = int(uid) + 1
            inter_data.append(sample)

        return inter_data

    def filter_by_behavior(self, behavior: str) -> "SSeqTargetNegSampleEvalDataset":
        ret_dataset = super().filter_by_behavior(behavior)
        for i in range(len(ret_dataset.inter_data)):
            ret_dataset.inter_data[i]['inter_behaviors'] = ret_dataset.inter_data[i]['inter_behaviors'].copy()
            ret_dataset.inter_data[i]['inter_behaviors'][-1] = self.behaviors.index(behavior)  # set the behavior of the target item
        return ret_dataset
