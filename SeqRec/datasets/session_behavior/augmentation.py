import os
import copy
import numpy as np
from loguru import logger

from SeqRec.datasets.session_behavior.base import BaseSMBDataset
from SeqRec.datasets.session_behavior.explicit import SMBExplicitDataset
from SeqRec.utils.runtime import set_seed, get_tqdm


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
                if self.train_session:
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
                else:
                    for i in range(1, len(items)):
                        inter_data.append({
                            "item": self.get_behavior_item(items[i], behaviors[i]),
                            "inters": self._get_inters(items[:i], behaviors[:i]),
                            "session_ids": self._generate_session_ids(sids[:i+1]),
                            "extended_session_ids": self._generate_extended_session_ids(sids[:i+1]),
                            "actions": self._generate_actions(behaviors[:i+1]),
                            "time": self._generate_times(times[:i+1]),
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
                "uid": uid,
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
                if 'uid' in d:
                    filtered_data[-1]['uid'] = d['uid']
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
                "uid": uid,
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
