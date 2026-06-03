import os
import zlib
import copy
import numpy as np
from loguru import logger

from SeqRec.datasets.session_behavior.explicit import SMBExplicitDataset
from SeqRec.utils.runtime import set_seed, get_tqdm


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
                if behavior == self.target_behavior:
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


class SMBFixedRatioDatasetForDecoder(SMBExplicitDataset):
    """
    Decoder-style dataset that normalises the behavior mix to a fixed ratio before training.

    For each user the full history is down-sampled once so that the number of lower-level
    interactions (e.g. play/click) does not exceed `level_ratios[level] / level_ratios[max_level]`
    times the number of target-behavior (max-level) interactions.  If the actual count is already
    below the target, nothing is dropped.

    The drop is applied deterministically per user (CRC-32 of uid as seed) so that
    train / valid / test share a consistent prefix:

      train  history  = drop(items[:valid_pos-1], seed=S)
      valid  history  = drop(items[:valid_pos-1], seed=S)  +  [items[valid_pos-1]]
      test   history  = valid_history  +  drop(items[valid_pos:test_pos], seed=S ^ 0xDEADBEEF)

    Each mode produces ONE sample per user (train) or one per target item (valid, as per the
    base-class convention) / one list per user (test).
    """

    def __init__(self, level_ratios: list[float] | None = None, **kwargs):
        # Store raw input; will be finalised in _load_data() once max_behavior_level is known.
        # level_ratios[i] = desired multiplier for behavior at level i relative to the
        # target (max-level) behavior.
        # Default heuristic: level 0 → 5.0, all higher levels → 1.0.
        self._level_ratios_input: list[float] | None = level_ratios
        # Set a temporary value so the property is always defined before _load_data runs
        self.level_ratios: list[float] = level_ratios if level_ratios is not None else []
        super().__init__(**kwargs)

    @property
    def cached_file_name(self) -> str:
        ratio_str = "_".join(str(r) for r in self.level_ratios)
        suffix = "" if self.behavior_first else ".behind"
        return os.path.join(
            self.data_path,
            self.dataset + f".{self.__class__.__name__}.{self.max_his_len}.SMB{suffix}.ratio{ratio_str}.{self.mode}{self.index_suffix}.pkl",
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_data(self):
        super()._load_data()
        # self.max_behavior_level is now available; finalise level_ratios to exactly
        # max_behavior_level + 1 elements so every level maps to a direct index.
        n_levels = self.max_behavior_level + 1
        if self._level_ratios_input is None:
            # Default: lowest level (0) gets 5×, all others get 1×
            self.level_ratios = [5.0] + [1.0] * (n_levels - 1)
        else:
            ratios = list(self._level_ratios_input)
            if len(ratios) < n_levels:
                # Pad with the last provided value
                ratios += [ratios[-1]] * (n_levels - len(ratios))
            self.level_ratios = ratios[:n_levels]
        logger.info(
            f"SMBFixedRatioDatasetForDecoder: {n_levels} behavior levels, "
            f"level_ratios = {self.level_ratios}"
        )

    def _user_seed(self, uid: str) -> int:
        return zlib.crc32(uid.encode()) & 0xFFFFFFFF

    def _drop_to_fixed_ratio(
        self,
        items: list[str],
        behaviors: list[str],
        sids: list[int],
        times: list[float],
        rng: np.random.Generator,
    ) -> tuple[list[str], list[str], list[int], list[float]]:
        """Randomly remove lower-level interactions so that the per-behavior counts match
        the configured level_ratios relative to the target-behavior count."""
        n_target = sum(1 for b in behaviors if b == self.target_behavior)
        if n_target == 0 or not items:
            return items, behaviors, sids, times

        # level_ratios is guaranteed to have exactly max_behavior_level+1 elements
        target_ratio = self.level_ratios[self.max_behavior_level]

        drop_indices: set[int] = set()
        for behavior, level in self.behavior_level.items():
            if behavior == self.target_behavior:
                continue
            indices = [i for i, b in enumerate(behaviors) if b == behavior]
            if not indices:
                continue
            ratio = self.level_ratios[level]
            target_count = int(n_target * ratio / target_ratio)
            if len(indices) > target_count:
                drop_idx = rng.choice(indices, size=len(indices) - target_count, replace=False)
                drop_indices.update(drop_idx.tolist())

        if not drop_indices:
            return items, behaviors, sids, times

        keep = np.ones(len(items), dtype=bool)
        for idx in drop_indices:
            keep[idx] = False

        items_arr = np.array(items)
        behaviors_arr = np.array(behaviors)
        sids_arr = np.array(sids)
        times_arr = np.array(times, dtype=float)
        return (
            items_arr[keep].tolist(),
            behaviors_arr[keep].tolist(),
            sids_arr[keep].tolist(),
            times_arr[keep].tolist(),
        )

    # ------------------------------------------------------------------
    # The three split processors
    # ------------------------------------------------------------------

    def _process_train_data(self) -> list[dict]:
        inter_data = []
        for uid in get_tqdm(self.remapped_inters, desc="Processing training data"):
            if self.valid_pos[uid] <= 0:
                continue
            pos = self.valid_pos[uid]
            rng = np.random.default_rng(self._user_seed(uid))

            items = self.remapped_inters[uid][:pos]
            behaviors = self.history_behaviors[uid][:pos]
            sids = self.session[uid][:pos]
            times = self.time[uid][:pos]

            # Drop the history (items before the last one); the last item is the target
            hist_items, hist_behaviors, hist_sids, hist_times = self._drop_to_fixed_ratio(
                items[:-1], behaviors[:-1], sids[:-1], times[:-1], rng
            )
            # Full sequence = dropped history + target
            all_items = hist_items + [items[-1]]
            all_behaviors = hist_behaviors + [behaviors[-1]]
            all_sids = hist_sids + [sids[-1]]
            all_times = hist_times + [times[-1]]
            if len(all_items) < 2:
                continue

            inter_data.append({
                "item": self.get_behavior_item(items[-1], behaviors[-1]),
                "inters": self._get_inters(hist_items, hist_behaviors),
                "session_ids": self._generate_session_ids(all_sids),
                "extended_session_ids": self._generate_extended_session_ids(all_sids),
                "actions": self._generate_actions(all_behaviors),
                "time": self._generate_times(all_times),
                "behavior": behaviors[-1],
            })
        return inter_data

    def _process_valid_data(self) -> list[dict]:
        inter_data = []
        for uid in get_tqdm(self.remapped_inters, desc="Processing validation data"):
            if self.valid_pos[uid] < 0:
                continue
            valid_pos = self.valid_pos[uid]
            rng = np.random.default_rng(self._user_seed(uid))

            items_full = self.remapped_inters[uid][: self.test_pos[uid]]
            behaviors_full = self.history_behaviors[uid][: self.test_pos[uid]]
            sids_full = self.session[uid][: self.test_pos[uid]]
            times_full = self.time[uid][: self.test_pos[uid]]

            # Drop items[:valid_pos-1] with the same seed as training
            hist_items, hist_behaviors, hist_sids, hist_times = self._drop_to_fixed_ratio(
                items_full[: valid_pos - 1],
                behaviors_full[: valid_pos - 1],
                sids_full[: valid_pos - 1],
                times_full[: valid_pos - 1],
                rng,
            )
            # Always include items[valid_pos-1] (== training target) in valid history
            if valid_pos > 0:
                hist_items = hist_items + [items_full[valid_pos - 1]]
                hist_behaviors = hist_behaviors + [behaviors_full[valid_pos - 1]]
                hist_sids = hist_sids + [sids_full[valid_pos - 1]]
                hist_times = hist_times + [times_full[valid_pos - 1]]

            for i in range(valid_pos, len(items_full)):
                item_i = items_full[i]
                behavior_i = behaviors_full[i]
                sid_i = sids_full[i]
                time_i = times_full[i]
                all_sids = hist_sids + [sid_i]
                all_behaviors = hist_behaviors + [behavior_i]
                all_times = hist_times + [time_i]
                inter_data.append({
                    "item": self.get_behavior_item(item_i, behavior_i),
                    "inters": self._get_inters(hist_items, hist_behaviors),
                    "session_ids": self._generate_session_ids(all_sids),
                    "extended_session_ids": self._generate_extended_session_ids(all_sids),
                    "actions": self._generate_actions(all_behaviors),
                    "time": self._generate_times(all_times),
                    "behavior": behavior_i,
                })
        return inter_data

    def _process_test_data(self) -> list[dict]:
        inter_data = []
        for uid in get_tqdm(self.remapped_inters, desc="Processing test data"):
            test_pos = self.test_pos[uid]
            valid_pos = self.valid_pos[uid]
            rng_base = np.random.default_rng(self._user_seed(uid))
            rng_ext = np.random.default_rng(self._user_seed(uid) ^ 0xDEADBEEF)

            items = self.remapped_inters[uid]
            behaviors = self.history_behaviors[uid]
            sids = self.session[uid]
            times = self.time[uid]

            session_items = [self.get_behavior_item(items[i], behaviors[i]) for i in range(test_pos, len(items))]
            session_behaviors = behaviors[test_pos:]
            assert len(session_items) > 0, f"Empty test session for user {uid}."

            if valid_pos > 0:
                # Consistent prefix: same drop as train/valid on [:valid_pos-1]
                hist_items, hist_behaviors, hist_sids, hist_times = self._drop_to_fixed_ratio(
                    items[: valid_pos - 1],
                    behaviors[: valid_pos - 1],
                    sids[: valid_pos - 1],
                    times[: valid_pos - 1],
                    rng_base,
                )
                # Always keep items[valid_pos-1] (the training boundary item)
                hist_items = hist_items + [items[valid_pos - 1]]
                hist_behaviors = hist_behaviors + [behaviors[valid_pos - 1]]
                hist_sids = hist_sids + [sids[valid_pos - 1]]
                hist_times = hist_times + [times[valid_pos - 1]]
                # Extend with dropped [valid_pos:test_pos]
                ext_items, ext_behaviors, ext_sids, ext_times = self._drop_to_fixed_ratio(
                    items[valid_pos:test_pos],
                    behaviors[valid_pos:test_pos],
                    sids[valid_pos:test_pos],
                    times[valid_pos:test_pos],
                    rng_ext,
                )
                hist_items = hist_items + ext_items
                hist_behaviors = hist_behaviors + ext_behaviors
                hist_sids = hist_sids + ext_sids
                hist_times = hist_times + ext_times
            else:
                # No valid split: drop the entire history with base seed
                hist_items, hist_behaviors, hist_sids, hist_times = self._drop_to_fixed_ratio(
                    items[:test_pos],
                    behaviors[:test_pos],
                    sids[:test_pos],
                    times[:test_pos],
                    rng_base,
                )

            inter_data.append({
                "uid": uid,
                "item": session_items,
                "inters": self._get_inters(hist_items, hist_behaviors),
                "inters_item_list": self._get_inters_with_only_items(hist_items),
                "session_ids": self._generate_session_ids(hist_sids),
                "extended_session_ids": self._generate_extended_session_ids(hist_sids),
                "actions": self._generate_actions(hist_behaviors),
                "time": self._generate_times(hist_times + [times[test_pos]]),
                "behavior": session_behaviors,
            })
        return inter_data
