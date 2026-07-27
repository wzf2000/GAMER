import copy
import os

from SeqRec.datasets.session_behavior.explicit import SMBExplicitDataset
from SeqRec.utils.runtime import get_tqdm


class SMBRankingDatasetForDecoder(SMBExplicitDataset):
    """
    Decoder-only SMB ranking data.

    Input is history interactions plus a raw candidate item.  The target behavior
    token is kept only in labels, and relation_actions carries the explicit
    behavior/action indices used by relation-bias.
    """

    def __init__(self, **kwargs):
        super().__init__(behavior_first=True, **kwargs)

    @property
    def cached_file_name(self) -> str:
        return os.path.join(
            self.data_path,
            self.dataset
            + f".{self.__class__.__name__}.{self.max_his_len}.SMB.ranking.{self.mode}{self.index_suffix}.pkl",
        )

    def _behavior_label(self, behavior: str) -> str:
        return f"<behavior_{behavior}>"

    def _relation_action_id(self, behavior: str) -> int:
        return self.behavior_level[behavior] + 1

    def _trim_history(
        self,
        items: list[str],
        behaviors: list[str],
        session_ids: list[int],
    ) -> tuple[list[str], list[str], list[int]]:
        if self.max_his_len > 0:
            return (
                items[-self.max_his_len :],
                behaviors[-self.max_his_len :],
                session_ids[-self.max_his_len :],
            )
        return items, behaviors, session_ids

    def _target_session_data(
        self,
        uid: str,
        target_offset: int,
    ) -> tuple[list[str], list[str], list[int], list[str], list[str], list[int], int] | None:
        user_session_ids = self.session[uid]
        unique_session_ids = sorted(set(user_session_ids))
        if len(unique_session_ids) < 3:
            return None

        target_session_id = unique_session_ids[target_offset]
        ordered_indices = sorted(
            range(len(user_session_ids)),
            key=lambda index: (user_session_ids[index], index),
        )
        history_indices = [
            index
            for index in ordered_indices
            if user_session_ids[index] < target_session_id
        ]
        target_indices = [
            index
            for index in ordered_indices
            if user_session_ids[index] == target_session_id
        ]
        if not target_indices:
            return None

        history_items = [self.remapped_inters[uid][index] for index in history_indices]
        history_behaviors = [self.history_behaviors[uid][index] for index in history_indices]
        history_sessions = [user_session_ids[index] for index in history_indices]
        history_items, history_behaviors, history_sessions = self._trim_history(
            history_items,
            history_behaviors,
            history_sessions,
        )

        target_items = [self.remapped_inters[uid][index] for index in target_indices]
        target_behaviors = [self.history_behaviors[uid][index] for index in target_indices]
        target_sessions = [user_session_ids[index] for index in target_indices]
        return (
            history_items,
            history_behaviors,
            history_sessions,
            target_items,
            target_behaviors,
            target_sessions,
            target_session_id,
        )

    def _history_relation_actions(self, history_behaviors: list[str]) -> list[int]:
        relation_actions: list[int] = []
        for behavior in history_behaviors:
            relation_actions.extend([self._relation_action_id(behavior)] * self.token_count())
        return relation_actions

    def _history_session_ids(self, history_session_ids: list[int]) -> list[int]:
        session_ids: list[int] = []
        for session_id in history_session_ids:
            session_ids.extend([session_id] * self.token_count())
        return session_ids

    def _history_extended_session_ids(self, history_session_ids: list[int]) -> list[int]:
        extended_session_ids: list[int] = []
        last_session_id: int | None = None
        remapped_session_id = -1
        for session_id in history_session_ids:
            if session_id != last_session_id:
                last_session_id = session_id
                remapped_session_id += 1
            base = remapped_session_id * self.token_count()
            extended_session_ids.extend(base + index for index in range(self.token_count()))
        return extended_session_ids

    def _build_candidate_sample(
        self,
        *,
        uid: str,
        history_items: list[str],
        history_behaviors: list[str],
        history_sessions: list[int],
        candidate_item: str,
        behavior: str,
        target_session_id: int,
    ) -> dict:
        input_ids = self._get_inters(history_items, history_behaviors) + candidate_item
        relation_actions = self._history_relation_actions(history_behaviors)
        session_ids = self._history_session_ids(history_sessions)
        extended_session_ids = self._history_extended_session_ids(history_sessions)
        candidate_session_ids = [target_session_id] * self.sole_item_len
        candidate_relation_actions = [0] * self.sole_item_len
        next_extended = (max(extended_session_ids) + 1) if extended_session_ids else 0

        return {
            "uid": uid,
            "input_ids": input_ids,
            "labels": self._behavior_label(behavior),
            "relation_actions": relation_actions + candidate_relation_actions,
            "actions": relation_actions + candidate_relation_actions,
            "session_ids": session_ids + candidate_session_ids,
            "extended_session_ids": extended_session_ids
            + [next_extended + index for index in range(self.sole_item_len)],
            "behavior": behavior,
            "target_item": candidate_item,
            "target_session_id": target_session_id,
            "split": self.mode,
        }

    def _build_session_sample(
        self,
        *,
        uid: str,
        history_items: list[str],
        history_behaviors: list[str],
        history_sessions: list[int],
        target_items: list[str],
        target_behaviors: list[str],
        target_session_id: int,
    ) -> dict:
        return {
            "uid": uid,
            "input_ids": self._get_inters(history_items, history_behaviors),
            "labels": target_items,
            "relation_actions": self._history_relation_actions(history_behaviors),
            "actions": self._history_relation_actions(history_behaviors),
            "session_ids": self._history_session_ids(history_sessions),
            "extended_session_ids": self._history_extended_session_ids(history_sessions),
            "behavior": target_behaviors,
            "target_item": target_items,
            "target_session_id": target_session_id,
            "split": self.mode,
        }

    def _process_split_candidates(self, target_offset: int, desc: str) -> list[dict]:
        inter_data = []
        for uid in get_tqdm(self.remapped_inters, desc=desc):
            session_data = self._target_session_data(uid, target_offset)
            if session_data is None:
                continue
            (
                history_items,
                history_behaviors,
                history_sessions,
                target_items,
                target_behaviors,
                _target_sessions,
                target_session_id,
            ) = session_data
            for candidate_item, behavior in zip(target_items, target_behaviors):
                inter_data.append(
                    self._build_candidate_sample(
                        uid=uid,
                        history_items=history_items,
                        history_behaviors=history_behaviors,
                        history_sessions=history_sessions,
                        candidate_item=candidate_item,
                        behavior=behavior,
                        target_session_id=target_session_id,
                    )
                )
        return inter_data

    def _process_train_data(self) -> list[dict]:
        return self._process_split_candidates(-3, "Processing SMB ranking training data")

    def _process_valid_data(self) -> list[dict]:
        return self._process_split_candidates(-2, "Processing SMB ranking validation data")

    def _process_test_data(self) -> list[dict]:
        inter_data = []
        for uid in get_tqdm(self.remapped_inters, desc="Processing SMB ranking test data"):
            session_data = self._target_session_data(uid, -1)
            if session_data is None:
                continue
            (
                history_items,
                history_behaviors,
                history_sessions,
                target_items,
                target_behaviors,
                _target_sessions,
                target_session_id,
            ) = session_data
            inter_data.append(
                self._build_session_sample(
                    uid=uid,
                    history_items=history_items,
                    history_behaviors=history_behaviors,
                    history_sessions=history_sessions,
                    target_items=target_items,
                    target_behaviors=target_behaviors,
                    target_session_id=target_session_id,
                )
            )
        return inter_data

    def _process_valid_test_data(self) -> list[dict]:
        inter_data = []
        for uid in get_tqdm(self.remapped_inters, desc="Processing SMB ranking valid-test data"):
            session_data = self._target_session_data(uid, -2)
            if session_data is None:
                continue
            (
                history_items,
                history_behaviors,
                history_sessions,
                target_items,
                target_behaviors,
                _target_sessions,
                target_session_id,
            ) = session_data
            inter_data.append(
                self._build_session_sample(
                    uid=uid,
                    history_items=history_items,
                    history_behaviors=history_behaviors,
                    history_sessions=history_sessions,
                    target_items=target_items,
                    target_behaviors=target_behaviors,
                    target_session_id=target_session_id,
                )
            )
        return inter_data

    def filter_by_behavior(self, behavior: str) -> "SMBRankingDatasetForDecoder":
        if not self.inter_data:
            copied_dataset = copy.copy(self)
            copied_dataset.inter_data = []
            copied_dataset.target_behavior = behavior
            return copied_dataset

        if isinstance(self.inter_data[0]["behavior"], list):
            filtered_data = []
            for sample in self.inter_data:
                target_items = [
                    item
                    for item, sample_behavior in zip(sample["target_item"], sample["behavior"])
                    if sample_behavior == behavior
                ]
                if not target_items:
                    continue
                filtered_sample = dict(sample)
                filtered_sample["labels"] = target_items
                filtered_sample["target_item"] = target_items
                filtered_sample["behavior"] = [behavior] * len(target_items)
                filtered_data.append(filtered_sample)
        else:
            filtered_data = [
                sample
                for sample in self.inter_data
                if sample["behavior"] == behavior
            ]

        copied_dataset = copy.copy(self)
        copied_dataset.inter_data = filtered_data
        copied_dataset.target_behavior = behavior
        return copied_dataset

    def __getitem__(self, index: int) -> dict:
        return dict(self.inter_data[index])
