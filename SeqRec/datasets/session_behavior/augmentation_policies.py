from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True)
class BehaviorSequence:
    items: list[str]
    behaviors: list[str]
    session_ids: list[int]
    times: list[float]

    def __post_init__(self):
        lengths = {
            len(self.items),
            len(self.behaviors),
            len(self.session_ids),
            len(self.times),
        }
        if len(lengths) != 1:
            raise ValueError("All behavior sequence fields must have the same length.")

    def select(self, keep_indices: list[int]) -> "BehaviorSequence":
        return BehaviorSequence(
            items=[self.items[index] for index in keep_indices],
            behaviors=[self.behaviors[index] for index in keep_indices],
            session_ids=[self.session_ids[index] for index in keep_indices],
            times=[self.times[index] for index in keep_indices],
        )


@dataclass(frozen=True)
class AugmentationContext:
    uid: str
    target_behavior: str
    target_level: int
    target_time: float
    behavior_level: dict[str, int]
    max_behavior_level: int


@dataclass(frozen=True)
class AugmentedView:
    name: str
    keep_indices: list[int]
    metadata: dict[str, Any]


class SequenceAugmentationPolicy(Protocol):
    name: str

    def generate_view(
        self,
        sequence: BehaviorSequence,
        context: AugmentationContext,
        rng: np.random.Generator,
    ) -> AugmentedView:
        ...

    def cache_config(self) -> dict[str, Any]:
        ...


def _restore_minimum_history(
    keep_mask: np.ndarray,
    scores: np.ndarray,
    min_history_items: int,
) -> np.ndarray:
    required = min(min_history_items, len(keep_mask))
    missing = required - int(keep_mask.sum())
    if missing <= 0:
        return keep_mask
    dropped_indices = np.flatnonzero(~keep_mask)
    restore_order = dropped_indices[np.argsort(scores[dropped_indices])]
    keep_mask[restore_order[:missing]] = True
    return keep_mask


@dataclass(frozen=True)
class TimeDecayDropoutPolicy:
    tau: float = 48.0
    severity: float = 0.5
    max_drop_probability: float = 0.9
    min_recent_items: int = 1
    min_history_items: int = 1
    preserve_target_level: bool = True
    decay_type: str = "exponential"
    name: str = "time_decay"

    def __post_init__(self):
        if self.tau <= 0:
            raise ValueError("time_decay_tau must be greater than 0.")
        if not 0 <= self.severity <= 1:
            raise ValueError("time_decay_severity must be in [0, 1].")
        if not 0 <= self.max_drop_probability <= 1:
            raise ValueError("time_decay_max_drop must be in [0, 1].")
        if self.min_recent_items < 0 or self.min_history_items < 0:
            raise ValueError("Minimum history settings must be non-negative.")
        if self.decay_type not in {"exponential", "linear_rank", "bucket"}:
            raise ValueError(f"Unsupported time decay type: {self.decay_type}.")

    def _age_weights(
        self,
        sequence: BehaviorSequence,
        context: AugmentationContext,
    ) -> np.ndarray:
        length = len(sequence.items)
        if length == 0:
            return np.array([], dtype=float)
        if self.decay_type == "linear_rank":
            if length == 1:
                return np.zeros(1, dtype=float)
            return np.linspace(1.0, 0.0, num=length)

        age = np.maximum(
            context.target_time - np.asarray(sequence.times, dtype=float),
            0.0,
        )
        if self.decay_type == "bucket":
            return np.select(
                [age <= self.tau, age <= 3 * self.tau],
                [0.0, 0.5],
                default=1.0,
            )
        return 1.0 - np.exp(-age / self.tau)

    def generate_view(
        self,
        sequence: BehaviorSequence,
        context: AugmentationContext,
        rng: np.random.Generator,
    ) -> AugmentedView:
        age_weights = self._age_weights(sequence, context)
        level_weights = np.asarray([
            1.0 / (context.behavior_level[behavior] + 1)
            for behavior in sequence.behaviors
        ])
        drop_probabilities = np.minimum(
            self.severity * age_weights * level_weights,
            self.max_drop_probability,
        )
        if self.preserve_target_level:
            for index, behavior in enumerate(sequence.behaviors):
                if context.behavior_level[behavior] == context.max_behavior_level:
                    drop_probabilities[index] = 0.0

        keep_mask = rng.random(len(sequence.items)) >= drop_probabilities
        if self.min_recent_items:
            keep_mask[-self.min_recent_items:] = True
        keep_mask = _restore_minimum_history(
            keep_mask,
            drop_probabilities,
            self.min_history_items,
        )
        keep_indices = np.flatnonzero(keep_mask).tolist()
        return AugmentedView(
            name=self.name,
            keep_indices=keep_indices,
            metadata={
                "mean_drop_probability": (
                    float(drop_probabilities.mean())
                    if len(drop_probabilities)
                    else 0.0
                ),
            },
        )

    def cache_config(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "tau": self.tau,
            "severity": self.severity,
            "max_drop_probability": self.max_drop_probability,
            "min_recent_items": self.min_recent_items,
            "min_history_items": self.min_history_items,
            "preserve_target_level": self.preserve_target_level,
            "decay_type": self.decay_type,
        }


@dataclass(frozen=True)
class SessionAwareDropoutPolicy:
    recent_session_count: int = 1
    base_keep_probability: float = 0.5
    time_decay_tau: float = 7.0
    high_level_bonus: float = 0.3
    preserve_target_level: bool = True
    min_history_items: int = 1
    name: str = "session"

    def __post_init__(self):
        if self.recent_session_count < 0:
            raise ValueError("recent_session_count must be non-negative.")
        if not 0 <= self.base_keep_probability <= 1:
            raise ValueError("session_keep_probability must be in [0, 1].")
        if self.time_decay_tau <= 0:
            raise ValueError("session_time_decay_tau must be greater than 0.")
        if not 0 <= self.high_level_bonus <= 1:
            raise ValueError("session_high_level_bonus must be in [0, 1].")

    def generate_view(
        self,
        sequence: BehaviorSequence,
        context: AugmentationContext,
        rng: np.random.Generator,
    ) -> AugmentedView:
        if not sequence.items:
            return AugmentedView(self.name, [], {"kept_sessions": 0})

        session_indices: dict[int, list[int]] = {}
        for index, session_id in enumerate(sequence.session_ids):
            session_indices.setdefault(session_id, []).append(index)
        session_ids = list(session_indices)
        protected_sessions = set(session_ids[-self.recent_session_count:])

        if self.preserve_target_level:
            high_level_sessions = [
                session_id
                for session_id, indices in session_indices.items()
                if any(
                    context.behavior_level[sequence.behaviors[index]]
                    == context.max_behavior_level
                    for index in indices
                )
            ]
            if high_level_sessions:
                protected_sessions.add(high_level_sessions[-1])

        latest_session_position = len(session_ids) - 1
        kept_sessions = set(protected_sessions)
        for position, session_id in enumerate(session_ids):
            if session_id in protected_sessions:
                continue
            distance = latest_session_position - position
            recency_weight = np.exp(-distance / self.time_decay_tau)
            max_level = max(
                context.behavior_level[sequence.behaviors[index]]
                for index in session_indices[session_id]
            )
            level_bonus = self.high_level_bonus * (
                max_level / max(context.max_behavior_level, 1)
            )
            keep_probability = min(
                1.0,
                self.base_keep_probability * recency_weight + level_bonus,
            )
            if rng.random() < keep_probability:
                kept_sessions.add(session_id)

        keep_mask = np.asarray([
            session_id in kept_sessions
            for session_id in sequence.session_ids
        ])
        required = min(self.min_history_items, len(sequence.items))
        if int(keep_mask.sum()) < required:
            remaining_sessions = [
                session_id
                for session_id in reversed(session_ids)
                if session_id not in kept_sessions
            ]
            for session_id in remaining_sessions:
                kept_sessions.add(session_id)
                for index in session_indices[session_id]:
                    keep_mask[index] = True
                if int(keep_mask.sum()) >= required:
                    break
        keep_indices = np.flatnonzero(keep_mask).tolist()
        return AugmentedView(
            name=self.name,
            keep_indices=keep_indices,
            metadata={
                "kept_sessions": len({
                    sequence.session_ids[index]
                    for index in keep_indices
                }),
            },
        )

    def cache_config(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "recent_session_count": self.recent_session_count,
            "base_keep_probability": self.base_keep_probability,
            "time_decay_tau": self.time_decay_tau,
            "high_level_bonus": self.high_level_bonus,
            "preserve_target_level": self.preserve_target_level,
            "min_history_items": self.min_history_items,
        }


@dataclass(frozen=True)
class DatasetProportionPolicy:
    target_proportions: tuple[float, ...]
    tolerance: float = 1.2
    min_history_items: int = 1
    preserve_target_level: bool = True
    name: str = "dataset_proportion"

    def __post_init__(self):
        if not self.target_proportions:
            raise ValueError("target_proportions must not be empty.")
        if any(proportion < 0 for proportion in self.target_proportions):
            raise ValueError("target_proportions must be non-negative.")
        if sum(self.target_proportions) <= 0:
            raise ValueError("target_proportions must contain a positive value.")
        if self.tolerance <= 0:
            raise ValueError(
                "dataset_proportion_tolerance must be greater than 0."
            )

    def generate_view(
        self,
        sequence: BehaviorSequence,
        context: AugmentationContext,
        rng: np.random.Generator,
    ) -> AugmentedView:
        length = len(sequence.items)
        if length == 0:
            return AugmentedView(self.name, [], {"dropped_items": 0})

        keep_mask = np.ones(length, dtype=bool)
        proportions = np.asarray(self.target_proportions, dtype=float)
        proportions = proportions / proportions.sum()
        max_levels = min(len(proportions), context.max_behavior_level + 1)
        for level in range(max_levels):
            if self.preserve_target_level and level == context.max_behavior_level:
                continue
            level_indices = [
                index
                for index, behavior in enumerate(sequence.behaviors)
                if context.behavior_level[behavior] == level
            ]
            max_count = int(np.ceil(
                length * proportions[level] * self.tolerance
            ))
            drop_count = max(0, len(level_indices) - max_count)
            if drop_count:
                dropped = rng.choice(
                    level_indices,
                    size=drop_count,
                    replace=False,
                )
                keep_mask[dropped] = False

        keep_mask = _restore_minimum_history(
            keep_mask,
            np.ones(length, dtype=float),
            self.min_history_items,
        )
        keep_indices = np.flatnonzero(keep_mask).tolist()
        return AugmentedView(
            name=self.name,
            keep_indices=keep_indices,
            metadata={"dropped_items": length - len(keep_indices)},
        )

    def cache_config(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "target_proportions": self.target_proportions,
            "tolerance": self.tolerance,
            "min_history_items": self.min_history_items,
            "preserve_target_level": self.preserve_target_level,
        }
