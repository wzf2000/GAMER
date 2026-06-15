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

    def generate_views(
        self,
        sequence: BehaviorSequence,
        context: AugmentationContext,
        rng: np.random.Generator,
    ) -> list[AugmentedView]:
        ...

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

    def generate_views(
        self,
        sequence: BehaviorSequence,
        context: AugmentationContext,
        rng: np.random.Generator,
    ) -> list[AugmentedView]:
        return [self.generate_view(sequence, context, rng)]

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

    def generate_views(
        self,
        sequence: BehaviorSequence,
        context: AugmentationContext,
        rng: np.random.Generator,
    ) -> list[AugmentedView]:
        return [self.generate_view(sequence, context, rng)]

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

    def generate_views(
        self,
        sequence: BehaviorSequence,
        context: AugmentationContext,
        rng: np.random.Generator,
    ) -> list[AugmentedView]:
        return [self.generate_view(sequence, context, rng)]

    def cache_config(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "target_proportions": self.target_proportions,
            "tolerance": self.tolerance,
            "min_history_items": self.min_history_items,
            "preserve_target_level": self.preserve_target_level,
        }


@dataclass(frozen=True)
class UserAdaptiveRatioPolicy:
    global_proportions: tuple[float, ...]
    smoothing: float = 5.0
    confidence_scale: float = 20.0
    min_ratio: float = 0.25
    max_ratio: float = 20.0
    tolerance: float = 1.0
    min_history_items: int = 1
    preserve_target_level: bool = True
    name: str = "user_adaptive_ratio"

    def __post_init__(self):
        if not self.global_proportions:
            raise ValueError("global_proportions must not be empty.")
        if any(value < 0 for value in self.global_proportions):
            raise ValueError("global_proportions must be non-negative.")
        if sum(self.global_proportions) <= 0:
            raise ValueError("global_proportions must contain a positive value.")
        if self.smoothing <= 0 or self.confidence_scale <= 0:
            raise ValueError("Adaptive ratio smoothing values must be positive.")
        if self.min_ratio < 0 or self.max_ratio < self.min_ratio:
            raise ValueError("Invalid adaptive ratio bounds.")
        if self.tolerance <= 0:
            raise ValueError("user_adaptive_tolerance must be positive.")

    def generate_view(
        self,
        sequence: BehaviorSequence,
        context: AugmentationContext,
        rng: np.random.Generator,
    ) -> AugmentedView:
        length = len(sequence.items)
        if length == 0:
            return AugmentedView(self.name, [], {"dropped_items": 0})

        level_count = context.max_behavior_level + 1
        counts = np.zeros(level_count, dtype=int)
        for behavior in sequence.behaviors:
            counts[context.behavior_level[behavior]] += 1

        proportions = np.asarray(self.global_proportions[:level_count], dtype=float)
        if len(proportions) < level_count:
            proportions = np.pad(
                proportions,
                (0, level_count - len(proportions)),
                constant_values=proportions[-1],
            )
        proportions = proportions / proportions.sum()
        target_level = context.max_behavior_level
        global_target_share = max(proportions[target_level], 1e-8)
        global_ratios = proportions / global_target_share
        target_count = int(counts[target_level])
        confidence = length / (length + self.confidence_scale)

        keep_mask = np.ones(length, dtype=bool)
        final_ratios = []
        for level in range(level_count):
            if self.preserve_target_level and level == target_level:
                final_ratios.append(1.0)
                continue
            user_ratio = (
                counts[level] + self.smoothing * global_ratios[level]
            ) / (target_count + self.smoothing)
            final_ratio = (
                confidence * user_ratio
                + (1.0 - confidence) * global_ratios[level]
            )
            final_ratio = float(np.clip(
                final_ratio,
                self.min_ratio,
                self.max_ratio,
            ))
            final_ratios.append(final_ratio)
            if target_count:
                cap = int(np.ceil(
                    target_count * final_ratio * self.tolerance
                ))
            else:
                cap = int(np.ceil(
                    length * proportions[level] * self.tolerance
                ))
            level_indices = [
                index
                for index, behavior in enumerate(sequence.behaviors)
                if context.behavior_level[behavior] == level
            ]
            drop_count = max(0, len(level_indices) - cap)
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
            metadata={
                "dropped_items": length - len(keep_indices),
                "final_ratios": final_ratios,
                "target_count": target_count,
            },
        )

    def generate_views(
        self,
        sequence: BehaviorSequence,
        context: AugmentationContext,
        rng: np.random.Generator,
    ) -> list[AugmentedView]:
        return [self.generate_view(sequence, context, rng)]

    def cache_config(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "global_proportions": self.global_proportions,
            "smoothing": self.smoothing,
            "confidence_scale": self.confidence_scale,
            "min_ratio": self.min_ratio,
            "max_ratio": self.max_ratio,
            "tolerance": self.tolerance,
            "min_history_items": self.min_history_items,
            "preserve_target_level": self.preserve_target_level,
        }


@dataclass(frozen=True)
class TargetConditionedPolicy:
    base_policy: SequenceAugmentationPolicy
    same_level_restore_probability: float = 0.8
    precursor_restore_probability: float = 0.8
    min_history_items: int = 1
    name: str = "target_conditioned"

    def __post_init__(self):
        probabilities = (
            self.same_level_restore_probability,
            self.precursor_restore_probability,
        )
        if any(not 0 <= value <= 1 for value in probabilities):
            raise ValueError("Target-conditioned probabilities must be in [0, 1].")

    def generate_view(
        self,
        sequence: BehaviorSequence,
        context: AugmentationContext,
        rng: np.random.Generator,
    ) -> AugmentedView:
        base_view = self.base_policy.generate_views(
            sequence,
            context,
            rng,
        )[0]
        keep_mask = np.zeros(len(sequence.items), dtype=bool)
        keep_mask[base_view.keep_indices] = True
        precursor_level = max(context.target_level - 1, 0)
        for index, behavior in enumerate(sequence.behaviors):
            if keep_mask[index]:
                continue
            level = context.behavior_level[behavior]
            if level == context.target_level:
                restore_probability = self.same_level_restore_probability
            elif (
                context.target_level > 0
                and level == precursor_level
            ):
                restore_probability = self.precursor_restore_probability
            else:
                continue
            if rng.random() < restore_probability:
                keep_mask[index] = True

        keep_mask = _restore_minimum_history(
            keep_mask,
            np.ones(len(sequence.items), dtype=float),
            self.min_history_items,
        )
        keep_indices = np.flatnonzero(keep_mask).tolist()
        return AugmentedView(
            name=self.name,
            keep_indices=keep_indices,
            metadata={
                "base_view": base_view.name,
                "target_level": context.target_level,
                "restored_items": (
                    len(keep_indices) - len(base_view.keep_indices)
                ),
            },
        )

    def generate_views(
        self,
        sequence: BehaviorSequence,
        context: AugmentationContext,
        rng: np.random.Generator,
    ) -> list[AugmentedView]:
        return [self.generate_view(sequence, context, rng)]

    def cache_config(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "base_policy": self.base_policy.cache_config(),
            "same_level_restore_probability": (
                self.same_level_restore_probability
            ),
            "precursor_restore_probability": (
                self.precursor_restore_probability
            ),
            "min_history_items": self.min_history_items,
        }


@dataclass(frozen=True)
class MultiViewAugmentationPolicy:
    recent_policy: TimeDecayDropoutPolicy
    session_policy: SessionAwareDropoutPolicy
    min_history_items: int = 1
    include_recent: bool = True
    include_hierarchy: bool = True
    include_session: bool = True
    name: str = "multi_view"

    def __post_init__(self):
        if not any((
            self.include_recent,
            self.include_hierarchy,
            self.include_session,
        )):
            raise ValueError("At least one multi-view component must be enabled.")

    def _hierarchy_view(
        self,
        sequence: BehaviorSequence,
        context: AugmentationContext,
    ) -> AugmentedView:
        relevant_levels = {context.target_level}
        if context.target_level > 0:
            relevant_levels.add(context.target_level - 1)
        keep_mask = np.asarray([
            context.behavior_level[behavior] in relevant_levels
            for behavior in sequence.behaviors
        ])
        if len(keep_mask):
            keep_mask[-1] = True
        keep_mask = _restore_minimum_history(
            keep_mask,
            np.ones(len(sequence.items), dtype=float),
            self.min_history_items,
        )
        return AugmentedView(
            name="multi_view_hierarchy",
            keep_indices=np.flatnonzero(keep_mask).tolist(),
            metadata={"relevant_levels": sorted(relevant_levels)},
        )

    def generate_views(
        self,
        sequence: BehaviorSequence,
        context: AugmentationContext,
        rng: np.random.Generator,
    ) -> list[AugmentedView]:
        child_seeds = rng.integers(
            0,
            np.iinfo(np.uint32).max,
            size=2,
            dtype=np.uint32,
        )
        views = []
        if self.include_recent:
            recent_view = self.recent_policy.generate_view(
                sequence,
                context,
                np.random.default_rng(int(child_seeds[0])),
            )
            views.append(AugmentedView(
                name="multi_view_recent",
                keep_indices=recent_view.keep_indices,
                metadata=recent_view.metadata,
            ))
        if self.include_hierarchy:
            views.append(self._hierarchy_view(sequence, context))
        if self.include_session:
            session_view = self.session_policy.generate_view(
                sequence,
                context,
                np.random.default_rng(int(child_seeds[1])),
            )
            views.append(AugmentedView(
                name="multi_view_session",
                keep_indices=session_view.keep_indices,
                metadata=session_view.metadata,
            ))
        return views

    def generate_view(
        self,
        sequence: BehaviorSequence,
        context: AugmentationContext,
        rng: np.random.Generator,
    ) -> AugmentedView:
        return self.generate_views(sequence, context, rng)[0]

    def cache_config(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "recent_policy": self.recent_policy.cache_config(),
            "session_policy": self.session_policy.cache_config(),
            "min_history_items": self.min_history_items,
            "include_recent": self.include_recent,
            "include_hierarchy": self.include_hierarchy,
            "include_session": self.include_session,
        }
