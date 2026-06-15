import hashlib
import json
import os
import zlib
from typing import Any

import numpy as np
from loguru import logger

from SeqRec.datasets.session_behavior.augmentation_policies import (
    AugmentationContext,
    BehaviorSequence,
    DatasetProportionPolicy,
    MultiViewAugmentationPolicy,
    SequenceAugmentationPolicy,
    SessionAwareDropoutPolicy,
    TargetConditionedPolicy,
    TimeDecayDropoutPolicy,
    UserAdaptiveRatioPolicy,
)
from SeqRec.datasets.session_behavior.decoder import SMBExplicitDatasetForDecoder
from SeqRec.datasets.session_behavior.statistics import (
    BehaviorLevelStatistics,
    compute_training_level_statistics,
)
from SeqRec.utils.runtime import get_tqdm


class SMBPolicyAugmentedDatasetForDecoder(SMBExplicitDatasetForDecoder):
    def __init__(
        self,
        sequence_augmentation: str,
        augmentation_views: int = 1,
        augmentation_seed: int = 42,
        augmentation_drop_original: bool = False,
        augmentation_config: dict[str, Any] | None = None,
        **kwargs,
    ):
        if augmentation_views < 1:
            raise ValueError(
                "augmentation_views must be greater than or equal to 1."
            )
        self.sequence_augmentation = sequence_augmentation
        self.augmentation_views = augmentation_views
        self.augmentation_seed = augmentation_seed
        self.augmentation_drop_original = augmentation_drop_original
        self.augmentation_config = augmentation_config or {}
        self.augmentation_policy: SequenceAugmentationPolicy | None = None
        self.training_statistics: BehaviorLevelStatistics | None = None
        super().__init__(augment=None, **kwargs)

    def _load_data(self):
        super()._load_data()
        self.training_statistics = compute_training_level_statistics(
            histories=self.history_behaviors,
            valid_positions=self.valid_pos,
            behavior_level=self.behavior_level,
            max_behavior_level=self.max_behavior_level,
        )
        self.augmentation_policy = self._build_policy()
        logger.info(
            "Using sequence augmentation policy {} with config {}.",
            self.sequence_augmentation,
            self.augmentation_policy.cache_config(),
        )

    @property
    def cached_file_name(self) -> str:
        if self.augmentation_policy is None:
            raise RuntimeError(
                "Augmentation policy must be initialized before cache lookup."
            )
        cache_payload = {
            "policy": self.augmentation_policy.cache_config(),
            "views": self.augmentation_views,
            "seed": self.augmentation_seed,
            "drop_original": self.augmentation_drop_original,
        }
        cache_hash = hashlib.sha256(
            json.dumps(
                cache_payload,
                sort_keys=True,
                default=list,
            ).encode()
        ).hexdigest()[:12]
        suffix = "" if self.behavior_first else ".behind"
        return os.path.join(
            self.data_path,
            (
                f"{self.dataset}.{self.__class__.__name__}.{self.max_his_len}"
                f".SMB{suffix}.{self.sequence_augmentation}.{cache_hash}"
                f".{self.mode}{self.index_suffix}.pkl"
            ),
        )

    def _training_level_proportions(self) -> tuple[float, ...]:
        if self.training_statistics is None:
            raise RuntimeError("Training statistics are not initialized.")
        return self.training_statistics.level_proportions

    def _build_time_decay_policy(self) -> TimeDecayDropoutPolicy:
        config = self.augmentation_config
        return TimeDecayDropoutPolicy(
            tau=config.get("time_decay_tau", 48.0),
            severity=config.get("time_decay_severity", 0.5),
            max_drop_probability=config.get(
                "time_decay_max_drop",
                0.9,
            ),
            min_recent_items=config.get(
                "time_decay_min_recent_items",
                1,
            ),
            min_history_items=config.get(
                "augmentation_min_history_items",
                1,
            ),
            preserve_target_level=not config.get(
                "time_decay_allow_target_level_drop",
                False,
            ),
            decay_type=config.get(
                "time_decay_type",
                "exponential",
            ),
        )

    def _build_session_policy(self) -> SessionAwareDropoutPolicy:
        config = self.augmentation_config
        return SessionAwareDropoutPolicy(
            recent_session_count=config.get(
                "recent_session_count",
                1,
            ),
            base_keep_probability=config.get(
                "session_keep_probability",
                0.5,
            ),
            time_decay_tau=config.get(
                "session_time_decay_tau",
                7.0,
            ),
            high_level_bonus=config.get(
                "session_high_level_bonus",
                0.3,
            ),
            preserve_target_level=not config.get(
                "session_allow_target_level_drop",
                False,
            ),
            min_history_items=config.get(
                "augmentation_min_history_items",
                1,
            ),
        )

    def _build_policy(self) -> SequenceAugmentationPolicy:
        config = self.augmentation_config
        if self.sequence_augmentation == "time_decay":
            return self._build_time_decay_policy()
        if self.sequence_augmentation == "session":
            return self._build_session_policy()
        if self.sequence_augmentation == "dataset_proportion":
            preset = config.get(
                "dataset_proportion_preset",
                "natural",
            )
            if preset == "natural":
                proportions = self._training_level_proportions()
            elif preset == "balanced":
                level_count = self.max_behavior_level + 1
                proportions = tuple(
                    1.0 / level_count
                    for _ in range(level_count)
                )
            else:
                raise ValueError(
                    f"Unsupported dataset proportion preset: {preset}."
                )
            return DatasetProportionPolicy(
                target_proportions=proportions,
                tolerance=config.get(
                    "dataset_proportion_tolerance",
                    1.2,
                ),
                min_history_items=config.get(
                    "augmentation_min_history_items",
                    1,
                ),
                preserve_target_level=not config.get(
                    "dataset_proportion_allow_target_level_drop",
                    False,
                ),
            )
        if self.sequence_augmentation == "user_adaptive_ratio":
            return UserAdaptiveRatioPolicy(
                global_proportions=self._training_level_proportions(),
                smoothing=config.get("user_adaptive_smoothing", 5.0),
                confidence_scale=config.get(
                    "user_adaptive_confidence_scale",
                    20.0,
                ),
                min_ratio=config.get("user_adaptive_min_ratio", 0.25),
                max_ratio=config.get("user_adaptive_max_ratio", 20.0),
                tolerance=config.get("user_adaptive_tolerance", 1.0),
                min_history_items=config.get(
                    "augmentation_min_history_items",
                    1,
                ),
                preserve_target_level=not config.get(
                    "user_adaptive_allow_target_level_drop",
                    False,
                ),
            )
        if self.sequence_augmentation == "target_conditioned":
            base_name = config.get(
                "target_conditioned_base_policy",
                "time_decay",
            )
            if base_name != "time_decay":
                raise ValueError(
                    "target_conditioned currently supports only "
                    "time_decay as its base policy."
                )
            return TargetConditionedPolicy(
                base_policy=self._build_time_decay_policy(),
                same_level_restore_probability=config.get(
                    "target_conditioned_same_level_restore",
                    0.8,
                ),
                precursor_restore_probability=config.get(
                    "target_conditioned_precursor_restore",
                    0.8,
                ),
                min_history_items=config.get(
                    "augmentation_min_history_items",
                    1,
                ),
            )
        if self.sequence_augmentation == "multi_view":
            return MultiViewAugmentationPolicy(
                recent_policy=self._build_time_decay_policy(),
                session_policy=self._build_session_policy(),
                min_history_items=config.get(
                    "augmentation_min_history_items",
                    1,
                ),
                include_recent=not config.get(
                    "multi_view_disable_recent",
                    False,
                ),
                include_hierarchy=not config.get(
                    "multi_view_disable_hierarchy",
                    False,
                ),
                include_session=not config.get(
                    "multi_view_disable_session",
                    False,
                ),
            )
        raise ValueError(
            "Unsupported sequence augmentation policy: "
            f"{self.sequence_augmentation}."
        )

    def _view_rng(
        self,
        uid: str,
        view_id: int,
    ) -> np.random.Generator:
        user_seed = zlib.crc32(uid.encode()) & 0xFFFFFFFF
        seed_sequence = np.random.SeedSequence([
            self.augmentation_seed,
            user_seed,
            view_id,
        ])
        return np.random.default_rng(seed_sequence)

    def _build_sample(
        self,
        history: BehaviorSequence,
        target_item: str,
        target_behavior: str,
        target_session_id: int,
        target_time: float,
    ) -> dict[str, str | list[int] | list[float]]:
        all_behaviors = history.behaviors + [target_behavior]
        all_session_ids = history.session_ids + [target_session_id]
        all_times = history.times + [target_time]
        return {
            "item": self.get_behavior_item(
                target_item,
                target_behavior,
            ),
            "inters": self._get_inters(
                history.items,
                history.behaviors,
            ),
            "session_ids": self._generate_session_ids(
                all_session_ids
            ),
            "extended_session_ids": (
                self._generate_extended_session_ids(
                    all_session_ids
                )
            ),
            "actions": self._generate_actions(all_behaviors),
            "time": self._generate_times(all_times),
            "behavior": target_behavior,
        }

    def _process_train_data(
        self,
    ) -> list[dict[str, str | list[int] | list[float]]]:
        if self.augmentation_policy is None:
            raise RuntimeError("Augmentation policy is not initialized.")
        inter_data = []
        view_counts: dict[str, int] = {}
        input_lengths = []
        output_lengths = []
        policy_view_count = 0
        unchanged_policy_views = 0
        input_level_counts = np.zeros(
            self.max_behavior_level + 1,
            dtype=int,
        )
        output_level_counts = np.zeros(
            self.max_behavior_level + 1,
            dtype=int,
        )
        input_time_bucket_counts = np.zeros(3, dtype=int)
        output_time_bucket_counts = np.zeros(3, dtype=int)
        for uid in get_tqdm(
            self.remapped_inters,
            desc="Processing policy-augmented training data",
        ):
            position = self.valid_pos[uid]
            if position <= 0:
                continue

            items = self.remapped_inters[uid][:position]
            behaviors = self.history_behaviors[uid][:position]
            session_ids = self.session[uid][:position]
            times = self.time[uid][:position]
            history = BehaviorSequence(
                items=items[:-1],
                behaviors=behaviors[:-1],
                session_ids=session_ids[:-1],
                times=times[:-1],
            )
            context = AugmentationContext(
                uid=uid,
                target_behavior=behaviors[-1],
                target_level=self.behavior_level[behaviors[-1]],
                target_time=times[-1],
                behavior_level=self.behavior_level,
                max_behavior_level=self.max_behavior_level,
            )
            seen_views: set[tuple[int, ...]] = set()
            if not self.augmentation_drop_original:
                original_indices = tuple(range(len(history.items)))
                seen_views.add(original_indices)
                inter_data.append(self._build_sample(
                    history,
                    items[-1],
                    behaviors[-1],
                    session_ids[-1],
                    times[-1],
                ))
                view_counts["original"] = (
                    view_counts.get("original", 0) + 1
                )
                input_lengths.append(len(history.items))
                output_lengths.append(len(history.items))

            for view_id in range(self.augmentation_views):
                views = self.augmentation_policy.generate_views(
                    history,
                    context,
                    self._view_rng(uid, view_id),
                )
                for view in views:
                    indices = tuple(view.keep_indices)
                    if any(
                        index < 0 or index >= len(history.items)
                        for index in indices
                    ):
                        raise ValueError(
                            f"Policy {view.name} returned an invalid history index."
                        )
                    if list(indices) != sorted(set(indices)):
                        raise ValueError(
                            f"Policy {view.name} must return sorted unique indices."
                        )
                    policy_view_count += 1
                    if len(indices) == len(history.items):
                        unchanged_policy_views += 1
                    age = np.maximum(
                        context.target_time
                        - np.asarray(history.times, dtype=float),
                        0.0,
                    )
                    time_buckets = np.digitize(age, bins=[48.0, 336.0])
                    for index, behavior in enumerate(history.behaviors):
                        level = self.behavior_level[behavior]
                        input_level_counts[level] += 1
                        input_time_bucket_counts[time_buckets[index]] += 1
                    for index in indices:
                        level = self.behavior_level[
                            history.behaviors[index]
                        ]
                        output_level_counts[level] += 1
                        output_time_bucket_counts[
                            time_buckets[index]
                        ] += 1
                    if indices in seen_views:
                        continue
                    seen_views.add(indices)
                    augmented_history = history.select(list(indices))
                    if not augmented_history.items:
                        continue
                    inter_data.append(self._build_sample(
                        augmented_history,
                        items[-1],
                        behaviors[-1],
                        session_ids[-1],
                        times[-1],
                    ))
                    view_counts[view.name] = (
                        view_counts.get(view.name, 0) + 1
                    )
                    input_lengths.append(len(history.items))
                    output_lengths.append(
                        len(augmented_history.items)
                    )

        mean_input = (
            float(np.mean(input_lengths))
            if input_lengths
            else 0.0
        )
        mean_output = (
            float(np.mean(output_lengths))
            if output_lengths
            else 0.0
        )
        logger.info(
            "Sequence augmentation summary: views={}, "
            "mean_input_length={:.2f}, mean_output_length={:.2f}, "
            "unchanged_policy_view_ratio={:.4f}, "
            "level_keep_rates={}, time_bucket_keep_rates={}.",
            view_counts,
            mean_input,
            mean_output,
            (
                unchanged_policy_views / policy_view_count
                if policy_view_count
                else 0.0
            ),
            np.divide(
                output_level_counts,
                input_level_counts,
                out=np.ones_like(output_level_counts, dtype=float),
                where=input_level_counts > 0,
            ).round(4).tolist(),
            np.divide(
                output_time_bucket_counts,
                input_time_bucket_counts,
                out=np.ones_like(
                    output_time_bucket_counts,
                    dtype=float,
                ),
                where=input_time_bucket_counts > 0,
            ).round(4).tolist(),
        )
        return inter_data
