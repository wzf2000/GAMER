from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BehaviorLevelStatistics:
    level_counts: tuple[int, ...]
    level_proportions: tuple[float, ...]
    total_interactions: int
    user_count: int

    def cache_config(self) -> dict[str, int | tuple[int, ...] | tuple[float, ...]]:
        return {
            "level_counts": self.level_counts,
            "level_proportions": self.level_proportions,
            "total_interactions": self.total_interactions,
            "user_count": self.user_count,
        }


def compute_training_level_statistics(
    histories: dict[str, list[str]],
    valid_positions: dict[str, int],
    behavior_level: dict[str, int],
    max_behavior_level: int,
) -> BehaviorLevelStatistics:
    level_counts = np.zeros(max_behavior_level + 1, dtype=int)
    user_count = 0
    for uid, behaviors in histories.items():
        end = max(valid_positions[uid], 0)
        if end == 0:
            continue
        user_count += 1
        for behavior in behaviors[:end]:
            level_counts[behavior_level[behavior]] += 1

    total_interactions = int(level_counts.sum())
    if total_interactions:
        level_proportions = level_counts / total_interactions
    else:
        level_proportions = np.full(
            max_behavior_level + 1,
            1.0 / (max_behavior_level + 1),
        )
    return BehaviorLevelStatistics(
        level_counts=tuple(int(count) for count in level_counts),
        level_proportions=tuple(
            float(proportion)
            for proportion in level_proportions
        ),
        total_interactions=total_interactions,
        user_count=user_count,
    )
