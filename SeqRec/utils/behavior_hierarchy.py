"""Opt-in behavior identity/level separation for V3 datasets."""

import json
from pathlib import Path


def load_behavior_schema(root, dataset, levels):
    path = Path(root) / f'{dataset}.behavior_schema.json'
    schema = json.loads(path.read_text()) if path.exists() else {}
    if not isinstance(schema, dict):
        raise ValueError('Behavior schema must be a JSON object.')
    separate = schema.get('separate_behavior_identity_and_level', False)
    if not isinstance(separate, bool):
        raise ValueError('separate_behavior_identity_and_level must be boolean.')
    if separate:
        if not levels or any(isinstance(level, bool) or not isinstance(level, int) or level < 0 for level in levels.values()):
            raise ValueError('Separated behavior levels must be nonnegative integers.')
        if set(levels.values()) != set(range(max(levels.values()) + 1)):
            raise ValueError('Separated behavior levels must be contiguous and start at zero.')
    return schema


def configure_behavior_hierarchy(config, behavior_maps, levels):
    """Keep identity IDs; add hierarchy metadata only for explicitly enabled V3 inputs."""
    if not levels:
        return
    if len(behavior_maps) != len(levels):
        raise ValueError('V3 requires one behavior token per behavior identity.')
    config.separate_behavior_identity_and_level = True
    config.behavior_level_maps = {token: level for token, level in zip(behavior_maps, levels.values())}
    config.num_behavior_levels = max(levels.values()) + 1


def behavior_level_lookup(config):
    if not getattr(config, 'separate_behavior_identity_and_level', False):
        return None
    identities = {int(token): index for token, index in config.behavior_maps.items()}
    levels = {int(token): level for token, level in config.behavior_level_maps.items()}
    if set(identities) != set(levels) or set(identities.values()) != set(range(config.num_behavior)):
        raise ValueError('V3 behavior identity and hierarchy maps must align.')
    if any(isinstance(level, bool) or not isinstance(level, int) or not 0 <= level < config.num_behavior_levels for level in levels.values()):
        raise ValueError('V3 hierarchy levels must be integer indices within num_behavior_levels.')
    ranks = getattr(config, 'th_prior_behavior_ranks', None)
    if ranks is not None:
        if not isinstance(ranks, (list, tuple)) or len(ranks) != config.num_behavior:
            raise ValueError('th_prior_behavior_ranks must have one rank per behavior identity.')
        if any(isinstance(rank, bool) or not isinstance(rank, int) or not 0 <= rank < config.num_behavior_levels for rank in ranks):
            raise ValueError('V3 prior ranks must be integer indices within num_behavior_levels.')
    lookup = [0] * (config.num_behavior + 1)
    for token, identity in identities.items():
        lookup[identity + 1] = (levels[token] if ranks is None else ranks[identity]) + 1
    return lookup
