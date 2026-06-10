from collections import namedtuple

import numpy as np


Episode = namedtuple(
    "Episode",
    ["observations", "actions", "rewards", "terminations", "truncations"],
)
CostEpisode = namedtuple(
    "CostEpisode",
    ["observations", "actions", "rewards", "costs", "terminations", "truncations"],
)


def episodes_from_transition_dataset(dataset):
    observations = np.asarray(dataset['observations'])
    actions = np.asarray(dataset['actions'])
    rewards = np.asarray(dataset['rewards']).reshape(-1)
    terminations = get_bool_transition_field(dataset, ['terminals', 'terminations'], len(actions))
    truncations = get_bool_transition_field(dataset, ['timeouts', 'truncations'], len(actions))
    next_observations = dataset.get('next_observations', None)
    if next_observations is not None:
        next_observations = np.asarray(next_observations)
    costs = dataset.get('costs', None)
    if costs is not None:
        costs = np.asarray(costs).reshape(-1)

    dones = np.logical_or(terminations, truncations)
    done_indices = np.argwhere(dones)[:, 0]
    if len(done_indices) == 0 and len(actions) > 0:
        done_indices = np.asarray([len(actions) - 1])
        truncations = truncations.copy()
        truncations[-1] = True

    boundaries = np.concatenate([[0], done_indices + 1], axis=0)
    episodes = []
    for b, e in zip(boundaries[:-1], boundaries[1:]):
        if e <= b:
            continue
        sample = {
            'observations': slice_observations(observations, next_observations, b, e),
            'actions': actions[b:e].copy(),
            'rewards': rewards[b:e].copy(),
            'terminations': terminations[b:e].copy(),
            'truncations': truncations[b:e].copy(),
        }
        if costs is not None:
            sample['costs'] = costs[b:e].copy()
        episodes.append(make_episode(sample))
    return episodes


def make_episode(sample_data):
    sample_data = dict(sample_data)
    if 'terminals' in sample_data and 'terminations' not in sample_data:
        sample_data['terminations'] = sample_data.pop('terminals')
    if 'timeouts' in sample_data and 'truncations' not in sample_data:
        sample_data['truncations'] = sample_data.pop('timeouts')
    if 'truncations' not in sample_data:
        sample_data['truncations'] = np.zeros_like(sample_data['rewards'], dtype=bool)

    kwargs = {
        'observations': np.asarray(sample_data['observations']).copy(),
        'actions': np.asarray(sample_data['actions']).copy(),
        'rewards': np.asarray(sample_data['rewards']).reshape(-1).copy(),
        'terminations': np.asarray(sample_data['terminations'], dtype=bool).reshape(-1).copy(),
        'truncations': np.asarray(sample_data['truncations'], dtype=bool).reshape(-1).copy(),
    }
    if 'costs' in sample_data:
        kwargs['costs'] = np.asarray(sample_data['costs']).reshape(-1).copy()
        return CostEpisode(**kwargs)
    return Episode(**kwargs)


def get_bool_transition_field(dataset, names, size):
    for name in names:
        if name in dataset:
            return np.asarray(dataset[name], dtype=bool).reshape(-1).copy()
    return np.zeros(size, dtype=bool)


def slice_observations(observations, next_observations, begin, end):
    if next_observations is not None:
        final_obs = next_observations[end - 1:end]
        return np.concatenate([observations[begin:end], final_obs], axis=0).copy()
    if len(observations) > end:
        return observations[begin:end + 1].copy()
    final_obs = observations[end - 1:end]
    return np.concatenate([observations[begin:end], final_obs], axis=0).copy()


def has_costs(episode):
    return isinstance(episode, tuple) and 'costs' in episode._fields
