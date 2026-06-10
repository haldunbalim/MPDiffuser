import os.path as op

import numpy as np

from mpdiffuser.dataset.episode import episodes_from_transition_dataset, make_episode
from mpdiffuser.dataset.seq_dataset import SequenceDataset


class HDFDataset(SequenceDataset):
    def __init__(self, *args, hdf_path=None, env_name=None, condition_fields=None, **kwargs):
        self.hdf_path = hdf_path
        self.env_name = env_name
        if isinstance(condition_fields, str):
            condition_fields = [condition_fields]
        else:
            condition_fields = list(condition_fields)
        condition_dim = kwargs.get('condition_dim')
        if condition_dim is not None and len(condition_fields) != int(condition_dim):
            raise ValueError(
                f"HDFDataset condition_fields has {len(condition_fields)} entries, "
                f"but condition_dim={condition_dim}"
            )
        self.condition_fields = condition_fields
        super().__init__(*args, **kwargs)

    def load_dataset(self):
        dset_name = self.hdf_path if self.hdf_path is not None else self.dset_name
        dataset = self.load_hdf_episodes(dset_name)
        self.set_dataset_dimensions(dataset)
        return dataset

    def load_hdf_episodes(self, dset_name):
        import h5py

        dset_path = self.resolve_hdf_path(dset_name)
        with h5py.File(dset_path, 'r') as f:
            if 'sample_0' in f:
                return self.read_hdf_episode_groups(f)
            raw_dataset = {key: f[key][()] for key in f.keys()}
            return episodes_from_transition_dataset(raw_dataset)

    @staticmethod
    def read_hdf_episode_groups(h5_file):
        dataset = []
        for sample_name in sorted(h5_file.keys(), key=sample_sort_key):
            grp = h5_file[sample_name]
            sample_data = {field: grp[field][()] for field in grp.keys()}
            dataset.append(make_episode(sample_data))
        return dataset

    @staticmethod
    def resolve_hdf_path(dset_name):
        roots = []
        if op.isabs(dset_name):
            roots.append(dset_name)
        elif op.sep in dset_name:
            roots.append(dset_name)
        else:
            roots.extend([
                op.join('data', dset_name),
                op.join('mpdiffuser', 'data', dset_name),
                dset_name,
            ])

        candidates = []
        for root in roots:
            candidates.append(root)
            if not root.endswith(('.h5', '.hdf5')):
                candidates.append(root + '.h5')

        for candidate in candidates:
            if op.exists(candidate):
                return candidate
        raise FileNotFoundError(
            f"Could not find HDF5 dataset for {dset_name!r}. Tried: {candidates}"
        )

    def get_env(self, num_envs=1):
        from gymnasium.vector import SyncVectorEnv
        import gymnasium as gym
        env_name = self.get_env_name()
        return SyncVectorEnv([lambda: gym.make(env_name) for _ in range(num_envs)])
    
    def get_normalized_scores(self, rewards):
        return rewards

    def get_env_name(self):
        if self.env_name is not None:
            return self.env_name
        name = op.basename(self.dset_name)
        for suffix in ('.hdf5', '.h5'):
            if name.endswith(suffix):
                name = name[:-len(suffix)]
        return name

    def get_conditions(self, batch, ep_idx, time_idx, ctx):
        conds = []
        for field in self.condition_fields:
            cond = batch[field]
            conds.append(cond[:, None] if cond.ndim == 1 else cond)
        return np.concatenate(conds, axis=-1)


def sample_sort_key(sample_name):
    try:
        return int(sample_name.split('_')[1])
    except (IndexError, ValueError):
        return sample_name
