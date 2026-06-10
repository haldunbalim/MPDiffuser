import numpy as np

from mpdiffuser.dataset.episode import episodes_from_transition_dataset
from mpdiffuser.dataset.seq_dataset import SequenceDataset


class D4RLDataset(SequenceDataset):
    def load_dataset(self):
        env = self.make_env()
        dataset = episodes_from_transition_dataset(env.get_dataset())
        self.set_dataset_dimensions(dataset)
        close = getattr(env, 'close', None)
        if close is not None:
            close()
        return dataset

    def make_env(self):
        import gym
        import d4rl

        try:
            import d4rl.gym_mujoco  # noqa: F401
        except ImportError:
            raise RuntimeError("d4rl is not installed, cannot create environment")
        return gym.make(self.dset_name)

    def get_env(self, num_envs=1):
        import gym
        import d4rl

        try:
            import d4rl.gym_mujoco  # noqa: F401
        except ImportError:
            raise RuntimeError("d4rl is not installed, cannot create environment")
        if num_envs > 1:
            from gym.vector import SyncVectorEnv
            return SyncVectorEnv([lambda name=self.dset_name: gym.make(name) for _ in range(num_envs)])
        return gym.make(self.dset_name)

    def get_normalized_scores(self, rewards):
        env = self.make_env()
        scores = env.get_normalized_score(np.asarray(rewards)) * 100
        close = getattr(env, 'close', None)
        if close is not None:
            close()
        return scores

    def get_conditions(self, batch, ep_idx, time_idx, ctx):
        # Condition on the normalised discounted return-to-go (scalar per sample).
        return batch['value_normalized'][:, None]
