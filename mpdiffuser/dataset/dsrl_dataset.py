import numpy as np

from mpdiffuser.dataset.episode import episodes_from_transition_dataset
from mpdiffuser.dataset.seq_dataset import SequenceDataset


class DSRLDataset(SequenceDataset):
    def load_dataset(self):
        import dsrl  # noqa: F401

        errors = []
        for make_fn in self.env_makers(vector=False):
            env = None
            try:
                env = make_fn(self.dset_name)
                if not hasattr(env, 'get_dataset'):
                    raise AttributeError(f"{type(env).__name__} does not expose get_dataset()")
                dataset = episodes_from_transition_dataset(env.get_dataset())
                self.set_dataset_dimensions(dataset)
                return dataset
            except Exception as exc:  # pragma: no cover - depends on optional env registration
                errors.append(exc)
            finally:
                if env is not None:
                    close = getattr(env, 'close', None)
                    if close is not None:
                        close()
        raise RuntimeError(f"Could not load DSRL dataset {self.dset_name!r}: {errors}")

    def make_env(self):
        import dsrl  # noqa: F401

        errors = []
        for make_fn in self.env_makers(vector=False):
            try:
                return make_fn(self.dset_name)
            except Exception as exc:  # pragma: no cover - depends on optional env registration
                errors.append(exc)
        raise RuntimeError(f"Could not create DSRL environment {self.dset_name!r}: {errors}")

    @staticmethod
    def env_makers(vector=False, num_envs=1):
        makers = []
        try:
            import safety_gymnasium

            if vector:
                makers.append(
                    lambda name: safety_gymnasium.vector.make(
                        name, num_envs=num_envs, asynchronous=False
                    )
                )
            else:
                makers.append(lambda name: safety_gymnasium.make(name))
        except ImportError:
            pass
        try:
            import gymnasium

            if vector:
                from gymnasium.vector import SyncVectorEnv

                makers.append(
                    lambda name: SyncVectorEnv(
                        [lambda env_name=name: gymnasium.make(env_name) for _ in range(num_envs)]
                    )
                )
            else:
                makers.append(lambda name: gymnasium.make(name))
        except ImportError:
            pass
        try:
            import gym

            if vector:
                from gym.vector import SyncVectorEnv

                makers.append(
                    lambda name: SyncVectorEnv(
                        [lambda env_name=name: gym.make(env_name) for _ in range(num_envs)]
                    )
                )
            else:
                makers.append(lambda name: gym.make(name))
        except ImportError:
            pass
        return makers

    def get_env(self, num_envs=1):
        import dsrl  # noqa: F401

        errors = []
        for make_fn in self.env_makers(vector=num_envs > 1, num_envs=num_envs):
            try:
                return make_fn(self.dset_name)
            except Exception as exc:  # pragma: no cover - depends on optional env registration
                errors.append(exc)
        raise RuntimeError(f"Could not create DSRL environment {self.dset_name!r}: {errors}")

    def get_normalized_scores(self, rewards):
        import dsrl  # noqa: F401

        errors = []
        for make_fn in self.env_makers(vector=False):
            env = None
            try:
                env = make_fn(self.dset_name)
                mn, mx = env.min_episode_reward, env.max_episode_reward
                return (np.asarray(rewards) - mn) / (mx - mn) * 100
            except Exception as exc:  # pragma: no cover - depends on optional env registration
                errors.append(exc)
            finally:
                if env is not None:
                    close = getattr(env, 'close', None)
                    if close is not None:
                        close()
        raise RuntimeError(f"Could not normalize DSRL scores for {self.dset_name!r}: {errors}")

    def get_conditions(self, batch, ep_idx, time_idx, ctx):
        # Condition on [normalised discounted RTG, finite-horizon cost-to-go].
        if 'cost_value' not in batch:
            raise ValueError(f"DSRL dataset {self.dset_name!r} is missing cost values for conditioning")
        returns = batch['value_normalized'][:, None]
        costs = batch['cost_value'][:, None]
        return np.concatenate([returns, costs], axis=-1)
