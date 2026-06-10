import numpy as np

from mpdiffuser.dataset.episode import has_costs
from mpdiffuser.dataset.preprocess import get_normalizer


class SequenceDataset:
    def __init__(self, dset_name, *, condition_dim, horizon=32, num_ctx_x=1, num_ctx_u=0,
                 discount=0.997, termination_pen=100, pad=True, finite_horizon=False,
                 normalize_obs=True, obs_normalizer_typ='cdf', normalize_actions=True, act_normalizer_typ='cdf',
                 normalize_values=True, skip_sequencing=False):
        if condition_dim is None:
            raise ValueError("condition_dim is required for every dataset")
        self.condition_dim = int(condition_dim)
        if self.condition_dim <= 0:
            raise ValueError(f"condition_dim must be positive, got {condition_dim!r}")
        self.horizon = horizon
        self.discount = discount
        self.num_ctx_x = num_ctx_x
        self.num_ctx_u = num_ctx_u
        self.termination_pen = termination_pen
        self.dset_name = dset_name
        self.pad = pad
        self.finite_horizon = finite_horizon
        self.normalize_obs = normalize_obs
        self.obs_normalizer_typ = obs_normalizer_typ
        self.normalize_actions = normalize_actions
        self.act_normalizer_typ = act_normalizer_typ
        self.normalize_values = normalize_values
        if skip_sequencing:
            if self.normalize_actions or self.normalize_obs:
                dataset = self.load_dataset()
                self.initialize_normalizers(dataset)
        else:
            self.sequence_dataset()

    def __len__(self):
        return self.indices[-1]

    def __getitem__(self, idx):
        ctx = max(self.num_ctx_x, self.num_ctx_u)
        ep_idx, time_idx = self.get_idx(idx)
        ep_idx, time_idx = np.atleast_1d(ep_idx), np.atleast_1d(time_idx)

        obs_offset = np.arange(ctx - self.num_ctx_x, self.horizon + ctx - self.num_ctx_x + 1)
        observations = self.observations[ep_idx[:, None], time_idx[:, None] + obs_offset]
        act_offset = np.arange(ctx - self.num_ctx_u - 1, self.horizon + ctx - self.num_ctx_u - 1)
        actions = self.actions[ep_idx[:, None], time_idx[:, None] + act_offset]
        values = self.values[ep_idx, time_idx + ctx - 1]
        reward = self.rewards[ep_idx, time_idx + ctx - 1]
        termination = self.terminations[ep_idx, time_idx + ctx - 1]
        values_normalized = (values / self.max_val) if self.normalize_values else values
        dct = {
            'context_x': observations[:, :self.num_ctx_x],
            'context_u': actions[:, :self.num_ctx_u],
            'target_x': observations[:, self.num_ctx_x:self.horizon + self.num_ctx_x],
            'target_u': actions[:, self.num_ctx_u:self.horizon + self.num_ctx_u],
            'value': values,
            'reward': reward,
            'value_normalized': values_normalized,
            'termination': termination,
        }
        if self.cost_values is not None:
            cost_values = self.cost_values[ep_idx, time_idx + ctx - 1]
            dct['cost_value'] = cost_values
            dct['cost'] = self.costs_all[ep_idx, time_idx + ctx - 1]
        conds = np.asarray(self.get_conditions(dct, ep_idx, time_idx, ctx))
        if conds.ndim == 1:
            conds = conds[:, None]
        if conds.shape[-1] != self.condition_dim:
            raise ValueError(
                f"{type(self).__name__} returned condition dim {conds.shape[-1]}, "
                f"but condition_dim={self.condition_dim}"
            )
        dct['conds'] = conds
        if dct['context_x'].shape[0] == 1:
            dct = {k: v[0] for k, v in dct.items()}  # Remove batch dimension
        return dct

    def load_dataset(self):
        raise NotImplementedError(
            f"{type(self).__name__} must load raw source data and return episode objects"
        )

    def set_dataset_dimensions(self, dataset):
        if len(dataset) == 0:
            raise ValueError(f"Dataset {self.dset_name!r} did not contain any complete episodes")
        self.nx = dataset[0].observations.shape[-1]
        self.nu = dataset[0].actions.shape[-1]

    def sequence_dataset(self):
        dataset = self.load_dataset()
        self.set_dataset_dimensions(dataset)
        ctx = max(self.num_ctx_x, self.num_ctx_u)
        max_len = max([ep.actions.shape[0] for ep in dataset])

        if (self.normalize_actions and not hasattr(self, 'u_normalizer')) or (self.normalize_obs and not hasattr(self, 'x_normalizer')):
            self.initialize_normalizers(dataset)

        dataset_has_costs = has_costs(dataset[0])
        lds = len(dataset)
        observations_all = np.zeros((lds, max_len + 1, dataset[0].observations.shape[-1]))
        actions_all = np.zeros((lds, max_len, dataset[0].actions.shape[-1]))
        values_all = np.zeros((lds, max_len))
        rewards_all = np.zeros((lds, max_len))
        terminations_all = np.zeros((lds, max_len), dtype=bool)
        costs_all = np.zeros((lds, max_len))
        cost_values_all = np.zeros((lds, max_len)) if dataset_has_costs else None
        lens_all = np.zeros((lds,), dtype=int)
        indices_all = []

        ep_idx = 0
        for episode in dataset:
            observations = episode.observations.copy()
            actions = episode.actions.copy()
            rewards = episode.rewards.copy()
            terminations = episode.terminations.copy()
            if dataset_has_costs:
                costs = episode.costs.copy()

            if episode.terminations[-1]:
                rewards[-1] -= self.termination_pen

            if self.pad:
                pad_len = min(self.horizon - 1, max_len - len(actions))
                observations = np.concatenate(
                    [observations, np.zeros((pad_len, observations.shape[1]))], axis=0)
                actions = np.concatenate(
                    [actions, np.zeros((pad_len, actions.shape[1]))], axis=0)
                rewards = np.concatenate(
                    [rewards, np.zeros((pad_len,))], axis=0)
                terminations = np.concatenate(
                    [terminations, np.zeros((pad_len,), dtype=bool)], axis=0)
                if dataset_has_costs:
                    costs = np.concatenate([costs, np.zeros((pad_len,))], axis=0)

            if self.normalize_obs:
                observations = self.x_normalizer.normalize(observations)
            if self.normalize_actions:
                actions = self.u_normalizer.normalize(actions)

            indices_all.append(list(range(ctx, len(observations) - self.horizon + 1)))
            observations_all[ep_idx, :len(observations), :] = observations
            actions_all[ep_idx, :len(actions), :] = actions
            rewards_all[ep_idx, :len(rewards)] = rewards
            terminations_all[ep_idx, :len(rewards)] = terminations
            lens_all[ep_idx] = len(actions) - (pad_len if self.pad else 0)
            if self.finite_horizon:
                values_all[ep_idx, :len(actions)] = finite_horizon_discounted_values(rewards, self.discount, self.horizon)
            else:
                values_all[ep_idx, :len(actions)] = discounted_cumsum(rewards, self.discount)
            if dataset_has_costs:
                cost_values_all[ep_idx, :len(actions)] = finite_horizon_discounted_values(costs, self.discount, self.horizon)
                costs_all[ep_idx, :len(rewards)] = costs
            ep_idx += 1

        self.observations = observations_all
        self.costs_all = costs_all if dataset_has_costs else None
        self.actions = actions_all
        self.rewards = rewards_all
        self.terminations = terminations_all
        self.values = values_all
        self.lens_all = lens_all
        self.indices = np.append(0, np.cumsum([len(idx) for idx in indices_all]))
        self.cost_values = cost_values_all if cost_values_all is not None else None
        self.max_val = np.max(values_all[:, 0])
        if self.max_val < 0:
            self.max_val = np.min(values_all[:, 0])
        return dataset

    def initialize_normalizers(self, dataset):
        obs_all, acts_all = [], []
        for episode in dataset:
            obs_all.append(episode.observations)
            acts_all.append(episode.actions)
        if len(obs_all) == 0:
            raise ValueError(
                f"Dataset {self.dset_name!r} has no episodes available for normalizer initialization"
            )
        obs_all = np.concatenate(obs_all)
        acts_all = np.concatenate(acts_all)
        if self.normalize_obs:
            self.x_normalizer = get_normalizer(self.obs_normalizer_typ)(obs_all)
        if self.normalize_actions:
            self.u_normalizer = get_normalizer(self.act_normalizer_typ)(acts_all)

    def get_env(self, num_envs=1):
        raise NotImplementedError(f"{type(self).__name__} does not define an environment")

    def get_normalized_scores(self, rewards):
        return rewards

    def get_conditions(self, batch, ep_idx, time_idx, ctx):
        raise NotImplementedError(
            f"{type(self).__name__} must decide how to build condition vectors"
        )

    def get_idx(self, queries):
        """
        For each query, return the largest element in sorted_array <= query.

        Args:
            sorted_array (np.ndarray): 1D increasing array
            queries (np.ndarray): 1D array of query values

        Returns:
            np.ndarray: 1D array of maximum lower bounds (same shape as queries)
        """
        indices = np.searchsorted(self.indices, queries, side='right') - 1
        indices = np.clip(indices, 0, len(self.indices) - 1)
        return indices, queries - self.indices[indices]


def discounted_cumsum(rewards, discount):
    t = len(rewards)
    discounts = discount ** np.arange(t)
    discounted_rewards = rewards * discounts
    cumsum = np.cumsum(discounted_rewards[::-1])[::-1]
    return cumsum / discounts


def finite_horizon_discounted_values(rewards, discount, horizon):
    T = len(rewards)
    discounts = discount ** np.arange(horizon)
    padded_rewards = np.pad(rewards, (0, horizon - 1), constant_values=0)
    strided = np.lib.stride_tricks.sliding_window_view(
        padded_rewards, window_shape=horizon)
    values = strided[:T] @ discounts
    return values