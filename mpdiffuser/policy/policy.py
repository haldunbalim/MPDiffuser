
import jax 
import jax.numpy as jnp
from functools import partial
from jax.tree_util import tree_map

class Policy:
    nx: int 
    nu: int

    def load_state(self, *states):
        raise NotImplementedError('')
    
    def get_sample_fn(self, n_samples, return_best_only=False, return_path=False, *args, **kwargs):
        """ Returns a function that samples from the policy """
        n_devices = jax.local_device_count()
        policy_sample = jax.vmap(partial(self._sample, return_path=return_path, *args, **kwargs))

        def sample(obs, key, trans=None, cost_limit=None, return_scale=None, extra_conds=None):
            obs = jnp.atleast_2d(obs)
            keys = jax.random.split(key, n_samples)
            obss = jnp.repeat(obs, n_samples, axis=0)
            if trans is not None:
                # repeat the transition for each sample
                trans = jnp.repeat(trans, n_samples, axis=0)
            if cost_limit is not None:
                # repeat the cost limit for each sample
                if cost_limit.ndim == 1:
                    cost_limit = cost_limit[None]
                cost_limit = jnp.repeat(cost_limit, n_samples, axis=0)
            if return_scale is not None:
                # repeat the return scale for each sample
                if return_scale.ndim == 1:
                    return_scale = return_scale[None]
                return_scale = jnp.repeat(return_scale, n_samples, axis=0)
            if extra_conds is not None:
                # repeat the extra conditions for each sample
                if extra_conds.ndim == 1:
                    extra_conds = extra_conds[None]
                extra_conds = jnp.repeat(extra_conds, n_samples, axis=0)
            return policy_sample(obss, keys, x=trans, cost_limit=cost_limit, return_scale=return_scale, extra_conds=extra_conds)
        
        sample_vmap = jax.jit(jax.vmap(sample))
        sample_pmap = jax.pmap(sample_vmap)
    
        def sample_fn(x0, key, trans=None, return_scale=None, cost_limit=None, extra_conds=None):
            x0 = jnp.atleast_2d(x0)
            keys = jax.random.split(key, x0.shape[0])

            # sample
            if x0.shape[0] % n_devices != 0 or n_devices == 1:
                sampled = sample_vmap(
                    x0, keys, trans=trans, cost_limit=cost_limit, return_scale=return_scale, extra_conds=extra_conds)
                sampled = tree_map(lambda x: x[:, :, 0], sampled)
            else:
                # shard for multiple devices
                x0_sharded = x0.reshape(n_devices, -1, *x0.shape[1:])
                key_sharded = keys.reshape(n_devices, -1, *keys.shape[1:]) 
                if cost_limit is not None:
                    # shard the cost limit as well
                    if cost_limit.ndim == 1:
                        cost_limit = cost_limit[None]
                    cost_limit = cost_limit.reshape(n_devices, -1, 1)
                if return_scale is not None:
                    # shard the return scale as well
                    if return_scale.ndim == 1:
                        return_scale = return_scale[None]
                    return_scale = return_scale.reshape(n_devices, -1, 1)
                if extra_conds is not None:
                    # shard the extra conditions as well
                    if extra_conds.ndim == 1:
                        extra_conds = extra_conds[None]
                    extra_conds = extra_conds.reshape(n_devices, -1, extra_conds.shape[-1])
                if trans is not None:
                    # shard the trans as well
                    if trans.ndim == 2:
                        trans = trans[None]
                    trans = trans.reshape(n_devices, -1, *trans.shape[1:])
                    
                # sample on multiple devices
                sampled = sample_pmap(
                    x0_sharded, key_sharded, return_scale=return_scale, cost_limit=cost_limit, trans=trans, extra_conds=extra_conds)
                # flatten the device axis
                sampled = tree_map(lambda x: x.reshape(x0.shape[0], *x.shape[2:])[:, :, 0], sampled)
                if cost_limit is not None:
                    # flatten the cost limit as well
                    cost_limit = cost_limit.reshape(-1)

            if n_samples == 1:
                if return_best_only:
                    # pop sample dimension
                    return tree_map(lambda x: x[:, 0], sampled)
                return sampled
            
            if return_path:
                # these are not sorted!
                return sampled
            else:
                if not isinstance(sampled, tuple) or 'values' not in sampled[1]:
                    # no values
                    return sampled
                # filter by cost limit if specified
                if cost_limit is not None:
                    cost_values = sampled[1]['cost_values']
                    cost_mask = cost_values <= cost_limit[:, None]
                    if return_best_only:
                        # if return_best_only, we only keep the best sample that is within the cost limit
                        best_indices_filtered = jnp.argmax(jnp.where(cost_mask, sampled[1]['values'], -jnp.inf), axis=1)
                        # best_indices = jnp.argmax(sampled[1]['values'], axis=1)
                        min_cost_indices = jnp.argmin(cost_values, axis=1)
                        # if none of the samples are within the cost limit, we take the one with the minimum cost
                        indices = jnp.where(jnp.any(cost_mask, axis=1), best_indices_filtered, min_cost_indices)
                    else:
                        raise NotImplementedError('Filtering by cost limit is not implemented for return_best_only=False.')
                else:  
                    if return_best_only:
                        indices = jnp.argmax(sampled[1]['values'], axis=1)
                    else:
                        indices = jnp.argsort(sampled[1]['values'], axis=1, descending=True)
                return tree_map(lambda x: jax.vmap(lambda y, i: y[i])(x, indices), sampled)

        return sample_fn
    
    def get_sample_action_fn(self, n_samples, skip=1, *args, **kwargs):
        """ Returns a function that samples actions from the policy """
        sample_fn = self.get_sample_fn(n_samples, return_best_only=True, return_path=False, *args, **kwargs)
        idx = 0
        sampled = None

        def sample_action_fn(x0, key, return_scale=None, cost_limit=None, extra_conds=None):
            nonlocal idx, sampled
            if idx % skip == 0:
                sampled = sample_fn(
                    x0=x0, key=key, cost_limit=cost_limit, return_scale=return_scale, extra_conds=extra_conds)
                if isinstance(sampled, tuple):
                    sampled, _ = sampled
            # sampled is now (batch, H, nx+nu)
            u = sampled[:, idx, -self.nu:] 
            idx = (idx + 1) % skip
            return u

        return sample_action_fn

    def get_sample_fn_replan(self, n_samples, t_replan_step=10, *args, **kwargs):
        policy_sample_scratch = self.get_sample_fn(n_samples=n_samples, return_best_only=True, return_path=False, t_beg=0, *args, **kwargs)

        policy_sample_warm = self.get_sample_fn(
            n_samples=n_samples, return_path=False, t_beg=t_replan_step, return_best_only=True, *args, **kwargs)
        sample_warm_single = jax.jit(policy_sample_warm)

        last_trans = None

        def sample_fn(x0, key, return_scale=None, cost_limit=None, extra_conds=None, skip=1):
            nonlocal last_trans
            x0 = jnp.atleast_2d(x0)
            if last_trans is None:
                last_trans = policy_sample_scratch(x0, key, return_scale=return_scale, cost_limit=cost_limit, extra_conds=extra_conds)
                if isinstance(last_trans, tuple):
                    last_trans, _ = last_trans
            else:
                # shift the last transition to the next step
                last_trans = jnp.concatenate([last_trans[..., skip:, :],
                                              last_trans[..., -skip:, :]], axis=-2)
                last_trans = sample_warm_single(
                    x0, key, trans=last_trans[:, None], return_scale=return_scale, cost_limit=cost_limit, extra_conds=extra_conds)
                if isinstance(last_trans, tuple):
                    last_trans, _ = last_trans
            return last_trans

        return sample_fn

    def get_sample_action_fn_replan(self, n_samples, t_replan_step=10, skip=1, *args, **kwargs):
        sample_fn = self.get_sample_fn_replan(
            n_samples, t_replan_step=t_replan_step, *args, **kwargs)

        idx = 0
        sampled = None

        def sample_action_fn(x0, key, return_scale=None, cost_limit=None, extra_conds=None):
            nonlocal idx, sampled
            if idx % skip == 0:
                sampled = sample_fn(
                    x0=x0, key=key, return_scale=return_scale, cost_limit=cost_limit, skip=skip, extra_conds=extra_conds)
                if isinstance(sampled, tuple):
                    sampled, _ = sampled
            # sampled is now (batch, H, nx+nu)
            u = sampled[:, idx, -self.nu:]
            idx = (idx + 1) % skip
            return u
        return sample_action_fn
        

    def _sample(self, x0, key, x=None, return_path=False, *args, **kwargs):
        raise NotImplementedError('')
