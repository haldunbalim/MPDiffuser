import jax.numpy as jnp
import jax

from mpdiffuser import GaussianDiffusion
from mpdiffuser.utils import apply_cfg
from mpdiffuser.model import Diffuser, TrainState, RewardModel, CostModel
from .policy import Policy

class DDPolicy(Policy):
    def __init__(self, diff_model: Diffuser, inv_dyn_model, value_model: RewardModel = None, cost_model: CostModel = None):
        self.diff_model = diff_model
        self.value_model = value_model
        self.inv_dyn_model = inv_dyn_model
        self.cost_model = cost_model
        self.nx, self.nu, self.horizon = self.diff_model.nx, self.diff_model.nu, self.diff_model.horizon
        self.n_steps = self.diff_model.n_steps

    def load_state(self, diff_model_state: TrainState, inv_dyn_model_state: TrainState = None, value_model_state: TrainState = None, cost_model_state: TrainState = None):
        """ Load the state of the policy """
        self.diff_model_state = diff_model_state
        self.inv_dyn_model_state = inv_dyn_model_state
        self.value_model_state = value_model_state
        self.cost_model_state = cost_model_state

    @property
    def diffusion(self) -> GaussianDiffusion:
        return self.diff_model.diffusion

    def _sample(self, x0, key, x=None, t_beg=0, return_path=False, cfg_scale=1.2, return_scale=0.9, temperature=1.0, clip=True, cost_scale=0.1, cost_limit=None, extra_conds=None):
        assert not return_path, "DDPolicyFiLM does not support return_path=True"    
        key, r = jax.random.split(key)

        # prepare x0
        x0 = jnp.atleast_2d(x0)
        dim = self.nx
        
        if x is None:
            x = jax.random.normal(r, (x0.shape[0], self.horizon, dim)) * jnp.sqrt(temperature)
        else:
            if x.ndim == 2:
                x = x[None]
            if x.shape[-1] == self.nx + self.nu:
                x = x[..., :self.nx]
            x_noise = jax.random.normal(r, x.shape) * jnp.sqrt(temperature)
            _t = jnp.ones(x.shape[0], dtype=int) * (self.n_steps - t_beg - 1)
            x = jax.vmap(self.diffusion.q_sample)(x, _t, x_noise)
        
        conds = [return_scale]
        if cost_limit is not None:
            cost_limit = jnp.maximum(0.0, cost_limit)
            conds.append(cost_limit * cost_scale)
        if extra_conds is not None:
            conds.append(extra_conds)
        conds = jnp.concatenate([jnp.atleast_2d(c) for c in conds], axis=-1)

        def sample_step(x, t):
            def _sample_step(conds, x, t):
                return self.diff_model.apply(self.diff_model_state.variables, x=x, t=t, conds=conds)
            return apply_cfg(_sample_step, conds, cfg_scale, x, t)
        
        def condition_fn(xs):
            return xs.at[..., 0, :self.nx].set(x0)

        sampled = self.diffusion.p_sample_loop(
            x, sample_step, key=key, temperature=temperature, t_beg=t_beg, clip=clip, condition_fn=condition_fn)

        actions = self.inv_dyn_model.apply(self.inv_dyn_model_state.variables, sampled[..., :-1, :], sampled[..., 1:, :])
        actions = jnp.concatenate([actions, jnp.zeros_like(actions[..., :1, :])], axis=-2)  # pad last action
        sampled = jnp.concatenate([sampled, actions], axis=-1)

        if self.value_model is None:
            return sampled
        
        rews = self.value_model.apply(
            self.value_model_state.variables, x=x, u=actions)
        values = jnp.sum(rews * (0.997 ** jnp.arange(rews.shape[1])), axis=1)
        info = {'values': values}
        if self.cost_model is not None:
            cost_vals = self.cost_model.apply(
                self.cost_model_state.variables, x=x, u=actions)
            cost_vals = jnp.sum(
                cost_vals * (0.997 ** jnp.arange(cost_vals.shape[1])), axis=1)
            info['cost_values'] = cost_vals
        return sampled, info
