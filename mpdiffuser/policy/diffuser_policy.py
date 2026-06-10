import jax.numpy as jnp
import jax

from mpdiffuser import GaussianDiffusion, GuidedSampler
from mpdiffuser.model import Diffuser, ValueFunction, TrainState, RewardModel, CostModel
from .policy import Policy

class DiffuserPolicy(Policy):
    def __init__(self, diff_model: Diffuser, value_model: ValueFunction=None, reward_model: RewardModel=None, cost_model: CostModel=None):
        """ Initialize the Guided Policy with a diffusion model and optional value and cost models. """
        self.diff_model = diff_model
        self.value_model = value_model
        self.reward_model = reward_model
        self.cost_model = cost_model
        self.nx, self.nu, self.horizon = self.diff_model.nx, self.diff_model.nu, self.diff_model.horizon
        self.n_steps = self.diff_model.n_steps
        if self.value_model is not None and hasattr(self.value_model, 'n_steps'):
            assert self.diff_model.n_steps == self.value_model.n_steps, \
                f"Diffusion model and value model must have the same number of steps, got {self.diff_model.n_steps} and {self.value_model.n_steps}."

    def load_state(self, diff_model_state: TrainState, value_model_state: TrainState=None, reward_model: TrainState=None, cost_model_state: TrainState=None):
        self.diff_model_state = diff_model_state
        self.value_model_state = value_model_state
        self.reward_model_state = reward_model
        self.cost_model_state = cost_model_state

    @property
    def diffusion(self) -> GaussianDiffusion:
        return self.diff_model.diffusion

    def _sample(self, x0, key, return_path=False, scale=1e-4, n_guide_steps=2, t_stopgrad=0, temperature=1.0, cost_limit=None, clip=True, extra_conds=None, **kwargs):
        key, r = jax.random.split(key)

        # prepare x0
        x0 = jnp.atleast_2d(x0)
        dim = self.nx + self.nu
        x = jax.random.normal(
            r, (x0.shape[0], self.horizon, dim)) * jnp.sqrt(temperature)

        def guide(x, t):
            return self.value_model.apply(self.value_model_state.variables, x, t)
        
        def condition_fn(xus):
            xus = xus.at[..., 0, :x0.shape[-1]].set(x0)
            if extra_conds is not None:
                conds = jnp.repeat(extra_conds[None], xus.shape[0], axis=0)
                conds = jnp.repeat(conds[:, None], xus.shape[1], axis=1)
                d = extra_conds.shape[-1]
                xus = xus.at[..., self.nx-d:self.nx].set(conds)
            return xus
        
        def sample_step(x, t):
            return self.diff_model.apply(self.diff_model_state.variables, x, t)
        
        if scale is not None:
            sampler = GuidedSampler(sample_step, self.diffusion, scale, n_guide_steps,
                                    t_stopgrad, guide=guide, condition_fn=condition_fn)
            sample_step = sampler.sample_step
        sampled = self.diffusion.p_sample_loop(x, sample_step, 
                                                condition_fn=condition_fn, key=key, 
                                                return_path=return_path, 
                                                temperature=temperature,
                                                clip=clip)
        if scale is None and self.value_model is not None:
            vals = self.value_model.apply(self.value_model_state.variables, sampled)
            sampled = sampled, {'values': vals}
        
        if self.reward_model is None:
            return sampled
        
        x, u = jnp.split(sampled, [self.nx], axis=-1)
        rews = self.reward_model.apply(
            self.value_model_state.variables, x=x, u=u)
        values = jnp.sum(rews * (0.997 ** jnp.arange(rews.shape[1])), axis=1)
        info = {'values': values}
        if self.cost_model is not None:
            cost_vals = self.cost_model.apply(
                self.cost_model_state.variables, x=x, u=u)
            cost_vals = jnp.sum(
                cost_vals * (0.997 ** jnp.arange(cost_vals.shape[1])), axis=1)
            info['cost_values'] = cost_vals
        return sampled, info