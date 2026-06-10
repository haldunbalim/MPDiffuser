import flax.linen as nn
import jax.numpy as jnp
import jax
from .networks import TemporalUnetFiLM, TrainState, BaseModel
from mpdiffuser import GaussianDiffusion
from functools import partial

class DynamicsModel(BaseModel):
    nx: int
    nu: int
    horizon: int

    unet_dim: int = 128
    unet_dim_mults: tuple = (1, 4, 8)
    attention: bool = False

    # film parameters
    x0_embed_dim: int = 128
    condition_dropout: float = 0.25
    # dimension of the condition vector (e.g., returns, costs)
    condition_dim: int = 1
    dtype: jnp.dtype = jnp.float32

    # diffusion parameters
    n_steps: int = 100
    predict_noise: bool = True

    def get_run_name(self) -> str:
        sfx = '-eps' if self.predict_noise else ''
        return f'cfg-h{self.horizon}-s{self.n_steps}{sfx}'

    def setup(self):
        self.network = TemporalUnetFiLM(dim=self.unet_dim, cond_embed_dim=self.x0_embed_dim,
                                        dim_mults=self.unet_dim_mults, out_dim=self.nx,
                                        attention=self.attention, input_transform_fn=input_transform_fn, 
                                        dtype=self.dtype)

    @property
    def diffusion(self) -> GaussianDiffusion:
        return GaussianDiffusion(self.n_steps, predict_noise=self.predict_noise)
    
    def get_batch_template(self):
        batch_template = (jnp.zeros((1, self.horizon, self.nx)),
                          jnp.zeros((1,), dtype=jnp.int32),
                          jnp.zeros((1, self.nx)),
                          jnp.zeros((1, self.horizon, self.nu)))
        if self.condition_dropout < 1.0:
            batch_template += (jnp.zeros((1, self.condition_dim)),)  # returns, costs, etc.
        return batch_template
    
    @nn.compact
    def __call__(self, x, t, x0, u, conds=None):
        if conds is None:
            return self.network(x, t, x0, u)
        else:
            return self.network(x, t, [x0, conds], u)

    def compute_outputs(self, input_dict: dict, key: jax.Array):
        # prepare input data
        x0 = input_dict['context_x'][:, 0]
        x = input_dict['target_x']
        u = input_dict['target_u']

        # randomly sample timesteps and noise
        k1, k2, k3, k4 = jax.random.split(key, 4)
        t = jax.random.randint(k1, (x.shape[0],), 0, self.n_steps)
        noise_x = jax.random.normal(k2, x.shape)
        noise_u = jax.random.normal(k3, u.shape)

        # q sample
        x_noise = jax.vmap(self.diffusion.q_sample)(x, t, noise_x)
        u_noise = jax.vmap(self.diffusion.q_sample)(u, t, noise_u)

        # compute model prediction
        if self.condition_dropout == 1.0:
            pred = self(x_noise, t, x0, u_noise)
        else:
            conds = input_dict['conds']
            mask = jax.random.bernoulli(k3, 1 - self.condition_dropout, conds.shape[0])
            conds = conds * mask[:, None]
            pred = self(x_noise, t, x0, u_noise, conds=conds)

        pred = jnp.asarray(pred, jnp.float32)  # for mixed precision training
        if self.predict_noise:
            mse = jnp.square(pred - noise_x)
        else:
            mse = jnp.square(pred - x)

        # compute losses / metrics
        loss = mse.mean()
        output_dict = {'optimized_loss': loss, 'loss_x': loss}
        return output_dict

    def val_step(self, state: TrainState, batch: dict, key: jax.Array, step: int):
        conds = None if self.condition_dropout == 1.0 else batch['conds']
                
        sampled = self.apply(state.ema_variables, x0=batch['context_x'][:, 0], us=batch['target_u'], 
                             method=self.sample, key=key, temperature=0.25, conds=conds)
        sampled = jnp.asarray(sampled, jnp.float32)
        return {'loss_x': jnp.square(batch['target_x'] - sampled).mean()}

    def sample(self, x0, us, key: jax.Array, temperature: float = 1.0, conds=None, cfg_scale: float = 1.0, clip: bool = True):
        key, r = jax.random.split(key)
        x0 = jnp.atleast_2d(x0)

        def sample_step(x, t):
            if conds is None:
                return self(x, t, x0=x0, u=us)
            else:
                cond = self(x, t, x0=x0, u=us, conds=conds)
                uncond = self(x, t, x0=x0, u=us, conds=jnp.zeros_like(conds))
                return uncond + cfg_scale * (cond - uncond)
            
        x = jax.random.normal(
            key, (x0.shape[0], self.horizon, self.nx)) * jnp.sqrt(temperature)
        return self.diffusion.p_sample_loop(x, sample_step, clip=clip, key=r, temperature=temperature)


def input_transform_fn(x, u):
    return jnp.concatenate([x, u], axis=-1)
