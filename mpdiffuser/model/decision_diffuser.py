import flax.linen as nn
import jax.numpy as jnp
import jax
from .networks import TemporalUnet, TrainState, BaseModel
from mpdiffuser import GaussianDiffusion


class DecisionDiffuser(BaseModel):
    nx: int
    nu: int
    horizon: int

    cond_embed_dim: int = 128
    condition_dropout: float = 0.25
    condition_dim: int = 1  # dimension of the condition vector (e.g., returns, costs)

    unet_dim: int = 128
    unet_dim_mults: tuple = (1, 4, 8)
    attention: bool = False
    predict_noise: bool = True

    # diffusion parameters
    n_steps: int = 100
    dtype: jnp.dtype = jnp.float32

    def get_run_name(self) -> str:
        sfx = '-eps' if self.predict_noise else ''
        return f'cfg-h{self.horizon}-s{self.n_steps}{sfx}'

    def setup(self):
        self.network = TemporalUnet(dim=self.unet_dim, cond_embed_dim=self.cond_embed_dim,
                                    dim_mults=self.unet_dim_mults, out_dim=self.nx,
                                    attention=self.attention, dtype=self.dtype)
        
    @property
    def diffusion(self) -> GaussianDiffusion:
        return GaussianDiffusion(self.n_steps, predict_noise=self.predict_noise)
    
    def get_batch_template(self):
        batch_template = (jnp.zeros((1, self.horizon, self.nx)),
                          jnp.zeros((1,), dtype=jnp.int32))
        if self.condition_dropout < 1.0:
            # returns
            batch_template += (jnp.zeros((1, self.condition_dim)),)
        return batch_template
    
    @nn.compact
    def __call__(self, x, t, conds=None):
        if conds is None:
            return self.network(x, t)
        else:
            return self.network(x, t, [conds])

    def compute_outputs(self, input_dict: dict, key: jax.Array):
        # prepare input data
        x0 = input_dict['context_x'][:, 0]
        x = jnp.concatenate([x0[:, None], input_dict['target_x'][:, :-1]], axis=1)

        # randomly sample timesteps and noise
        k1, k2, k3 = jax.random.split(key, 3)
        t = jax.random.randint(k1, (x.shape[0],), 0, self.n_steps)
        noise = jax.random.normal(k2, x.shape)

        # q sample
        x_noise = jax.vmap(self.diffusion.q_sample)(x, t, noise)
        x_noise = x_noise.at[:, 0].set(x0)
        
        # pred
        if self.condition_dropout == 1.0:
            pred = self(x_noise, t)
        else:
            conds = input_dict['conds']
            mask = jax.random.bernoulli(k3, 1 - self.condition_dropout, conds.shape[0])
            conds = conds * mask[:, None]
            pred = self(x_noise, t, conds=conds)
        pred = pred.at[:, 0].set(x0)

        # compute losses / metrics
        if self.predict_noise:
            mse = jnp.square(pred - noise)
        else:
            mse = jnp.square(pred - x)
        return {'optimized_loss': mse.mean()}

    def val_step(self, state: TrainState, batch: dict, key: jax.Array, step: int):
        x = jnp.concatenate([batch['context_x'], batch['target_x'][:, :-1]], axis=1)
        x0 = batch['context_x'][:, 0]

        conds = None if self.condition_dropout == 1.0 else batch['conds']
        sampled = self.apply(state.ema_variables, x0=x0, method=self.sample, key=key, temperature=0.25, conds=conds)

        mse = jnp.square(x - sampled)
        mse = mse.mean(axis=0)
        return {'loss': mse.mean()}

    def sample(self, x0, key: jax.Array, temperature: float = 1.0, conds=None, cfg_scale=1.0, clip: bool = True):
        ''' Sample from p(x) '''
        key, r = jax.random.split(key)
        x = jax.random.normal(r, (x0.shape[0], self.horizon, self.nx)) * jnp.sqrt(temperature)

        def sample_step(x, t):
            if conds is None:
                return self(x, t)
            else:
                cond = self(x, t, conds=conds)
                uncond = self(x, t, conds=jnp.zeros_like(conds))
                return uncond + cfg_scale * (cond - uncond)

        def condition_fn(x):
            return x.at[:, 0].set(x0)

        trans = self.diffusion.p_sample_loop(
            x, sample_step, condition_fn=condition_fn, key=key, temperature=temperature, clip=clip)
        return trans
