import flax.linen as nn
import jax.numpy as jnp
import jax
from .networks import TemporalUnet, TrainState, BaseModel
from mpdiffuser import GaussianDiffusion


class Diffuser(BaseModel):
    nx: int
    nu: int
    horizon: int

    unet_dim: int = 128
    unet_dim_mults: tuple = (1, 4, 8)
    action_loss_wgt: float = 10.0
    attention: bool = False
    predict_noise: bool = True

    # diffusion parameters
    n_steps: int = 100
    dtype: jnp.dtype = jnp.float32

    def get_run_name(self) -> str:
        sfx = '-eps' if self.predict_noise else ''
        return f'cfg-h{self.horizon}-s{self.n_steps}{sfx}'

    def setup(self):
        self.network = TemporalUnet(
            dim=self.unet_dim, dim_mults=self.unet_dim_mults, out_dim=self.nx + self.nu, attention=self.attention, dtype=self.dtype)
        
    @property
    def diffusion(self) -> GaussianDiffusion:
        return GaussianDiffusion(self.n_steps, predict_noise=self.predict_noise)
    
    def get_batch_template(self):
        batch_template = (jnp.zeros((1, self.horizon, self.nx + self.nu)),
                          jnp.zeros((1,), dtype=jnp.int32))
        return batch_template
    
    @nn.compact
    def __call__(self, trans, t):
        return self.network(trans, t)

    def compute_outputs(self, input_dict: dict, key: jax.Array):
        # prepare input data
        x = jnp.concatenate([input_dict['context_x'], input_dict['target_x'][:, :-1]], axis=1)
        trans = jnp.concatenate([x, input_dict['target_u']], axis=-1)

        # randomly sample timesteps and noise
        key, k1, k2 = jax.random.split(key, 3)
        t = jax.random.randint(k1, (trans.shape[0],), 0, self.n_steps)
        noise = jax.random.normal(k2, trans.shape)

        # q sample
        x_noise = jax.vmap(self.diffusion.q_sample)(trans, t, noise)
        x0 = trans[:, 0, :self.nx]
        x_noise = x_noise.at[:, 0, :self.nx].set(x0)
        
        # pred
        pred = self(x_noise, t)
        pred = pred.at[:, 0, :self.nx].set(x0)

        # compute losses / metrics
        if self.predict_noise:
            mse = jnp.square(pred - noise)
        else:
            mse = jnp.square(pred - trans)
        output_dict = {}
        mse = mse.mean(axis=0)
        output_dict['loss_x'] = mse[:, :self.nx].mean()
        output_dict['loss_u'] = mse[:, self.nx:].mean()
        output_dict['loss_u_0'] = mse[0, self.nx:].mean()
        output_dict['loss'] = mse.mean()
        # weight first action loss
        mse = mse.at[..., 0, self.nx:].set(
            mse[..., 0, self.nx:] * self.action_loss_wgt)
        output_dict['optimized_loss'] = mse.mean()

        return output_dict

    def val_step(self, state: TrainState, batch: dict, key: jax.Array, step: int):
        x = jnp.concatenate(
            [batch['context_x'], batch['target_x'][:, :-1]], axis=1)
        trans = jnp.concatenate([x, batch['target_u']], axis=-1)

        x0 = batch['context_x'][:, 0]
        sampled = self.apply(state.ema_variables, x0=x0, method=self.sample, key=key, temperature=0.25)

        mse = jnp.square(trans - sampled)
        mse = mse.mean(axis=0)
        output_dict = {
            'loss_x': mse[:, :self.nx].mean(),
            'loss_u': mse[:, self.nx:].mean(),
            'loss_u_0': mse[0, self.nx:].mean(),
            'loss': mse.mean()
        }
        return output_dict

    def sample(self, x0, key: jax.Array, temperature: float = 1.0, clip: bool = True):
        ''' Sample from p(x) '''
        key, r = jax.random.split(key)

        dim = self.nx + self.nu
        x = jax.random.normal(
            r, (x0.shape[0], self.horizon, dim)) * jnp.sqrt(temperature)

        def condition_fn(x):
            return x.at[:, 0, :self.nx].set(x0)

        trans = self.diffusion.p_sample_loop(
            x, self, condition_fn=condition_fn, key=key, temperature=temperature, clip=clip)
        return trans
