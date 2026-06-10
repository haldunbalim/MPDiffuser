import flax.linen as nn
import jax.numpy as jnp
import jax
from .networks import TemporalUnetFiLM, TrainState, BaseModel
from mpdiffuser import GaussianDiffusion
from functools import partial

class Planner(BaseModel):
    nx: int
    nu: int
    horizon: int

    unet_dim: int = 128
    unet_dim_mults: tuple = (1, 4, 8)
    action_loss_wgt: float = 10.0
    attention: bool = False
    
    condition_dropout: float = 0.25
    condition_dim: int = 1 # dimension of the condition vector (e.g., returns, costs)

    # film parameters
    x0_embed_dim: int = 128

    # diffusion parameters
    n_steps: int = 100
    predict_noise: bool = True 
    dtype: jnp.dtype = jnp.float32

    def get_run_name(self) -> str:
        sfx = '-eps' if self.predict_noise else ''
        return f'cfg-h{self.horizon}-s{self.n_steps}{sfx}'

    def setup(self):
        self.network = TemporalUnetFiLM(dim=self.unet_dim, cond_embed_dim=self.x0_embed_dim,
                                        dim_mults=self.unet_dim_mults, out_dim=self.nx + self.nu,
                                        attention=self.attention, dtype=self.dtype)
    @property
    def diffusion(self) -> GaussianDiffusion:
        return GaussianDiffusion(self.n_steps, predict_noise=self.predict_noise)

    def get_batch_template(self):
        batch_template = (jnp.zeros((1, self.horizon, self.nx + self.nu)),
                          jnp.zeros((1,), dtype=jnp.int32),
                          jnp.zeros((1, self.nx)))
        if self.condition_dropout < 1.0:
            # returns, costs
            batch_template += (jnp.zeros((1, self.condition_dim)),)
        return batch_template
    
    @nn.compact
    def __call__(self, trans, t, x0, conds=None):
        if conds is None:
            return self.network(trans, t, x0)
        else:
            return self.network(trans, t, [x0, conds])

    def compute_outputs(self, input_dict: dict, key: jax.Array):
        # prepare input data
        trans = jnp.concatenate([input_dict['target_x'], input_dict['target_u']], axis=-1)
        x0 = input_dict['context_x'][:, 0]

        # randomly sample timesteps and noise
        k1, k2, k3 = jax.random.split(key, 3)
        t = jax.random.randint(k1, (trans.shape[0],), 0, self.n_steps)
        noise = jax.random.normal(k2, trans.shape)

        # q sample
        trans_noise = jax.vmap(self.diffusion.q_sample)(trans, t, noise)

        # compute model prediction
        if self.condition_dropout == 1.0:
            pred = self(trans_noise, t, x0)
        else:
            conds = input_dict['conds']
            mask = jax.random.bernoulli(k3, 1 - self.condition_dropout, conds.shape[0])
            conds = conds * mask[:, None]
            pred = self(trans_noise, t, x0, conds=conds)

        pred = jnp.asarray(pred, jnp.float32)  # for mixed precision training
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
        trans = jnp.concatenate(
            [batch['target_x'], batch['target_u']], axis=-1)
        x0 = batch['context_x'][:, 0]
        
        conds = None if self.condition_dropout == 1.0 else batch['conds']
                
        sampled = self.apply(state.ema_variables, x0=x0,
                             method=self.sample, key=key, 
                             conds=conds, temperature=0.25)
        sampled = jnp.asarray(sampled, jnp.float32)  # for mixed precision training

        mse = jnp.square(trans - sampled)
        mse = mse.mean(axis=0)
        output_dict = {
            'loss_x': mse[:, :self.nx].mean(),
            'loss_u': mse[:, self.nx:].mean(),
            'loss_u_0': mse[0, self.nx:].mean(),
            'loss': mse.mean()
        }
        return output_dict

    def sample(self, x0, key: jax.Array, temperature: float = 1.0, conds=None, cfg_scale=1.0, clip: bool = True):
        ''' Sample from p(x) '''
        key, r = jax.random.split(key)
        x0 = jnp.atleast_2d(x0)

        def sample_step(x, t):
            if conds is None:
                return self.sample_step(x, t, x0)
            else:
                cond = self.sample_step(x, t, x0, conds=conds)
                uncond = self.sample_step(x, t, x0, conds=jnp.zeros_like(conds))
                return uncond + cfg_scale * (cond - uncond)
            
        dim = self.nx + self.nu
        x = jax.random.normal(
            key, (x0.shape[0], self.horizon, dim)) * jnp.sqrt(temperature)
        trans = self.diffusion.p_sample_loop(
            x, sample_step, key=r, temperature=temperature, clip=clip)
        return trans
