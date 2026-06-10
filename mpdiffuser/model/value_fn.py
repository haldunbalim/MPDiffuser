import flax.linen as nn
import jax.numpy as jnp
import jax
from .networks import Resnet1D, BaseModel
from mpdiffuser import GaussianDiffusion

class ValueFunction(BaseModel):
    nx: int
    nu: int
    horizon: int
    resnet_dim: int = 32
    resnet_dim_mults: tuple = (1, 2, 4, 8)

    # diffusion parameters
    n_steps: int = 20

    def get_run_name(self) -> str:
        return f'value-h{self.horizon}'

    def setup(self):
        self.network = Resnet1D(dim=self.resnet_dim, dim_mults=self.resnet_dim_mults)

    @property
    def diffusion(self) -> GaussianDiffusion:
        return GaussianDiffusion(self.n_steps, False)
    
    def get_batch_template(self):
        batch_template = (jnp.zeros((1, self.horizon, self.nx + self.nu)), 
                          jnp.zeros((1,), dtype=jnp.int32))
        return batch_template

    @nn.compact
    def __call__(self, trans, t):
        return self.network(trans, t)

    def compute_outputs(self, input_dict: dict, key: jax.Array):
        # prepare input data
        x = jnp.concatenate(
            [input_dict['context_x'], input_dict['target_x'][:, :-1]], axis=1)
        trans = jnp.concatenate([x, input_dict['target_u']], axis=-1)

        # randomly sample timesteps and noise
        k1, k2 = jax.random.split(key, 2)
        t = jax.random.randint(k1, (trans.shape[0],), 0, self.n_steps)
        noise = jax.random.normal(k2, trans.shape)

        # q sample
        x_noise = jax.vmap(self.diffusion.q_sample)(trans, t, noise)

        # pred
        x0 = trans[:, 0, :self.nx]
        x_noise = x_noise.at[:, 0, :self.nx].set(x0) 
        pred = self(x_noise, t)

        # compute loss
        return {'optimized_loss': jnp.square(pred - input_dict['value']).mean()}
