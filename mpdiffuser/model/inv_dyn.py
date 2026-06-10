import flax.linen as nn
import jax.numpy as jnp
import jax
from typing import Tuple
from .networks import MLP, BaseModel


class InvDynModel(BaseModel):
    nx: int
    nu: int
    horizon: int

    hidden_dims: tuple[int] = (256, 256)
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.network = MLP((*self.hidden_dims, 256), act_out=None, dtype=self.dtype)
    
    def get_batch_template(self):
        batch_template = (jnp.zeros((1, self.nx)),
                          jnp.zeros((1, self.nx)))
        return batch_template
    
    @nn.compact
    def __call__(self, x, xp):
        inp = jnp.concatenate([x, xp], axis=-1)
        return self.network(inp)

    def compute_outputs(self, input_dict: dict, key: jax.Array):
        # prepare input data
        xs = jnp.concatenate([input_dict['context_x'], input_dict['target_x']], axis=1)
        x, xp = xs[:, :-1].reshape((-1, self.nx)), xs[:, 1:].reshape((-1, self.nx))
        u_pred = self(x, xp)
        lbl = input_dict['target_u'].reshape((-1, self.nu))

        return {'optimized_loss': jnp.square(u_pred - lbl).mean()}
