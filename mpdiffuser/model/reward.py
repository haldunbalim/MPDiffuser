import flax.linen as nn
import jax.numpy as jnp
from typing import Tuple
import jax
from .networks import MLP, BaseModel
import optax


class RewardModel(BaseModel):
    nx: int
    nu: int
    horizon: int

    hidden_dims: tuple[int] = (256, 256)
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.network = MLP((*self.hidden_dims, 2), act_out=None, dtype=self.dtype)
    
    def get_batch_template(self):
        batch_template = (jnp.zeros((1, self.nx)),
                          jnp.zeros((1, self.nu)))
        return batch_template
    
    @nn.compact
    def __call__(self, x, u):
        inp = jnp.concatenate([x, u], axis=-1)
        r_pred, term_pred =  jnp.split(self.network(inp), 2, axis=-1)
        return r_pred.squeeze(-1), term_pred.squeeze(-1)

    def compute_outputs(self, input_dict: dict, key: jax.Array):
        # prepare input data
        x = input_dict['context_x'][:, 0]
        u = input_dict['target_u'][:, 0]
        
        r_pred, term_pred = self(x, u)
        r_lbl = input_dict['reward'].reshape(-1)
        term_lbl = input_dict['termination'].reshape(-1)

        r_loss = jnp.square(r_pred - r_lbl).mean()
        term_loss = optax.sigmoid_binary_cross_entropy(term_pred, term_lbl).mean()

        term_acc = jnp.mean((term_pred > 0.5) == term_lbl)

        return {'optimized_loss': r_loss + term_loss, 
                'r_loss': r_loss, 
                'term_loss': term_loss, 
                'term_acc': term_acc}

