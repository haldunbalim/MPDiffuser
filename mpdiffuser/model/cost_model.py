import flax.linen as nn
import jax.numpy as jnp
import jax
from .networks import MLP, BaseModel
import optax


class CostModel(BaseModel):
    nx: int
    nu: int
    horizon: int

    hidden_dims: tuple[int] = (256, 256)
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.network = MLP((*self.hidden_dims, 1), act_out=None, squeeze_output=True, dtype=self.dtype)
    
    def get_batch_template(self):
        batch_template = (jnp.zeros((1, self.nx)),
                          jnp.zeros((1, self.nu)))
        return batch_template
    
    @nn.compact
    def __call__(self, x, u):
        return self.network(jnp.concatenate([x, u], axis=-1))

    def compute_outputs(self, input_dict: dict, key: jax.Array):
        # prepare input data
        x = input_dict['context_x'][:, 0]
        u = input_dict['target_u'][:, 0]
        
        cost_pred = self(x, u)
        loss = jnp.mean((cost_pred - input_dict['cost'])**2)
        acc = jnp.mean((cost_pred > 0.5) == input_dict['cost'])

        return {'optimized_loss': loss, 'accuracy': acc}

