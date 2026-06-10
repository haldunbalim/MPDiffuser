import jax.numpy as jnp
import flax.linen as nn
import math
import chex
import jax
import optax
from flax.struct import dataclass
from typing import Optional, Callable
from jax._src import dtypes as jax_dtypes
from typing import Tuple
from functools import partial
from mpdiffuser.utils.misc import get_optimizer, delay_target_update, update_params

act = jax.nn.swish

@dataclass
class TrainState:
    variables: jax.tree
    ema_variables: jax.tree
    opt_state: optax.OptState


class BaseModel(nn.Module):
    def get_run_name(self) -> str:
        """Human-readable sub-folder name written under outputs/{dset}/{ClassName}/."""
        return 'model'

    def get_batch_template(self) -> dict:
        raise NotImplementedError(' Base class ')

    def init_train_state(self, cfg) -> TrainState:
        batch_template = self.get_batch_template()
        variables = self.init(jax.random.PRNGKey(cfg.seed), *batch_template)
        self.ema_fn = partial(
            delay_target_update, update_ema_every=cfg.update_ema_every, ema_tau=cfg.ema_tau)
        self.optimizer = get_optimizer(
            lr0=cfg.lr0, weight_decay=cfg.weight_decay, gradient_clip_val=cfg.gradient_clip_val)
        opt_state = self.optimizer.init(variables)
        return TrainState(variables=variables, ema_variables=variables, opt_state=opt_state)

    def train_step(self, state: TrainState, input_dict: dict, key: jax.Array, step: int) -> tuple[TrainState, dict]:
        def loss_fn(variables):
            output_dict = self.apply(variables, input_dict,
                                     key=key, method=self.compute_outputs)

            return output_dict['optimized_loss'], output_dict

        variables, opt_state, output_dict = update_params(
            loss_fn, state.variables, state.opt_state, self.optimizer)
        ema_variables = self.ema_fn(step, variables, state.ema_variables)
        state = state.replace(variables=variables,
                              opt_state=opt_state, ema_variables=ema_variables)
        return state, output_dict

    def compute_outputs(self, input_dict: dict, key: jax.Array) -> dict:
        raise NotImplementedError(' Base class ')

    def val_step(self, state: TrainState, input_dict: dict, key: jax.Array, step: int) -> dict:
        output_dict = self.apply(
            state.ema_variables, input_dict, key=key, method=self.compute_outputs)
        return output_dict


def sinusoidal_pos_emb(x, dim):
    half_dim = dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = jnp.exp(jnp.arange(half_dim) * -emb)
    emb = x[:, None] * emb[None, :]
    emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)
    return emb


class Downsample1d(nn.Module):
    dim: int
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        return nn.Conv(self.dim, kernel_size=3, strides=2, padding="SAME", dtype=self.dtype)(x)


class Upsample1d(nn.Module):
    dim: int
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        return nn.ConvTranspose(self.dim, kernel_size=4, strides=2, padding="SAME", dtype=self.dtype)(x)


class Conv1dBlock(nn.Module):
    '''
        Conv1d --> GroupNorm --> Swish
    '''
    out_channels: int
    kernel_size: int
    n_groups: int = 8
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(self.out_channels, self.kernel_size,
                    padding='SAME', dtype=self.dtype)(x)
        x = nn.GroupNorm(num_groups=self.n_groups, dtype=self.dtype)(x)
        x = act(x)
        return x


class ResidualBlock(nn.Module):

    out_channels: int
    kernel_size: int = 5

    @nn.compact
    def __call__(self, x):
        '''
            x : [ batch_size x horizon x inp_channels]
            t : [ batch_size x embed_dim ]
            returns:
            out : [ batch_size x horizon x out_channels ]
        '''
        out = Conv1dBlock(self.out_channels, self.kernel_size)(x)
        out = Conv1dBlock(self.out_channels, self.kernel_size)(out)
        if x.shape[-1] != self.out_channels:
            x = nn.Conv(self.out_channels, kernel_size=1, padding='same')(x)
        return out + x


class ResidualTemporalBlock(nn.Module):

    out_channels: int
    kernel_size: int = 5

    @nn.compact
    def __call__(self, x, t):
        '''
            x : [ batch_size x horizon x inp_channels]
            t : [ batch_size x embed_dim ]
            returns:
            out : [ batch_size x horizon x out_channels ]
        '''
        out = Conv1dBlock(self.out_channels, self.kernel_size)(
            x) + nn.Dense(self.out_channels)(act(t))[:, None]
        out = Conv1dBlock(self.out_channels, self.kernel_size)(out)
        if x.shape[-1] != self.out_channels:
            x = nn.Conv(self.out_channels, kernel_size=1, padding='same')(x)
        return out + x


class FiLMBlock(nn.Module):
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x, cond):
        '''
            x : [ batch_size x horizon x inp_channels]
            t : [ batch_size x embed_dim ]
            returns:
            out : [ batch_size x horizon x out_channels ]
        '''
        # film
        scale_bias = MLP([cond.shape[-1], x.shape[-1] * 2],
                         hidden_act_class=act, dtype=self.dtype)(cond)
        scale, bias = jnp.split(scale_bias, 2, axis=-1)
        return x * scale[:, None] + bias[:, None]


class Conv1dFilmBlock(nn.Module):
    '''
        Conv1d --> GroupNorm --> Swish
    '''
    out_channels: int
    kernel_size: int
    n_groups: int = 8
    act_out: bool = False
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x, t):
        x = nn.Conv(self.out_channels, self.kernel_size, padding='SAME', dtype=self.dtype)(x)
        x = nn.GroupNorm(num_groups=self.n_groups, dtype=self.dtype)(x)
        x = FiLMBlock(dtype=self.dtype)(x, t)
        if self.act_out:
            x = act(x)
        return x


class ResidualTemporalFiLMBlock(nn.Module):

    out_channels: int
    kernel_size: int = 5
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x, t):
        out = Conv1dFilmBlock(
            self.out_channels, self.kernel_size, dtype=self.dtype)(x, t)
        out = Conv1dFilmBlock(
            self.out_channels, self.kernel_size, act_out=False, dtype=self.dtype)(out, t)
        if x.shape[-1] != self.out_channels:
            x = nn.Conv(self.out_channels, kernel_size=1,
                        padding='same', dtype=self.dtype)(x)
        return act(out + x)


class LinearAttention(nn.Module):
    heads: int = 4
    dim_head: int = 32
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        dim = x.shape[-1]
        hidden_dim = self.dim_head * self.heads
        qkv = nn.Conv(hidden_dim * 3, kernel_size=1, strides=1,
                      padding="SAME", use_bias=False, dtype=self.dtype)(x)
        qkv = jnp.split(qkv, 3, axis=-1)
        qkv = map(lambda t: jnp.reshape(
            t, (t.shape[0], -1, self.heads, self.dim_head)), qkv)  # b, t, h, d
        q, k, v = map(lambda t: jnp.transpose(
            t, (0, 2, 3, 1)), qkv)  # b, h, d, t
        q = q * (self.dim_head ** -0.5)

        k = jax.nn.softmax(k, axis=1)
        context = jnp.einsum('b h d n, b h e n -> b h d e', k, v)

        out = jnp.einsum('b h d e, b h d n -> b h e n', context, q)
        out = out.reshape(out.shape[0], -1, out.shape[-1])  # b, h*d, t
        out = jnp.transpose(out, (0, 2, 1))  # b, t, h*d
        out = nn.Conv(dim, kernel_size=1, strides=1,
                      padding="SAME", dtype=self.dtype)(out)
        return out


class TemporalUnet(nn.Module):
    dim: int = 32
    dim_mults: tuple = (1, 2, 4, 8)
    out_dim: int = 1
    attention: bool = False
    cond_embed_dim: int = 32
    input_transform_fn: Optional[Callable] = None

    @nn.compact
    def __call__(self, x, time, conds=[], *args):
        '''
            x : [ batch x horizon x transition ]
        '''

        if self.input_transform_fn is not None:
            x = self.input_transform_fn(x, *args)
        chex.assert_is_divisible(x.shape[1], 2 ** (len(self.dim_mults)-1))

        t = sinusoidal_pos_emb(time, self.dim)
        t = nn.Dense(self.dim * 4)(t)
        t = act(t)
        t = nn.Dense(self.dim)(t)

        # Process the conditions
        if not isinstance(conds, list):
            conds = [conds]
        _conds = []
        for cond in conds:
            cond = nn.Dense(self.cond_embed_dim * 4)(cond)
            cond = act(cond)
            cond = nn.Dense(self.cond_embed_dim)(cond)
            _conds.append(cond)
        t = jnp.concatenate([t, *_conds], axis=-1)

        dims = list(map(lambda m: self.dim * m, self.dim_mults))
        h = []
        for ind, d in enumerate(dims):
            x = ResidualTemporalBlock(d)(x, t)
            x = ResidualTemporalBlock(d)(x, t)
            if self.attention:
                x = x + LinearAttention()(nn.LayerNorm()(x))
            h.append(x)
            if ind < len(dims) - 1:
                x = Downsample1d(d)(x)

        x = ResidualTemporalBlock(dims[-1])(x, t)
        x = ResidualTemporalBlock(dims[-1])(x, t)

        for ind, d in enumerate(reversed(dims[:-1])):
            x = jnp.concatenate((x, h.pop()), axis=-1)
            x = ResidualTemporalBlock(d)(x, t)
            x = ResidualTemporalBlock(d)(x, t)
            if self.attention:
                x = x + LinearAttention()(nn.LayerNorm()(x))
            if ind < len(dims) - 1:
                x = Upsample1d(d)(x)

        x = Conv1dBlock(self.dim, kernel_size=5)(x)
        x = nn.Conv(self.out_dim, kernel_size=1, padding='SAME')(x)
        return x


class TemporalUnetFiLM(nn.Module):
    dim: int = 32
    cond_embed_dim: int = 64
    dim_mults: tuple = (1, 2, 4, 8)
    out_dim: int = 1
    attention: bool = False
    input_transform_fn: Optional[Callable] = None
    dtype: jnp.dtype = jnp.float32  

    @nn.compact
    def __call__(self, x, time, conds, *args):
        '''
            x : [ batch x horizon x transition ]
        '''

        if self.input_transform_fn is not None:
            x = self.input_transform_fn(x, *args)
        chex.assert_is_divisible(x.shape[1], 2 ** (len(self.dim_mults)-1))

        t = sinusoidal_pos_emb(time, self.dim)
        t = nn.Dense(self.dim * 4, dtype=self.dtype)(t)
        t = act(t)
        t = nn.Dense(self.dim, dtype=self.dtype)(t)

        # Process the conditions
        if not isinstance(conds, list):
            conds = [conds]
        _conds = []
        for cond in conds:
            cond = nn.Dense(self.cond_embed_dim * 4, dtype=self.dtype)(cond)
            cond = act(cond)
            cond = nn.Dense(self.cond_embed_dim, dtype=self.dtype)(cond)
            _conds.append(cond)

        t = jnp.concatenate([t, *_conds], axis=-1)  # concatenate time and cond

        dims = list(map(lambda m: self.dim * m, self.dim_mults))
        h = []
        for ind, d in enumerate(dims):
            x = ResidualTemporalFiLMBlock(d, dtype=self.dtype)(x, t)
            x = ResidualTemporalFiLMBlock(d, dtype=self.dtype)(x, t)
            if self.attention:
                x = x + \
                    LinearAttention(dtype=self.dtype)(
                        nn.LayerNorm(dtype=self.dtype)(x))
            h.append(x)
            if ind < len(dims) - 1:
                x = Downsample1d(d, dtype=self.dtype)(x)

        x = ResidualTemporalFiLMBlock(dims[-1], dtype=self.dtype)(x, t)
        x = ResidualTemporalFiLMBlock(dims[-1], dtype=self.dtype)(x, t)

        for ind, d in enumerate(reversed(dims[:-1])):
            x = jnp.concatenate((x, h.pop()), axis=-1)
            x = ResidualTemporalFiLMBlock(d, dtype=self.dtype)(x, t)
            x = ResidualTemporalFiLMBlock(d, dtype=self.dtype)(x, t)
            if self.attention:
                x = x + \
                    LinearAttention(dtype=self.dtype)(
                        nn.LayerNorm(dtype=self.dtype)(x))
            if ind < len(dims) - 1:
                x = Upsample1d(d, dtype=self.dtype)(x)

        x = Conv1dBlock(self.dim, kernel_size=5, dtype=self.dtype)(x)
        x = nn.Conv(self.out_dim, kernel_size=1, padding='SAME', dtype=self.dtype)(x)
        return x


class Resnet1D(nn.Module):
    dim: int = 32
    dim_mults: tuple = (1, 2, 4, 8)

    @nn.compact
    def __call__(self, x, time):
        '''
            x : [ batch x horizon x transition ]
        '''

        # chex.assert_is_divisible(x.shape[1], 2 ** (len(self.dim_mults)-1))
        dims = list(map(lambda m: self.dim * m, self.dim_mults))

        t = sinusoidal_pos_emb(time, self.dim)
        t = nn.Dense(self.dim * 4)(t)
        t = act(t)
        t = nn.Dense(self.dim)(t)
        for d in dims:
            x = ResidualTemporalBlock(d)(x, t)
            x = ResidualTemporalBlock(d)(x, t)
            x = Downsample1d(d)(x)

        x = ResidualTemporalBlock(dims[-1] // 2)(x, t)
        x = Downsample1d(dims[-1])(x)
        x = ResidualTemporalBlock(dims[-1] // 4)(x, t)
        x = Downsample1d(dims[-1])(x)

        x = x.reshape(x.shape[0], -1)
        x = nn.Dense(x.shape[-1] // 2)(jnp.concatenate((x, t), axis=-1))
        x = act(x)
        x = nn.Dense(1)(x)
        return x[:, 0]

def bias_init(dtype=jnp.float_):
    def init(key, shape, dtype=dtype):
        feats = shape[0]
        scale = jnp.sqrt(1 / feats)
        dtype = jax_dtypes.canonicalize_dtype(dtype)
        return jax.random.uniform(key, shape, dtype, minval=-scale, maxval=scale)
    return init


class Resnet1DFilm(nn.Module):
    dim: int = 32
    dim_mults: tuple = (1, 2, 4, 8)
    cond_embed_dim: int = 64

    @nn.compact
    def __call__(self, x, time, cond):
        '''
            x : [ batch x horizon x transition ]
        '''

        chex.assert_is_divisible(x.shape[1], 2 ** (len(self.dim_mults)-1))
        dims = list(map(lambda m: self.dim * m, self.dim_mults))

        t = sinusoidal_pos_emb(time, self.dim)
        t = nn.Dense(self.dim * 4)(t)
        t = act(t)
        t = nn.Dense(self.dim)(t)

        cond = nn.Dense(self.cond_embed_dim * 4)(cond)
        cond = act(cond)
        cond = nn.Dense(self.cond_embed_dim)(cond)

        t = jnp.concatenate([t, cond], axis=-1)  # concatenate time and cond

        for d in dims:
            x = ResidualTemporalFiLMBlock(d)(x, t)
            x = ResidualTemporalFiLMBlock(d)(x, t)
            x = Downsample1d(d)(x)

        x = ResidualTemporalFiLMBlock(dims[-1] // 2)(x, t)
        x = Downsample1d(dims[-1])(x)
        x = ResidualTemporalFiLMBlock(dims[-1] // 4)(x, t)
        x = Downsample1d(dims[-1])(x)

        x = x.reshape(x.shape[0], -1)
        x = nn.Dense(x.shape[-1] // 2)(jnp.concatenate((x, t), axis=-1))
        x = act(x)
        x = nn.Dense(1)(x)
        return x[:, 0]


class MLP(nn.Module):
    feature_sizes: list
    hidden_act_class: type = nn.relu
    act_out: Optional[callable] = None
    # kernel_init: nn.initializers = nn.initializers.variance_scaling(
    #    scale=1/3, mode='fan_in', distribution='uniform')
    # bias_init: nn.initializers = bias_init()
    use_layer_norm: bool = False
    squeeze_output: bool = False
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        for i, sz in enumerate(self.feature_sizes):
            x = nn.Dense(sz, dtype=self.dtype)(x)  # Linear layer
            if i != len(self.feature_sizes) - 1:
                if self.use_layer_norm:
                    x = nn.LayerNorm(dtype=self.dtype)(x)
                x = self.hidden_act_class(x)
        if self.act_out is not None:
            x = self.act_out(x)  # Optional output activation
        if self.squeeze_output:
            x = jnp.squeeze(x, axis=-1)
        return x
