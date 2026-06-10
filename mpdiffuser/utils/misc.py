import jax
import json
import hashlib
from functools import reduce
import numpy as np
import chex
import wandb
import os.path as op
import tempfile
import random
import numpy as np
import inspect
from functools import partial
import contextlib
import io
import logging
import optax
import jax.numpy as jnp

logger = logging.getLogger(__name__)

def compose(*funcs):
    return lambda x: reduce(lambda v, f: f(v), funcs, x)


def get_seed():
    return np.random.randint(0, 2**32 - 1)

def repr_tree(d, prefix=''):
    """Recursively prints a dictionary as a tree structure."""
    st = ''
    for key, value in d.items():
        if isinstance(value, dict):
            # If the value is a dictionary, print the key and recurse
            st += f"{prefix}├── {key}\n"
            st += repr_tree(value, prefix + "│   ")
        else:
            # If the value is not a dictionary, print the key-value pair
            st += f"{prefix}├── {key}: size = {value}\n"
    return st


def wandb_log_model_arch(params):
    shapes = jax.tree.map(lambda x: x.shape, params)
    str_model = repr_tree(shapes)
    artifact = wandb.Artifact(name="model_arch", type="arch_str")
    with tempfile.TemporaryDirectory() as tmp_dir:
        filename = op.join(tmp_dir, 'arch.txt')
        with open(filename, 'w') as f:
            f.write(str_model)
        artifact.add_file(filename)
    wandb.log_artifact(artifact)


def flatten_dict(data, parent_key='', sep='-'):
    """
    Flattens a nested dictionary into a single-level dictionary.

    Parameters:
    - data (dict): The nested dictionary structure.
    - parent_key (str): The base key used for recursion (default: '').
    - sep (str): The separator for concatenated keys (default: '-').

    Returns:
    - dict: A flattened dictionary with concatenated keys.
    """
    flat_dict = {}
    for key, value in data.items():
        # Create new key by appending current key to the parent key
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        if isinstance(value, dict):
            # Recursively flatten the nested dictionary
            flat_dict.update(flatten_dict(value, new_key, sep=sep))
        else:
            # Add to flat dictionary
            flat_dict[new_key] = value
    return flat_dict


def accumulate_metrics(metrics):
    return {
        k: np.mean([metric[k] for metric in metrics])
        for k in metrics[0]
    }

def hash_config(cfg):
    config_str = json.dumps(cfg, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()


def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)


def log_info(st):
    if type(st) != str:
        st = str(st)
    return logger.info(st)


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(
                Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class HiddenPrints(contextlib.redirect_stdout):
    def __init__(self):
        super().__init__(io.StringIO())


def create_iter_dataset(dataset, batch_size, seed=None):
    rng = np.random.default_rng(seed)
    while True:
        batch_indices = rng.permutation(len(dataset))  # shuffle
        batch_indices = batch_indices[:len(
            dataset) - (len(dataset) % batch_size)]  # drop last
        batch_indices = batch_indices.reshape(-1, batch_size)  # batch
        for batch_idx in batch_indices:
            yield dataset[batch_idx]

def get_optimizer(lr0=2e-4, weight_decay=0.0, gradient_clip_val=1.0):
    optimizer = optax.adamw(lr0, weight_decay=weight_decay)
    if gradient_clip_val > 0.0:
        optimizer = optax.chain(optax.clip_by_global_norm(gradient_clip_val), optimizer)
    return optimizer


def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate


def update_params(loss_fn, params, opt_state, optimizer, has_aux=True):
    gradient_fn = jax.value_and_grad(loss_fn, has_aux=has_aux)
    if has_aux:
        (_, aux), grads = gradient_fn(params)
    else:
        _, grads = gradient_fn(params)

    grads = jax.lax.pmean(grads, axis_name='devices')
    param_updates, opt_state = optimizer.update(
        grads, opt_state, params)
    params = optax.apply_updates(params, param_updates)
    if has_aux:
        return params, opt_state, aux
    else:
        return params, opt_state
    

def delay_target_update(step, vars, target_vars, update_ema_every, ema_tau):
    return jax.lax.cond(
        step % update_ema_every == 0,
        lambda target_params: optax.incremental_update(
            vars, target_params, ema_tau),
        lambda target_params: target_params,
        target_vars
    )


def apply_cfg(sample_fn, conds, cfg_scale, *args):
    if cfg_scale == 1.0:
        return sample_fn(conds, *args)
    else:
        conds = jnp.concatenate([jnp.zeros_like(conds), conds], axis=0)
        args = [jnp.repeat(arg, 2, axis=0) for arg in args]

        ret = sample_fn(conds, *args)
        uncond, cond = ret[:ret.shape[0]//2], ret[ret.shape[0]//2:]
        return (1-cfg_scale) * uncond + cfg_scale * cond
