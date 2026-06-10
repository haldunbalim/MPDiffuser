import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm
from .misc import get_seed
import yaml
import os.path as op
from omegaconf import OmegaConf
from hydra.utils import instantiate
from mpdiffuser.model import Trainer, TrainState

def test(sample_action_fn, dset, n_trials, cost_limit=None, return_scale=None, seed=None, ep_len=1000):
    vec_env = dset.get_env(num_envs=n_trials)
    if seed is None:
        seed = get_seed()
    
    key = jax.random.PRNGKey(seed)
    
    if 'OfflinePoint' in dset.dset_name:
        obs = vec_env.reset(seed=get_seed()) # If I dont do this all envs are duplicate for some reason...
    else:
        obs = vec_env.reset()

    extra_conds = None
    if isinstance(obs, tuple):  # for gymnasium
        obs, info = obs
        extra_conds = info['conds'] if 'conds' in info else None
    if isinstance(obs, dict):
        obs = obs['observation']
    rewards, dones, costs = [], [], []
    term = np.array([False] * n_trials)
    if cost_limit is not None:
        cost_limit = jnp.ones(n_trials) * cost_limit
    if return_scale is not None:
        return_scale = jnp.ones(n_trials) * return_scale

    for ep_time in tqdm(range(ep_len)):
        key, r = jax.random.split(key)

        if dset.normalize_obs:
            obs = dset.x_normalizer.normalize(obs)
        action = sample_action_fn(
            obs, r, cost_limit=cost_limit, return_scale=return_scale, extra_conds=extra_conds)
        if dset.normalize_actions:
            action = dset.u_normalizer.unnormalize(action)
        ret = vec_env.step(action)
        if len(ret) == 6: # safety_gymnasium
            obs, rew, cost, terminated, truncated, info = ret
            done = terminated | truncated
            costs.append(cost)
            cost_limit = jnp.where(~term, cost_limit - cost, 0)
        elif len(ret) == 5:  # for gymnasium
            obs, rew, terminated, truncated, info = ret
            done = terminated | truncated
        elif len(ret) == 4:  # for gym
            obs, rew, done, info = ret
        else:
            raise ValueError('Unrecognized env return')
        extra_conds = info['conds'] if 'conds' in info else None
        
        if isinstance(obs, dict):
            obs = obs['observation']

        dones.append(done)
        rewards.append(rew)
        
        term = term | done
        if term.all():
            break

    dones = np.array(dones).T
    rewards = np.array(rewards).T
    if 'kitchen' in dset.dset_name:
        return rewards[:, -1], None
    dead_idx = (dones.argmax(axis=1) + 1) + (~dones.any(axis=1)) * dones.shape[1]
    rewards = jnp.where(jnp.arange(
        rewards.shape[1]) < dead_idx[:, None], rewards, 0)
    rewards = rewards.sum(axis=1)
    print(rewards)
    if costs == []:
        return rewards, None
    costs = np.array(costs).T
    costs = jnp.where(jnp.arange(
        costs.shape[1]) < dead_idx[:, None], costs, 0)
    costs = costs.sum(axis=1)
    print(costs)
    return rewards, costs

def load_dset(path, skip_sequencing=False):
    with open(op.join(path, 'combined.yaml'), 'r') as f:
        raw_dict = yaml.safe_load(f)
        cfg = OmegaConf.create(raw_dict)
    cfg = OmegaConf.merge(cfg.dataset, {"skip_sequencing": skip_sequencing})
    return instantiate(cfg)

def load_model(path, dset, epoch_num=None, ema=True, dtype: jnp.dtype = jnp.float32):
    with open(op.join(path, 'combined.yaml'), 'r') as f:
        raw_dict = yaml.safe_load(f)
        cfg = OmegaConf.create(raw_dict)
    dtype_node = OmegaConf.create({"dtype": dtype}, flags={"allow_objects": True})
    model_cfg = OmegaConf.merge(
        cfg.model, {"nx": dset.nx, "nu": dset.nu}, dtype_node)
    net = instantiate(model_cfg)

    state = Trainer.get_ckpt_manager(op.join(path, 'checkpoints')).restore(epoch_num)
    if ema and 'ema_variables' in state and state['ema_variables'] is not None:
        state['variables'] = state['ema_variables']
    state = TrainState(**state)
    return state, net

def save_results(args, parser, score, cost=None, filename='results'):
    filename = filename + '.txt'
    def get_save_str():
        kws = []
        for action in parser._actions:
            if action.dest in ['help', 'dset_name', 'no_save', 'use_fp32']:
                continue
            opts = [st for st in action.option_strings if st.startswith(
                '-') and st[:2] != '--']  # remove single-dash options
            assert len(
                opts) == 1, f"Option {action.dest} has multiple short options: {opts}"
            opt = opts[0].replace('-', '')
            kws.append(opt+'='+str(getattr(args, action.dest)))
        return ' '.join(kws)

    dset_name = args.dset_name.replace('/', '-')
    filename = op.join('outputs', dset_name, filename)
    save_st = get_save_str()
    if op.exists(filename):
        # create dct
        with open(filename, 'r') as f:
            dct = {}
            if cost is None:
                for line in f.readlines():
                    st, scr = line.split('\t')
                    dct[st] = float(scr.strip())
            else:
                for line in f.readlines():
                    st, scr, cost_scr = line.split('\t')
                    dct[st] = (float(scr.strip()), float(cost_scr.strip()))
        # update dct
        if save_st in dct:
            old_score = dct[save_st] if cost is None else dct[save_st][0]
            if score > old_score:
                dct[save_st] = (score, cost.mean()) if cost is not None else score
        else:
            dct[save_st] = (score, cost.mean()) if cost is not None else score
    else:
        dct = {save_st: (score, cost.mean()) if cost is not None else score}
    sorted_keys = [(k, v[0]) if cost is not None else (k, v) for k, v in dct.items()]
    sorted_keys = list(map(lambda x: x[0], sorted(sorted_keys, key=lambda x: x[1], reverse=True)))
    res = []
    for key in sorted_keys:
        if cost is not None:
            res.append(f'{key}\t{dct[key][0]}\t{dct[key][1]}')
        else:
            res.append(f'{key}\t{dct[key]}')
    with open(filename, 'w') as f:
        f.write('\n'.join(res))





    