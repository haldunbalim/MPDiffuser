import argparse
import glob
import os
import os.path as op

import jax.numpy as jnp
import numpy as np

from mpdiffuser import *
from mpdiffuser.utils.test import *

METHODS = ('planner', 'mpdiffuser', 'dd', 'guided')

# Maps method name → title-cased class name of the diffusion model used.
MODEL_DIR = {
    'planner':    'Planner',          # Planner class
    'mpdiffuser': 'Planner',          # Planner class (FiLM-conditioned)
    'dd':         'Decisiondiffuser', # DecisionDiffuser class
    'guided':     'Diffuser',         # Diffuser class
}

RESULT_FILENAME = {
    'planner':         'results-cfg',
    'mpdiffuser':      'results',
    'mpdiffuser_multi':'results-multi',
    'dd':              'results-dd',
}


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, choices=METHODS, required=True,
                        help='planner=PlannerPolicy, mpdiffuser=MPDiffuser, '
                             'dd=DDPolicy, guided=DiffuserPolicy')
    parser.add_argument('--dset-name', type=str, default='mujoco/hopper/expert-v0')
    parser.add_argument('--out-dir', type=str, default='outputs')

    # Path overrides (optional; constructed automatically if not given)
    parser.add_argument('--path', type=str, default=None,
                        help='Override diffusion model path.')
    parser.add_argument('--dyn-path', type=str, default=None,
                        help='Override dynamics model path (mpdiffuser only).')
    parser.add_argument('--value-path', type=str, default=None,
                        help='Override value/reward model path.')
    parser.add_argument('--cost-path', type=str, default=None,
                        help='Override cost model path.')
    parser.add_argument('--inv-dyn-path', type=str, default=None,
                        help='Override inverse dynamics model path (dd only).')

    # Diffusion / run settings
    parser.add_argument('--horizon', '-hr', type=int, default=None)
    parser.add_argument('--n-steps', '-s', type=int, default=100)
    parser.add_argument('--predict-noise', action='store_true', default=False,
                        help='Model was trained with predict_noise=True → adds -eps to folder name.')
    parser.add_argument('--suffix', '-sfx', type=str, default=None,
                        help='Optional extra suffix appended to the run folder name.')

    # Evaluation settings
    parser.add_argument('--n-samples', '-ns', type=int, default=1)
    parser.add_argument('--n-trials', '-nt', type=int, default=50)
    parser.add_argument('--cfg-scale', '-cfg', type=float, default=2.0)
    parser.add_argument('--return-scale', '-crs', type=float, default=1.1)
    parser.add_argument('--cost-scale', '-cs', type=float, default=0.0)
    parser.add_argument('--cost-limit', '-cl', type=float, default=None)
    parser.add_argument('--temperature', '-tmp', type=float, default=0.001)
    parser.add_argument('--t-replan-step', '-trpln', type=int, default=None)
    parser.add_argument('--skip-sample', '-ss', type=int, default=1)
    parser.add_argument('--ddim', '-dd', action='store_true', default=False)
    parser.add_argument('--use-fp32', action='store_true', default=False)
    parser.add_argument('--no-save', action='store_true', default=False)

    # guided-only
    parser.add_argument('--n-guide-steps', type=int, default=2)
    parser.add_argument('--t-stopgrad', type=int, default=2)
    parser.add_argument('--scale', type=float, default=0.1)
    return parser


def infer_horizon(dset_name):
    if 'expert' in dset_name:
        return 32
    if 'medium' in dset_name:
        return 16
    if 'simple' in dset_name:
        return 8
    raise ValueError(f'Cannot infer horizon for {dset_name!r} — pass --horizon explicitly.')


def build_paths(args):
    dset_name = args.dset_name.replace('/', '-')
    sfx = '-eps' if args.predict_noise else ''
    sfx += f'-{args.suffix}' if args.suffix else ''
    run_name = f'cfg-h{args.horizon}-s{args.n_steps}{sfx}'

    path = args.path or op.join(args.out_dir, dset_name, MODEL_DIR[args.method], run_name)
    paths = {'path': path}

    if args.method == 'mpdiffuser':
        paths['dyn_path'] = args.dyn_path or op.join(
            args.out_dir, dset_name, 'Dynamicsmodel', run_name)

    if args.method == 'dd':
        paths['inv_dyn_path'] = args.inv_dyn_path or op.join(
            args.out_dir, dset_name, 'Invdynmodel', 'model')

    if args.method == 'guided':
        paths['value_path'] = args.value_path or op.join(
            args.out_dir, dset_name, 'Valuefunction', f'value-h{args.horizon}')
    else:
        paths['value_path'] = args.value_path or op.join(
            args.out_dir, dset_name, 'Rewardmodel', 'model')
        paths['cost_path'] = args.cost_path or op.join(
            args.out_dir, dset_name, 'Costmodel', 'model')

    return paths


def build_policy(args, paths, dset, dtype):
    state, diff_model = load_model(paths['path'], dset, epoch_num=None, dtype=dtype)

    if args.method == 'guided':
        value_state, value_model = load_model(paths['value_path'], dset, epoch_num=None, dtype=dtype)
        poli = DiffuserPolicy(diff_model, value_model)
        poli.load_state(state, value_state)
        return poli

    if args.n_samples > 1:
        value_state, value_model = load_model(paths['value_path'], dset, epoch_num=None, dtype=dtype)
        if args.cost_limit is not None:
            cost_state, cost_model = load_model(paths['cost_path'], dset, epoch_num=None, dtype=dtype)
        else:
            cost_state, cost_model = None, None
    else:
        value_state, value_model = None, None
        cost_state, cost_model = None, None

    if args.method == 'planner':
        poli = PlannerPolicy(diff_model, value_model, cost_model)
        poli.load_state(state, value_state, cost_state)
    elif args.method == 'mpdiffuser':
        dy_state, dy_model = load_model(paths['dyn_path'], dset, epoch_num=None, dtype=dtype)
        poli = MPDiffuser(diff_model, dy_model, value_model, cost_model)
        poli.load_state(state, dy_state, value_state, cost_state)
    elif args.method == 'dd':
        inv_dyn_state, inv_dyn_model = load_model(paths['inv_dyn_path'], dset, epoch_num=None)
        poli = DDPolicy(diff_model, inv_dyn_model, value_model, cost_model)
        poli.load_state(state, inv_dyn_state, value_state, cost_state)
    else:
        raise ValueError(f'Unknown method: {args.method}')
    return poli


def get_sample_action_fn(args, poli, clip):
    if args.method == 'guided':
        return poli.get_sample_action_fn(
            n_samples=args.n_samples, temperature=args.temperature, scale=args.scale,
            n_guide_steps=args.n_guide_steps, t_stopgrad=args.t_stopgrad, clip=clip)

    common = dict(n_samples=args.n_samples, cfg_scale=args.cfg_scale,
                  temperature=args.temperature, cost_scale=args.cost_scale, clip=clip)
    if args.method != 'dd':
        common.update(skip=args.skip_sample, ddim=args.ddim)

    if args.t_replan_step is not None and args.t_replan_step < args.n_steps:
        return poli.get_sample_action_fn_replan(t_replan_step=args.t_replan_step, **common)
    return poli.get_sample_action_fn(**common)


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.horizon is None:
        args.horizon = infer_horizon(args.dset_name)

    dtype = jnp.float32 if args.use_fp32 else jnp.bfloat16
    paths = build_paths(args)

    dset = load_dset(paths['path'], skip_sequencing=True)
    poli = build_policy(args, paths, dset, dtype)

    clip = dset.normalize_obs and dset.obs_normalizer_typ != 'std'
    sample_action_fn = get_sample_action_fn(args, poli, clip)

    rewards, cost = test(sample_action_fn, dset, args.n_trials,
                         return_scale=args.return_scale,
                         cost_limit=args.cost_limit)

    dset_name = args.dset_name.replace('/', '-')
    print('Tested on', dset_name, 'with', op.basename(paths['path']))
    if args.method == 'mpdiffuser':
        print(args.n_steps, args.cfg_scale, args.return_scale)
    rewards = dset.get_normalized_scores(rewards)
    print('Normalized return:', rewards.mean(), '±', rewards.std() / np.sqrt(len(rewards)))
    if cost is not None:
        cost = cost / args.cost_limit
        print('Normalized cost:', cost.mean(), '±', cost.std() / np.sqrt(len(cost)))

    if args.no_save or args.method == 'guided':
        return

    if args.method == 'mpdiffuser':
        filename = RESULT_FILENAME['mpdiffuser_multi'] if args.n_samples > 1 else RESULT_FILENAME['mpdiffuser']
    else:
        filename = RESULT_FILENAME[args.method]
    save_results(args, parser, rewards.mean(), cost=cost, filename=filename)


if __name__ == '__main__':
    main()
