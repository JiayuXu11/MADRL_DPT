#!/usr/bin/env python
import sys
import os
import socket
import numpy as np
from pathlib import Path
import torch
from config import get_config
from envs.env_wrappers import SubprocVecEnv, DummyVecEnv, EvalVecEnv
from runners.separated.runner import CRunner as Runner
import yaml
import random


def seed_torch(seed=0):
    random.seed(seed)   
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)   
    torch.manual_seed(seed)   
    torch.cuda.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed)
    # if benchmark=True, deterministic will be False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True   


seed_torch()


def make_train_env(all_args):
    return SubprocVecEnv(all_args)


def make_eval_env(all_args, eval_tf=True):
    return EvalVecEnv(all_args, eval_tf)


def parse_args(args, parser):
    all_args = parser.parse_known_args(args)[0]
    return all_args


if __name__ == "__main__":

    if not torch.cuda.is_available():
        print("no cuda available")
        sys.exit(1)

    parser = get_config()

    
    all_args = parse_args(sys.argv[1:], parser)


    if all_args.yaml_path:
        with open(all_args.yaml_path, 'r') as f:
            yml = yaml.load(f, Loader=yaml.FullLoader)
        parser.set_defaults(**yml)
    all_args = parse_args(sys.argv[1:], parser)
    print(all_args.seed)

    # initialization setting of SKU
    demand_mean = np.load('envs/demand_mean.npy',allow_pickle=True).item()
    demand_max = np.load('envs/demand_max.npy',allow_pickle=True).item()
    skus = ['SKU006', 'SKU019', 'SKU022', 'SKU029', 'SKU032',
        'SKU045', 'SKU046', 'SKU062']
    scales = [7, 53, 142, 25, 6, 98, 46, 20]
    scale_dict = dict(zip(skus, scales))
    if all_args.SKU_id:
        all_args.test_dir = './test_data/{}'.format(all_args.SKU_id) 
        all_args.eval_dir = './eval_data/{}'.format(all_args.SKU_id)
        all_args.train_dir = './train_data/{}'.format(all_args.SKU_id)
        all_args.demand_for_action_dim = demand_mean[str(all_args.SKU_id)]
        all_args.demand_max_for_clip = demand_max[str(all_args.SKU_id)]
        all_args.FIXED_COST_scale = scale_dict[str(all_args.SKU_id)] if all_args.use_scale else 1
        all_args.FIXED_COST = all_args.FIXED_COST_scale

    # actions within the last leadtime days are not included in the training
    all_args.train_episode_length = all_args.episode_length-all_args.lead_time-1
    # If the value function represents infinite future returns, then training is conducted every day
    if (not all_args.setting_time_end) and (not all_args.ignore_after):
        all_args.train_episode_length = all_args.episode_length

    all_args.num_env_steps = all_args.num_episodes * \
        all_args.episode_length * all_args.n_rollout_threads


    seeds = all_args.seed

    print("all config: ", all_args)
    # cuda
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    all_args.device = device

    for seed in seeds:
        print("-------------------------------------------------Training starts for seed: " +
              str(seed) + "---------------------------------------------------")

        run_dir = Path("results") / all_args.env_name / all_args.scenario_name / \
            all_args.algorithm_name / all_args.experiment_name
        if not run_dir.exists():
            os.makedirs(str(run_dir), mode=0o777)

        curr_run = 'run_seed_%i' % (seed + 1)

        seed_res_record_file = run_dir / "{}.txt".format(all_args.experiment_name)

        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))

        if not os.path.exists(seed_res_record_file):
            open(seed_res_record_file, 'a+')

        seed_torch(seed)

        all_args.clip_bound = [(1-all_args.clip_param),
                               (1+all_args.clip_param)]

        num_agents = all_args.num_agents

        # Match the number of stores (num_involver) with the number of training agents (num_agents)
        all_args.num_involver = max(num_agents, all_args.num_involver)
        num_agents = max(num_agents, all_args.num_involver)

        if all_args.central_controller:
            num_agents = 1
            all_args.instant_info_sharing = True
            all_args.action_type = 'central_multi_discrete' if 'multi' in all_args.action_type else 'central_discrete'
            all_args.alpha = 0

        # env
        envs = make_train_env(all_args)
        eval_envs = make_eval_env(all_args, eval_tf=True)
        test_envs = make_eval_env(all_args, eval_tf=False)

        config = {
            "all_args": all_args,
            "envs": envs,
            "eval_envs": eval_envs,
            "test_envs": test_envs,
            "num_agents": num_agents,
            "device": device,
            "run_dir": run_dir
        }

        # run experiments
        runner = Runner(config)

        eval_reward, eval_dict, test_reward, test_dict = runner.run()

        with open(seed_res_record_file, 'a+') as f:
            f.write(str(seed) + 'eval_cost' + str(eval_reward) +
                    'test_cost' + str(test_reward))
            f.write('\n')
            f.write('eval_cost_composition' + str(eval_dict))
            f.write('\n')
            f.write('test_cost_composition' + str(test_dict))
            f.write('\n')
            f.write('\n')
            f.write('all_args'+str(all_args))
            f.write('\n')

        if all_args.action_type == 'continue':
            all_args.std_y_coef = [j/2. for j in all_args.std_y_coef]
            all_args.lr = all_args.lr/2

        all_args.model_dir = str(config['run_dir'] / 'models')

        # post process
        envs.close()
        if all_args.use_eval and eval_envs is not envs:
            eval_envs.close()
            # break
