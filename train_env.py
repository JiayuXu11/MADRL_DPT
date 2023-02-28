#!/usr/bin/env python
import sys
import os
import socket
import numpy as np
from pathlib import Path
import torch
from config import get_config
from envs.env_wrappers_continue import SubprocVecEnv, DummyVecEnv
from runners.separated.runner import CRunner as Runner
import yaml

def make_train_env(all_args):
    return SubprocVecEnv(all_args)

def make_eval_env(all_args):
    return DummyVecEnv(all_args)

def parse_args(args, parser):
    all_args = parser.parse_known_args(args)[0]
    return all_args


if __name__ == "__main__":
    parser = get_config()
    all_args = parse_args(sys.argv[1:], parser)
    if all_args.yaml_path:
        with open(all_args.yaml_path,'r') as f:
            yml = yaml.load(f,Loader=yaml.FullLoader)
        parser.set_defaults(**yml)
    all_args = parse_args(sys.argv[1:], parser)
    print(all_args.seed)

    # all_args.algorithm_name='heuristic2'
    # all_args.discrete=False
    # all_args.norm_input =False

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

    for seed in seeds:
        print("-------------------------------------------------Training starts for seed: " + str(seed)+ "---------------------------------------------------")

        # run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
        #                 0] + "/results") / all_args.env_name / all_args.scenario_name / all_args.algorithm_name / all_args.experiment_name
        run_dir = Path("results") / all_args.env_name / all_args.scenario_name / all_args.algorithm_name / all_args.experiment_name
        if not run_dir.exists():
            os.makedirs(str(run_dir), mode=0o777)

        curr_run = 'run_seed_%i' % (seed + 1)

        seed_res_record_file = run_dir / "seed_results.txt"
        
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))
        

        if not os.path.exists(seed_res_record_file):
            open(seed_res_record_file, 'a+')

        # seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        # 手动调节波动
        # all_args.std_y_coef=[j/2**4 for j in all_args.std_y_coef]
        # all_args.lr=all_args.lr/2**4
        all_args.clip_bound=[(1-all_args.clip_param),(1+all_args.clip_param)]
        for i in range(4):
            # env
            envs = make_train_env(all_args)
            eval_envs = make_eval_env(all_args) if all_args.use_eval else None
            num_agents = 1 if all_args.central_controller else all_args.num_agents

            config = {
                "all_args": all_args,
                "envs": envs,
                "eval_envs": eval_envs,
                "num_agents": num_agents,
                "device": device,
                "run_dir": run_dir
            }

            # run experiments
            runner = Runner(config)

            eval_reward, test_reward = runner.run()

            with open(seed_res_record_file, 'a+') as f:
                f.write(str(all_args.std_y_coef)+'   '+str(seed) + ' ' + str(eval_reward) + ' ' + str(test_reward))
                f.write('\n')
            if all_args.discrete or all_args.beta:
                break
            all_args.std_y_coef=[j/2. for j in all_args.std_y_coef]
            all_args.lr=all_args.lr/2
            all_args.model_dir = str(config['run_dir'] / 'models')
            all_args.clip_bound=[x**1.2 for x in all_args.clip_bound]

        # post process
        envs.close()
        if all_args.use_eval and eval_envs is not envs:
            eval_envs.close()
            break

    