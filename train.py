"""Train an algorithm."""
import argparse
import ast
import json
from harl.utils.configs_tools import get_defaults_yaml_args, update_args, t_or_f, deep_update
import sys
import os

if hasattr(os, "umask"):
    os.umask(0)

def is_debug():
    return sys.gettrace() is not None

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="happo",
        choices=["happo", "hasac"],
        help="Policy algorithm.",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="multi_lt_transship",
        choices=[
            "multi_lt_transship"
        ],
        help="Inventory and transshipment environment.",
    )
    parser.add_argument(
        "--exp_name", type=str, default="experiment", help="Experiment name."
    )
    parser.add_argument(
        "--load_config",
        type=str,
        default="configs/backlog/default.json",
        help="If set, load existing experiment config file instead of reading from yaml config file.",
    )
    parser.add_argument(
        "--load_change_config",
        type=str,
        default='',



        help="Optional configuration overlays, separated by commas.",
    )
    parser.add_argument(
        "--num_transitions_per_agent",
        type=int,
        default=20000,
        help="Transitions per retailer per iteration; derives parallel environment count.",
    )
    parser.add_argument(
        "--train_times",
        type=int,
        default=20000,
        help="Maximum training iterations; derives total environment steps.",
    )

    args, unparsed_args = parser.parse_known_args()

    def process(arg):
        try:
            return ast.literal_eval(arg)
        except:
            return arg

    keys = [k[2:] for k in unparsed_args[0::2]]  # remove -- from argument
    values = [process(v) for v in unparsed_args[1::2]]
    unparsed_dict = {k: v for k, v in zip(keys, values)}
    args = vars(args)  # convert to dict
    if args["load_config"] != "":  # load config from existing config file
        with open(args["load_config"], encoding="utf-8") as file:
            all_config = json.load(file)
        if args["load_change_config"] != "":
            change_config_paths = [path.strip() for path in args["load_change_config"].split(',')]
            for change_config_path in change_config_paths:
                if os.path.exists(change_config_path):
                    with open(change_config_path, encoding="utf-8") as file:
                        change_config = json.load(file)
                    all_config = deep_update(all_config, change_config)
                else:
                    raise FileNotFoundError(change_config_path)
        print(all_config)
        # 更新args,除了exp_name
        exp_name = args["exp_name"]
        args = deep_update(args, all_config["main_args"])
        args["exp_name"] = exp_name
        algo_args = all_config["algo_args"]
        env_args = all_config["env_args"]
        if "algo_mechanism" in all_config["main_args"]:
            args["algo_mechanism"] = all_config["main_args"]["algo_mechanism"]
            algo_mechanism_args = all_config["algo_mechanism_args"]
    else:  # load config from corresponding yaml file
        raise ValueError("--load_config must name an experiment configuration")
    update_args(unparsed_dict, algo_args, env_args)  # update args from command line

    # 根据 num_transitions_per_agent 和 train_times 调整训练参数
    num_agents = env_args["num_agents"]
    share_param_tf = algo_args["algo"].get("share_param", False)
    num_transitions_per_agent = args.get("num_transitions_per_agent")
    train_times = args.get("train_times")
    episode_length = algo_args["train"]["episode_length"]

    if num_transitions_per_agent is not None:
        # 计算 n_rollout_threads = num_transitions_per_agent / episode_length
        if share_param_tf:
            n_rollout_threads = num_transitions_per_agent // episode_length // num_agents
        else:
            n_rollout_threads = num_transitions_per_agent // episode_length
        if n_rollout_threads < 1:
            n_rollout_threads = 1
        algo_args["train"]["n_rollout_threads"] = n_rollout_threads
        env_args["num_envs"] = n_rollout_threads
        if "algo_mechanism" in args:
            algo_mechanism_args["train"]["n_rollout_threads"] = n_rollout_threads

    if train_times is not None:
        # 计算 num_env_steps = train_times * n_rollout_threads * episode_length
        n_rollout_threads = algo_args["train"]["n_rollout_threads"]
        num_env_steps = train_times * n_rollout_threads * episode_length
        algo_args["train"]["num_env_steps"] = num_env_steps
        if "algo_mechanism" in args:
            algo_mechanism_args["train"]["num_env_steps"] = num_env_steps

    algo_mechanism_args["model"]["num_agents"] = num_agents
    algo_mechanism_args["model"]["mechanism_agent_type"] = env_args["mechanism_agent_type"]
    algo_mechanism_args["train"]["mechanism_agent_type"] = env_args["mechanism_agent_type"]
    algo_args["train"]["mechanism_agent_type"] = env_args["mechanism_agent_type"]

    # 识别是否在debug
    if is_debug():
        algo_args["train"]["n_rollout_threads"] = 4
        algo_args["train"]["eval_interval"] = 6
        algo_args["eval"]["n_eval_rollout_threads"] = 20
        if "algo_mechanism" in args:
            algo_mechanism_args["train"]["n_rollout_threads"] = 4
            algo_mechanism_args["train"]["warmup_steps"] = 200
            # algo_mechanism_args["eval"]["n_rollout_threads"] = 2

    # start training
    from harl.runners import RUNNER_REGISTRY

    runner = RUNNER_REGISTRY["mix"](args, algo_args, env_args, algo_mechanism_args)

    runner.run()
    runner.close()



if __name__ == "__main__":
    main()
