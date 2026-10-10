import json
import os
import time
import argparse
import ast
import csv

import numpy as np
from scipy import stats
from tensorboardX import SummaryWriter

from harl.envs.multi_lt_transship.transship_vec_env import VecTransshipEnv
from harl.envs.multi_lt_transship.baseline_vec import VecBaseline, VecEppen
from harl.envs import LOGGER_REGISTRY
from harl.utils.configs_tools import deep_update


class BaselineExperiment:
    """
    A class to run baseline strategies (search, quantile_k, eppen) with reusable components.

    Args:
        args (dict): Default arguments for environment and baseline configuration.
                     Should include:
                     - 'eval_episodes': Number of evaluation episodes (default: 1)
                     - 'episode_length': Length of each episode (default: 10)
                     - 'search_param': Dict with search parameters:
                       - 'search_k_range_start': Start of k range (default: -4)
                       - 'search_k_range_end': End of k range (default: 10)
                       - 'search_k_range_step': Step size for k range (default: 0.3)
                       - 'search_eval_episodes_raw': Episodes for coarse search (default: 5)
                       - 'search_consecutive_worse_limit': Stop after N worse results (default: 2)
        result_dir (str): Directory to save results
        run_function (list or str): Function(s) to run. Can be a single string or a list of:
                                    ['search', 'cal_quantile_k', 'eppen']
    """

    def __init__(
        self,
        args,
        result_dir=None,
        run_function=['cal_quantile_k'],
        exp_name='baseline',
        summary_file='baseline_summary.csv',
    ):
        self.args = args.copy()
        self.args['baseline_forecast_method'] = self.args.get(
            'baseline_forecast_method', self.args.get('estimate_demand_method', 'ema')
        )
        # Baseline forecasts are independent of the original RL/environment EMA.
        self.args['estimate_demand_method'] = 'ema'
        self.result_dir = result_dir
        self.exp_name = exp_name
        self.summary_file = summary_file

        # Ensure run_function is a list
        if isinstance(run_function, str):
            self.run_function = [run_function]
        else:
            self.run_function = run_function

        # Get parameters from args with defaults
        self.eval_episodes = self.args.get('eval_episodes', 30)
        self.episode_length = self.args.get('episode_length', 50)

        search_param = self.args.get('search_param', {})
        self.search_k_range_start = search_param.get("search_k_range_start", -4)
        self.search_k_range_end = search_param.get("search_k_range_end", 10)
        self.search_k_range_step = search_param.get("search_k_range_step", 0.3)
        self.search_eval_episodes_raw = search_param.get("search_eval_episodes_raw", 5)
        self.search_consecutive_worse_limit = search_param.get("search_consecutive_worse_limit", 2)

        # Create result directory if not exists
        if self.result_dir and not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)

        # Initialize logger for baseline experiments (will be called per function in run())
        self.logger = None
        self.current_function_name = None

    def _save_summary(self, func_name, eval_reward, test_reward):
        """Save summary results to a CSV file."""
        summary_file = self.summary_file
        summary_parent = os.path.dirname(summary_file)
        if summary_parent:
            os.makedirs(summary_parent, exist_ok=True)
        file_exists = os.path.isfile(summary_file)

        # Convert rewards to float if they are numpy arrays/scalars
        eval_reward = float(eval_reward) if hasattr(eval_reward, '__float__') else eval_reward
        test_reward = float(test_reward) if hasattr(test_reward, '__float__') else test_reward

        with open(summary_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    'exp_name', 'run_function', 'forecast_method', 'ema_alpha',
                    'eval_reward', 'test_reward'
                ])
            writer.writerow([
                self.exp_name,
                func_name,
                self.args['baseline_forecast_method'],
                self.args.get('alpha', ''),
                eval_reward,
                test_reward,
            ])


    def _init_logger(self, function_name):
        """Initialize logger for recording baseline evaluation results.

        Args:
            function_name: Name of the function (search, cal_quantile_k, eppen)
        """
        self.current_function_name = function_name

        # Create logger directory under function_name folder
        logger_dir = os.path.join(self.result_dir, function_name) if self.result_dir else function_name
        os.makedirs(logger_dir, exist_ok=True)

        # Create a dummy tensorboard writer that doesn't actually write
        # Since we only want progress.txt and csv files
        log_path = os.path.join(logger_dir, "logs")
        os.makedirs(log_path, exist_ok=True)
        writter = SummaryWriter(log_path)

        # Prepare args for logger
        args_dict = {
            'env': 'multi_lt_transship',
            'algo': 'baseline',
            'exp_name': 'baseline_exp',
            'write_tensorboard_tf': False
        }

        # Prepare algo_args (logger needs some structure)
        algo_args = {
            'train': {
                'episode_length': self.episode_length,
                'n_rollout_threads': 1,  # baseline uses single thread
            },
            'eval': {
                'n_eval_rollout_threads': 1,  # baseline uses single thread
                'eval_episodes': self.eval_episodes,
            }
        }

        # Prepare env_args from self.args
        env_args = {
            'scenario': self.args.get('dataset_name', 'default'),
            'action_type': self.args.get('action_type', 'heu_discrete'),
        }

        # Create logger instance
        self.logger = LOGGER_REGISTRY['multi_lt_transship'](
            args_dict, algo_args, env_args,
            self.args['num_agents'], writter, logger_dir
        )

        # Initialize logger
        self.logger.episode = 0  # Set episode to 0 for baseline
        self.logger.start = time.time()

    def _run_episode(self, env, baseline, mode='eval'):
        """
        Run a single episode and return cumulative rewards.

        Args:
            env: VecTransshipEnv instance
            baseline: VecBaseline or VecEppen instance
            mode: 'eval' or 'test'

        Returns:
            reward_cum: np.ndarray of shape (num_agents,)
        """
        reward_cum = np.zeros(self.args['num_agents'])
        obs = env.reset(mode)[0]

        while True:
            actions = baseline.get_action(env, obs[0])
            obs = env.step(actions)
            rewards = obs[2]
            reward_cum += np.mean(rewards, axis=0)

            if env.step_index[0] >= self.episode_length:
                break

        return reward_cum

    def _run_episode_with_scalar_reward(self, env, baseline, mode='eval'):
        """
        Run a single episode and return both cumulative rewards and scalar average reward.

        Args:
            env: VecTransshipEnv instance
            baseline: VecBaseline or VecEppen instance
            mode: 'eval' or 'test'

        Returns:
            reward_cum: np.ndarray of shape (num_agents,)
            scalar_reward: float, average reward per agent
        """
        reward_cum = np.zeros(self.args['num_agents'])
        scalar_reward = 0.0
        obs = env.reset(mode)[0]

        while True:
            actions = baseline.get_action(env, obs[0])
            obs = env.step(actions)
            rewards = obs[2]
            reward_cum += np.mean(rewards, axis=0)
            scalar_reward += np.mean(rewards) / env.num_agents

            if env.step_index[0] >= self.episode_length:
                break

        return reward_cum, scalar_reward

    def _run_episode_with_logging(self, env, baseline, mode='eval', episode_idx=0):
        """
        Run a single episode with logging support.

        Args:
            env: VecTransshipEnv instance
            baseline: VecBaseline or VecEppen instance
            mode: 'eval' or 'test'
            episode_idx: Index of the episode (for logging)

        Returns:
            reward_cum: np.ndarray of shape (num_agents,)
        """
        # Don't call eval_init here - it should be called once before all episodes

        reward_cum = np.zeros(self.args['num_agents'])
        obs = env.reset(mode)[0]

        step_count = 0
        while True:
            actions = baseline.get_action(env, obs)
            step_result = env.step(actions)

            # Parse step result: [obs, share_obs, rewards, dones, infos, available_actions, lt_today, lt_window]
            obs = step_result[0]
            share_obs = step_result[1] if len(step_result) > 1 else obs
            rewards = step_result[2]
            dones = step_result[3] if len(step_result) > 3 else np.zeros((1, self.args['num_agents']), dtype=bool)
            infos = step_result[4] if len(step_result) > 4 else [[{} for _ in range(self.args['num_agents'])]]
            available_actions = step_result[5] if len(step_result) > 5 else None
            lt_today = step_result[6] if len(step_result) > 6 else None
            lt_window = step_result[7] if len(step_result) > 7 else None

            reward_cum += np.mean(rewards, axis=0)

            # Prepare eval_data for logger (format: tuple of 8 items)
            eval_data = (
                obs,
                share_obs,
                rewards,
                dones,
                infos,
                available_actions,
                lt_today,
                lt_window,
            )

            step_count += 1
            if env.step_index[0] >= self.episode_length:
                break


        return reward_cum

    def _save_args(self, args, result_path):
        """Save default arguments to result file."""
        with open(result_path, 'w') as f:
            # f.write(str(self.args))
            f.write(str(args))
            f.write('\n')

    def _save_results(self, result_path, *results):
        """Save results to result file."""
        with open(result_path, 'a') as f:
            for result in results:
                f.write(str(result))
                f.write('\n')

    def _evaluate_with_k(self, k, mode='eval', baseline_class=VecBaseline, num_episodes=None, stationary_tf=False):
        """
        Evaluate baseline with a specific k value.

        Args:
            k: k parameter value
            mode: 'eval' or 'test'
            baseline_class: VecBaseline or VecEppen class
            num_episodes: Number of episodes to run (default: self.eval_episodes)

        Returns:
            reward_cum: np.ndarray of shape (num_agents,)
        """
        if num_episodes is None:
            num_episodes = self.eval_episodes

        # Initialize logger once for the entire evaluation
        self.logger.eval_init(mode)

        args = self.args.copy()
        args['k'] = k
        args['stationary_tf'] = stationary_tf

        reward_cum = np.zeros(self.args['num_agents'])
        env = VecTransshipEnv(args)
        baseline = baseline_class(args)
        episode_reward = self._run_episode_with_logging(env, baseline, mode, 0)
        reward_cum = episode_reward

        return reward_cum

    def _evaluate_with_k_scalar(self, k, mode='eval', baseline_class=VecBaseline, num_episodes=None, stationary_tf=False):
        """
        Evaluate baseline with a specific k value, returning both vector and scalar rewards.

        Args:
            k: k parameter value
            mode: 'eval' or 'test'
            baseline_class: VecBaseline or VecEppen class
            num_episodes: Number of episodes to run (default: self.eval_episodes)

        Returns:
            reward_cum: np.ndarray of shape (num_agents,)
            scalar_reward: float
        """
        if num_episodes is None:
            num_episodes = self.eval_episodes

        # Initialize logger once for the entire evaluation
        self.logger.eval_init(mode)

        copy_args = self.args.copy()
        copy_args['k'] = k
        copy_args['stationary_tf'] = stationary_tf

        reward_cum = np.zeros(self.args['num_agents'])
        scalar_reward = 0.0
        env = VecTransshipEnv(copy_args)
        baseline = baseline_class(copy_args)
        episode_reward = self._run_episode_with_logging(env, baseline, mode, 0)
        reward_cum = episode_reward
        scalar_reward = np.mean(episode_reward)


        # eval_avg_rewards = self.logger.eval_log(num_episodes)

        return reward_cum, scalar_reward

    def _get_quantile_k(self, env):
        """Calculate quantile k from environment H and B."""
        h, b = env.H, env.B
        k = stats.norm.ppf(b / (b + h))
        return k

    def search(self, result_path=None, stationary_tf=False):
        """
        Search for optimal k value using two-stage search.

        Args:
            result_path: Path to save results (default: auto-generated)
        """
        if result_path is None:
              result_path = os.path.join(self.logger.run_dir, f'search_{self.args["num_agents"]}.txt')

        update_args = self.args.copy()
        update_args.update({'stationary_tf': stationary_tf})
        self._save_args(update_args, result_path)

        # Stage 1: Coarse search
        k_range = np.arange(self.search_k_range_start, self.search_k_range_end, self.search_k_range_step)
        reward_cum_dict = {k: np.zeros(self.args['num_agents']) for k in k_range}
        reward_cum_scalar_dict = {k: -1e7 for k in k_range}
        best_reward = float('-inf')
        consecutive_worse = 0

        for k in k_range:
            reward_cum, scalar_reward = self._evaluate_with_k_scalar(
                k, mode='eval', baseline_class=VecBaseline, num_episodes=self.search_eval_episodes_raw,
                stationary_tf=stationary_tf
            )
            reward_cum_dict[k] = reward_cum
            reward_cum_scalar_dict[k] = scalar_reward

            if scalar_reward > best_reward:
                best_reward = scalar_reward
                consecutive_worse = 0
            else:
                consecutive_worse += 1
                if consecutive_worse >= self.search_consecutive_worse_limit:
                    break

        self._save_results(result_path, reward_cum_dict)

        # Stage 2: Fine search around best k
        best_k = max(reward_cum_scalar_dict, key=reward_cum_scalar_dict.get)
        k_range = np.arange(best_k - 0.55, best_k + 0.55, 0.05)
        reward_cum_dict_2 = {k: np.zeros(self.args['num_agents']) for k in k_range}
        reward_cum_scalar_dict_2 = {k: 0 for k in k_range}

        for k in k_range:
            reward_cum, scalar_reward = self._evaluate_with_k_scalar(
                k, mode='eval', baseline_class=VecBaseline, num_episodes=self.eval_episodes,
                stationary_tf=stationary_tf
            )
            reward_cum_dict_2[k] = reward_cum
            reward_cum_scalar_dict_2[k] = scalar_reward

        self._save_results(result_path, reward_cum_dict_2)

        # Find best k and test
        best_k = max(reward_cum_scalar_dict_2, key=reward_cum_scalar_dict_2.get)
        print(f"Best k: {best_k}")
        print(f"Best reward: {reward_cum_dict_2[best_k]}")

        # Test with best k
        reward_test = self._evaluate_with_k(best_k, mode='test', baseline_class=VecBaseline)
        avg_test_reward = np.average(reward_test)
        print(f"Test reward: {reward_test}, avg test reward: {avg_test_reward}")

        self._save_results(result_path, best_k, reward_test, avg_test_reward)

        return reward_cum_scalar_dict_2[best_k], avg_test_reward

    def cal_quantile_k(self, result_path=None):
        """
        Calculate results using quantile k.

        Args:
            result_path: Path to save results (default: auto-generated)
        """
        if result_path is None:
                result_path = os.path.join(self.logger.run_dir, f'quantile_k_{self.args["num_agents"]}.txt')

        self._save_args(self.args, result_path)

        # Get quantile k
        env = VecTransshipEnv(self.args)
        k = self._get_quantile_k(env)

        # Eval with quantile k
        reward_cum_val = self._evaluate_with_k(k, mode='eval', baseline_class=VecBaseline)
        avg_eval_reward = np.average(reward_cum_val)
        print(f"Eval reward: {reward_cum_val}, avg eval reward: {avg_eval_reward}")
        self._save_results(result_path, reward_cum_val, avg_eval_reward)

        # Test with quantile k
        reward_test = self._evaluate_with_k(k, mode='test', baseline_class=VecBaseline)
        avg_test_reward = np.average(reward_test)
        print(f"Test reward: {reward_test}, avg test reward: {avg_test_reward}")

        self._save_results(result_path, k, reward_test, avg_test_reward)

        return avg_eval_reward, avg_test_reward

    def eppen(self, result_path=None):
        """
        Run VecEppen strategy.

        Args:
            result_path: Path to save results (default: auto-generated)
        """
        if result_path is None:
                result_path = os.path.join(self.logger.run_dir, f'eppen_{self.args["num_agents"]}.txt')

        self._save_args(self.args, result_path)

        # Get quantile k (VecEppen uses quantile k)
        env = VecTransshipEnv(self.args)
        k = self._get_quantile_k(env)

        # Eval with VecEppen
        reward_cum_val = self._evaluate_with_k(k, mode='eval', baseline_class=VecEppen)
        avg_eval_reward = np.average(reward_cum_val)
        print(f"Eval reward: {reward_cum_val}", f"avg eval reward: {avg_eval_reward}")
        self._save_results(result_path, reward_cum_val, avg_eval_reward)

        # Test with VecEppen
        reward_test = self._evaluate_with_k(k, mode='test', baseline_class=VecEppen)
        avg_test_reward = np.average(reward_test)
        print(f"Test reward: {reward_test}", f"avg test reward: {avg_test_reward}")
        self._save_results(result_path, k, reward_test, avg_test_reward)

        return avg_eval_reward, avg_test_reward


    def capped_base_stock(self, result_path=None):
        """Joint eval-only search for target k and order cap; test once selected."""
        from harl.envs.multi_lt_transship.capped_baseline_vec import VecCappedBaseStock

        if result_path is None:
            result_path = os.path.join(
                self.logger.run_dir, f'capped_base_stock_{self.args["num_agents"]}.txt')
        # The environment resets NumPy's global RNG when loading evaluation data.
        # Isolate this new strategy even when followed by legacy run functions.
        rng_state = np.random.get_state()
        try:
            args = self.args.copy()
            args['stationary_tf'] = False
            env = VecTransshipEnv(args)
            if env.backlog_tf:
                raise ValueError('capped_base_stock requires backlog_tf=False (lost sales)')
            self._save_args(args, result_path)
            k_values = np.asarray(args.get('capped_k_values', np.arange(-2, 10.01, 0.5)), dtype=float)
            ratios = list(args.get('capped_cap_ratios', [0.5, 0.75, 0.9, 1., 1.1, 1.25, 1.5, 2., 3.]))
            if k_values.ndim != 1 or not len(k_values) or not np.all(np.isfinite(k_values)):
                raise ValueError('capped_k_values must be a nonempty list of finite numbers')
            ratios = [float(r) for r in ratios if r is not None]
            if any(not np.isfinite(r) or r < 0 for r in ratios):
                raise ValueError('capped_cap_ratios must contain finite nonnegative numbers')
            ratios = list(dict.fromkeys(ratios))
            # Always include the exact existing quantile baseline, and optionally
            # an explicitly supplied base-stock coefficient.
            quantile_k = self._get_quantile_k(env)
            references = [quantile_k]
            if 'capped_reference_k' in args:
                references.append(np.asarray(args['capped_reference_k'], dtype=float))
            if any(not np.all(np.isfinite(k)) for k in references):
                raise ValueError('Reference k must be finite')

            candidates = []
            cache = {}
            def evaluate(k, ratio):
                key = (tuple(np.asarray(k).reshape(-1)), ratio)
                if key in cache:
                    return cache[key]
                policy_args = {**args, 'k': k, 'capped_cap_ratio': ratio}
                policy = VecCappedBaseStock(policy_args)
                reward = self._run_episode_with_logging(env, policy, 'eval')
                score = float(np.mean(reward))
                if not np.isfinite(score):
                    raise ValueError('Non-finite capped base-stock evaluation reward')
                record = {'k': np.asarray(k).tolist(), 'cap_ratio': ratio,
                          'eval_reward': score, 'eval_reward_by_agent': reward.tolist(),
                          'actual_order_cap': policy.actual_order_cap,
                          'cap_binding_count': policy.cap_binding_count,
                          'cap_decision_count': policy.cap_decision_count,
                          'cap_binding_fraction': policy.cap_binding_count / max(1, policy.cap_decision_count)}
                candidates.append(record)
                cache[key] = record
                if len(candidates) % 25 == 0:
                    print(f'Capped base-stock: evaluated {len(candidates)} candidates', flush=True)
                return record

            # Uncapped candidates first: ties favor the simpler existing policy.
            for k in references + list(k_values):
                evaluate(k, None)
            for ratio in ratios:
                for k in references + list(k_values):
                    evaluate(k, ratio)
            def rank(row):
                return row['eval_reward'], row['cap_ratio'] is None

            best = max(candidates, key=rank)
            # Refine several cap regimes, including the uncapped baseline.
            # Coordinate sweeps keep the budget bounded; they are not a proof
            # of global optimality for this nonconvex, integer-action objective.
            if args.get('capped_refine', True):
                seeds = [max((r for r in candidates if r['cap_ratio'] is None), key=rank)]
                cap_seeds = [max((r for r in candidates if r['cap_ratio'] == ratio), key=rank)
                             for ratio in ratios]
                seeds += sorted(cap_seeds, key=rank, reverse=True)[:3]
                for seed in seeds:
                    point = seed
                    for _ in range(2):
                        k_neighbors = [evaluate(np.asarray(point['k']) + delta, point['cap_ratio'])
                                       for delta in np.round(np.arange(-0.5, 0.501, 0.05), 8)]
                        point = max([point] + k_neighbors, key=rank)
                        if point['cap_ratio'] is not None:
                            # Cover gaps between adjacent coarse caps, rather
                            # than only +/-0.1 around one winning grid point.
                            below = [r for r in ratios if r < seed['cap_ratio']]
                            above = [r for r in ratios if r > seed['cap_ratio']]
                            lower = max(below) if below else max(0., seed['cap_ratio'] - 0.5)
                            upper = min(above) if above else seed['cap_ratio'] + 0.5
                            fine_ratios = sorted(set([lower, upper, point['cap_ratio']]
                                                    + np.round(np.arange(lower, upper, 0.05), 8).tolist()))
                            cap_neighbors = [evaluate(np.asarray(point['k']), ratio)
                                             for ratio in fine_ratios + [None]]
                            point = max([point] + cap_neighbors, key=rank)
                best = max(candidates, key=rank)
            uncapped = max((r for r in candidates if r['cap_ratio'] is None),
                           key=lambda row: row['eval_reward'])
            selection = {'selected': best, 'best_uncapped_eval': uncapped,
                         'search_version': 2,
                         'refinement': 'two coordinate rounds from best uncapped and top three coarse cap regimes',
                         'cap_basis': args.get('capped_cap_basis', 'reference'),
                         'selection_split': 'eval', 'candidates': candidates}
            # Persist selection BEFORE accessing test; test cannot change it.
            json_path = os.path.splitext(result_path)[0] + '.json'
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(selection, f, indent=2, allow_nan=False)
            policy = VecCappedBaseStock({**args, 'k': np.asarray(best['k']),
                                        'capped_cap_ratio': best['cap_ratio']})
            reward_test = self._run_episode_with_logging(env, policy, 'test')
            test_score = float(np.mean(reward_test))
            selection.update(test_reward=test_score, test_reward_by_agent=reward_test.tolist())
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(selection, f, indent=2, allow_nan=False)
            self._save_results(result_path, best, reward_test, test_score)
            print(f'Capped base-stock selected: {best}; test reward: {test_score}', flush=True)
            return best['eval_reward'], test_score
        finally:
            np.random.set_state(rng_state)

    def run(self, result_path=None):
        """
        Run the specified function(s).

        Args:
            result_path: Path to save results (optional, can be a dict mapping function names to paths)
        """
        valid_functions = ['search', 'cal_quantile_k', 'eppen', 'sta_search', 'capped_base_stock']

        results = {}
        for func_name in self.run_function:
            if func_name not in valid_functions:
                raise ValueError(f"Unknown run_function: {func_name}. Must be one of {valid_functions}")

            # Initialize logger for this function
            self._init_logger(func_name)

            # Get result path for this function
            if result_path is None:
                func_result_path = None
            elif isinstance(result_path, dict):
                func_result_path = result_path.get(func_name, None)
            else:
                func_result_path = result_path

            # Run the function and capture rewards
            eval_reward, test_reward = 0, 0
            if func_name == 'search':
                eval_reward, test_reward = self.search(func_result_path)
            elif func_name == 'cal_quantile_k':
                eval_reward, test_reward = self.cal_quantile_k(func_result_path)
            elif func_name == 'eppen':
                eval_reward, test_reward = self.eppen(func_result_path)
            elif func_name == 'sta_search':
                eval_reward, test_reward = self.search(func_result_path, stationary_tf=True)
            elif func_name == 'capped_base_stock':
                eval_reward, test_reward = self.capped_base_stock(func_result_path)

            # Save results to summary
            self._save_summary(func_name, eval_reward, test_reward)
            results[func_name] = {
                'eval_reward': float(eval_reward),
                'test_reward': float(test_reward),
            }

        return results


def main():
    """Main function to run baseline experiments."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--exp_name", type=str, default="baseline_test", help="Experiment name."
    )
    parser.add_argument(
        "--load_config",
        type=str,
        default="",
        help="If set, load existing experiment config file.",
    )
    parser.add_argument(
        "--load_change_config",
        type=str,
        default='',
        help="Modify config on top of load_config. Multiple paths separated by comma.",
    )
    parser.add_argument(
        "--run_function",
        type=str,
        default='cal_quantile_k',
        help="Functions to run, separated by comma. Choices: search, cal_quantile_k, eppen, sta_search, capped_base_stock",
    )

    parser.add_argument(
        "--estimate_demand_method", "--baseline_forecast_method",
        choices=("ema", "arima", "xgboost"), default=None,
        help="Heuristic demand forecast; overrides the configuration when specified.",
    )

    args, unparsed_args = parser.parse_known_args()

    # Default arguments (base)
    default_args = {
        'action_type': "heu_discrete",
        'num_agents':3,
        'k': 0.629,
        'reactive_tf': False,
        'payment_type': 'selling_first',
        'num_envs': 20,
        'dataset_name': 'default',
        'transship_type': 'force_to_transship',
        'product_allocation_method': {'allocate_func': 'anup', 'allocate_args': {'ratio_tf': True}},
        'eval_episodes': 20,
        'episode_length': 50,
        # Fixed heuristic EMA weight; environment observations have their own EMA.
        'estimate_demand_method': 'ema',
        'alpha': 0.3,
        # Optional Eppen-only extension; estimate correlation from gen_source.
        'eppen_use_demand_correlation': False,
        'eppen_correlation_train_length': None,
        'search_param': {
            'search_k_range_start': -2,
            'search_k_range_end': 10,
            'search_k_range_step': 0.1,
            'search_eval_episodes_raw': 5,
            'search_consecutive_worse_limit': 2
        }
    }

    # Process unparsed args into a dict
    def process(arg):
        try:
            return ast.literal_eval(arg)
        except:
            return arg

    keys = [k[2:] for k in unparsed_args[0::2]]
    values = [process(v) for v in unparsed_args[1::2]]
    unparsed_dict = {k: v for k, v in zip(keys, values)}

    # Start with default_args
    final_args = default_args.copy()


    if args.load_config != "":
        with open(args.load_config, encoding="utf-8") as file:
            all_config = json.load(file)
        # Merge parts into final_args
        if "main_args" in all_config:
            final_args = deep_update(final_args, all_config["main_args"])
        if "env_args" in all_config:
            final_args = deep_update(final_args, all_config["env_args"])
        if "algo_args" in all_config:
            if "train" in all_config["algo_args"]:
                final_args = deep_update(final_args, all_config["algo_args"]["train"])
            if "eval" in all_config["algo_args"]:
                final_args = deep_update(final_args, all_config["algo_args"]["eval"])

    # Apply load_change_config
    if args.load_change_config != "":
        change_config_paths = [path.strip() for path in args.load_change_config.split(',')]
        for change_config_path in change_config_paths:
            if os.path.exists(change_config_path):
                with open(change_config_path, encoding="utf-8") as file:
                    change_config = json.load(file)
                # Merge parts into final_args
                if "main_args" in change_config:
                    final_args = deep_update(final_args, change_config["main_args"])
                if "env_args" in change_config:
                    final_args = deep_update(final_args, change_config["env_args"])
                if "algo_args" in change_config:
                    if "train" in change_config["algo_args"]:
                        final_args = deep_update(final_args, change_config["algo_args"]["train"])
                    if "eval" in change_config["algo_args"]:
                        final_args = deep_update(final_args, change_config["algo_args"]["eval"])
            else:
                raise FileNotFoundError(change_config_path)

    # Apply unparsed command line arguments
    final_args = deep_update(final_args, unparsed_dict)

    if args.estimate_demand_method is not None:
        final_args['estimate_demand_method'] = args.estimate_demand_method
        final_args['baseline_forecast_method'] = args.estimate_demand_method

    # Keep baseline evaluation conditions independent of training configs.
    # Baseline policies output actual order quantities, and the reference
    # comparison uses 20 vectorized environments with Anup allocation.
    baseline_args = {
        'action_type': 'heu_discrete',
        'num_envs': 20,
        'transship_type': 'force_to_transship',
        'product_allocation_method': {
            'allocate_func': 'anup',
            'allocate_args': {'ratio_tf': True},
        },
    }
    for key, baseline_value in baseline_args.items():
        configured_value = final_args.get(key)
        if configured_value != baseline_value:
            print(
                f"Ignoring configured {key}={configured_value!r}; "
                f"baseline evaluation always uses {baseline_value!r}."
            )
        final_args[key] = baseline_value

    # Set run functions
    run_functions = [f.strip() for f in args.run_function.split(',')]

    # Setup result directory
    num_agents = final_args.get('num_agents', 3)
    reactive_tf = final_args.get('reactive_tf', False)
    allocate_func = final_args.get('product_allocation_method', {}).get('allocate_func', 'anup')

    result_dir = 'results_heu/heu{}{}{}{}'.format(
        '_agent' + str(num_agents),
        '' if reactive_tf else '_proactive',
        '_' + allocate_func,
        time.strftime('_%m%d_%H%M%S', time.localtime())
    )

    # Run experiment
    runner = BaselineExperiment(
        args=final_args,
        result_dir=result_dir,
        run_function=run_functions,
        exp_name=args.exp_name
    )
    runner.run()



if __name__ == "__main__":
    main()
