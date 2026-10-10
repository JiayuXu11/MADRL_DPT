"""Base logger."""

import time
import os
import numpy as np


class BaseLogger:
    """Base logger class.
    Used for logging information in the on-policy training pipeline.
    """

    def __init__(self, args, algo_args, env_args, num_agents, writter, run_dir):
        """Initialize the logger.

        Args:
            args: General arguments
            algo_args: Algorithm-specific arguments
            env_args: Environment-specific arguments
            num_agents: Number of agents
            writter: Tensorboard writer instance
            run_dir: Directory for saving logs
        """
        self.args = args
        self.algo_args = algo_args
        self.env_args = env_args
        self.task_name = self.get_task_name()
        self.num_agents = num_agents
        self.writter = writter
        self.run_dir = run_dir
        self.log_file = open(
            os.path.join(run_dir, "progress.txt"), "w", encoding="utf-8"
        )
        self.write_tensorboard_tf = args.get('write_tensorboard_tf', True)

        # Initialize attributes that were previously uninitialized
        self.start = 0
        self.end = 0
        self.episodes = 0
        self.episode = 0
        self.total_num_steps = 0
        self.train_episode_rewards = None
        self.done_episodes_rewards = []
        self.eval_episode_rewards = []
        self.one_episode_rewards = []
        self.eval_infos = None

    def get_task_name(self):
        """Get the task name."""
        raise NotImplementedError

    def init(self, episodes):
        """Initialize the logger."""
        self.start = time.time()
        self.episodes = episodes
        self.train_episode_rewards = np.zeros(
            (self.algo_args["train"]["n_rollout_threads"],
            self.num_agents)
        )
        self.done_episodes_rewards = []

    def episode_init(self, episode):
        """Initialize the logger for each episode."""
        self.episode = episode

    def per_step(self, data):
        """Process data per step."""
        (
            obs,
            share_obs,
            rewards,
            dones,
            infos,
            available_actions,
            values,
            actions,
            action_log_probs,
            rnn_states,
            rnn_states_critic,
            *extra,
        ) = data
        self._latest_lt_today = extra[0] if len(extra) > 0 else None
        self._latest_lt_window = extra[1] if len(extra) > 1 else None
        dones_env = np.all(dones, axis=1)
        reward_env = np.squeeze(rewards)
        self.train_episode_rewards += reward_env
        for t in range(self.algo_args["train"]["n_rollout_threads"]):
            if dones_env[t]:
                self.done_episodes_rewards.append(self.train_episode_rewards[t].copy())
                self.train_episode_rewards[t] = 0

    def episode_log(
        self, actor_train_infos, critic_train_infos, actor_buffer, critic_buffer
    ):
        """Log information for each episode."""
        self.total_num_steps = (
            self.episode
            * self.algo_args["train"]["episode_length"]
            * self.algo_args["train"]["n_rollout_threads"]
        )
        self.end = time.time()
        print(
            "Env {} Task {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.".format(
                self.args["env"],
                self.task_name,
                self.args["algo"],
                self.args["exp_name"],
                self.episode,
                self.episodes,
                self.total_num_steps,
                self.algo_args["train"]["num_env_steps"],
                int(self.total_num_steps / (self.end - self.start)),
            )
        )
        for critic_buffer_id in range(len(critic_buffer)):
            critic_train_infos[critic_buffer_id]["average_step_rewards"] = critic_buffer[critic_buffer_id].get_mean_rewards()
            print(
                "Average step reward for agent{} is {}.".format(
                    critic_buffer_id,
                    critic_train_infos[critic_buffer_id]["average_step_rewards"]
                )
            )
        self.log_train(actor_train_infos, critic_train_infos)



        if len(self.done_episodes_rewards) > 0:
            aver_episode_rewards = np.mean(self.done_episodes_rewards, axis=0).flatten()
            print(
                "Some episodes done, average episode reward is {}.\n".format(
                    aver_episode_rewards
                )
            )
            self.log_env_agent({"average_episode_rewards": aver_episode_rewards})
            self.done_episodes_rewards = []

    def eval_init(self, usage=None):
        """Initialize the logger for evaluation."""
        self.total_num_steps = (
            self.episode
            * self.algo_args["train"]["episode_length"]
            * self.algo_args["train"]["n_rollout_threads"]
        )
        self.eval_episode_rewards = []
        self.one_episode_rewards = []
        for eval_i in range(self.algo_args["eval"]["n_eval_rollout_threads"]):
            self.one_episode_rewards.append([])
            self.eval_episode_rewards.append([])

    def eval_per_step(self, eval_data):
        """Log evaluation information per step."""
        (
            eval_obs,
            eval_share_obs,
            eval_rewards,
            eval_dones,
            eval_infos,
            eval_available_actions,
            *extra,
        ) = eval_data
        self._latest_eval_lt_today = extra[0] if len(extra) > 0 else None
        self._latest_eval_lt_window = extra[1] if len(extra) > 1 else None
        for eval_i in range(self.algo_args["eval"]["n_eval_rollout_threads"]):
            self.one_episode_rewards[eval_i].append(eval_rewards[eval_i])
        self.eval_infos = eval_infos

    def eval_thread_done(self, tid):
        """Log evaluation information."""
        self.eval_episode_rewards[tid].append(
            np.sum(self.one_episode_rewards[tid], axis=0)
        )
        self.one_episode_rewards[tid] = []

    def eval_log(self, eval_episode):
        """Log evaluation information."""
        self.eval_episode_rewards = np.concatenate(
            [rewards for rewards in self.eval_episode_rewards if rewards]
        )
        eval_env_infos = {
            "eval_average_episode_rewards": np.average(self.eval_episode_rewards, axis=(0, 2)).flatten(),
            "eval_max_episode_rewards": np.max(self.eval_episode_rewards, axis=(0, 2)).flatten(),
        }
        self.log_env_agent(eval_env_infos)
        eval_avg_rew = eval_env_infos["eval_average_episode_rewards"]
        print("Evaluation average episode reward is {}.\n".format(eval_avg_rew))
        self.log_file.write(
            ",".join(map(str, [self.total_num_steps, eval_avg_rew])) + "\n"
        )
        self.log_file.flush()

    def log_train(self, actor_train_infos, critic_train_infos):
        """Log training information."""
        if not self.write_tensorboard_tf:
            return

        # log actor
        for agent_id in range(self.num_agents):
            for k, v in actor_train_infos[agent_id].items():
                agent_k = "agent%i/" % agent_id + k
                self.writter.add_scalar(agent_k, v, self.total_num_steps)
        # log critic
        for critic_train_info_id in range(len(critic_train_infos)):
            for k, v in critic_train_infos[critic_train_info_id].items():
                critic_k = "critic%i/" % critic_train_info_id + k
                self.writter.add_scalar(critic_k, v, self.total_num_steps)

    def log_env(self, env_infos):
        """Log environment information."""
        if not self.write_tensorboard_tf:
            return

        for k, v in env_infos.items():
            if len(v) > 0:
                self.writter.add_scalar(k, np.mean(v), self.total_num_steps)

    def close(self):
        """Close the logger."""
        self.log_file.close()

    def log_env_agent(self, env_infos):
        """
        Log environment information for each agent.
        env_infos的格式为：{"key1": [agent1_value, agent2_value, ...], "key2": [agent1_value, agent2_value, ...], ...}
        """
        if not self.write_tensorboard_tf:
            return

        for agent_id in range(self.num_agents):
            agent_k = "agent%i/" % agent_id + 'env'
            info_dict = {k: v[agent_id] for k, v in env_infos.items()}
            self.writter.add_scalars(agent_k, info_dict, self.total_num_steps)
