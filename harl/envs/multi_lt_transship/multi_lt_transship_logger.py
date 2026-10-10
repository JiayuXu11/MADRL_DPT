import os
import time

import numpy as np
from harl.common.base_logger import BaseLogger
from harl.utils.basic_tools_wo_torch import del_folder, filter_scalar_dict

NUM_LOG = 2

class MultiLtTransshipLogger(BaseLogger):
    '''
    Multi-LT Transshipment
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 在类的init里import，防止异步时import占用过多内存
        from tensorboardX import SummaryWriter
        self.SummaryWriter = SummaryWriter

    def get_task_name(self):
        return f"{self.env_args['scenario']}-{self.env_args['action_type']}"

    def clear_tensorboard(self, usage):
        '''
        清空微观数据的tensorboard记录，防止数据混乱
        usage: eval/test
        '''
        if not self.write_tensorboard_tf:
            return

        self.writter.flush()
        self.writter.close()
        log_dir = self.writter.logdir
        dir_name = usage + '_{}'
        for i in range(NUM_LOG):
            del_folder(os.path.join(log_dir, dir_name.format(str(i))))
        self.writter = self.SummaryWriter(log_dir)

    def eval_init(self, usage):
        '''
        识别是测试/验证，重置对应writer + 初始化一些变量
        '''
        self.usage = usage
        self.clear_tensorboard(usage)
        # 用于记录微观数据的tensorboard名格式
        self.micro_fig_name = usage + '_{}/agent{}'
        self.total_num_steps = (
            self.episode
            * self.algo_args["train"]["episode_length"]
            * self.algo_args["train"]["n_rollout_threads"]
        )
        self.eval_step = 0
        self.eval_episode_rewards = []
        self.one_episode_rewards = []
        self.eval_episode_info = []
        self.one_episode_info = []
        for eval_i in range(self.algo_args["eval"]["n_eval_rollout_threads"]):
            self.one_episode_rewards.append([])
            self.eval_episode_rewards.append([])
            self.one_episode_info.append([])
            self.eval_episode_info.append([])

    def eval_per_step(self, eval_data):
        '''
        更新存储的一些变量 + 记录微观数据
        '''
        # Handle both old format (6 items) and new format (8 items)
        (
            eval_obs,
            eval_share_obs,
            eval_rewards,
            eval_dones,
            eval_infos,
            eval_available_actions,
            _,
            _,
        ) = eval_data

        for eval_i in range(self.algo_args["eval"]["n_eval_rollout_threads"]):
            self.one_episode_rewards[eval_i].append(eval_rewards[eval_i])
            self.one_episode_info[eval_i].append(eval_infos[eval_i])

        # 记录前NUM_LOG个rollout的微观数据
        if self.write_tensorboard_tf:
            for i in range(min(NUM_LOG, len(eval_infos))):
                for agent_id in range(self.num_agents):
                    self.writter.add_scalars(
                        self.micro_fig_name.format(str(i), str(agent_id)),
                        filter_scalar_dict(eval_infos[i][agent_id]),
                        self.eval_step,
                    )
        self.eval_step += 1

    def eval_thread_done(self, tid):
        """Log evaluation information."""
        # eval_episode_rewards: (n_eval_rollout_threads, n_agents, dim)
        self.eval_episode_rewards[tid].append(
            np.sum(self.one_episode_rewards[tid], axis=0)
        )
        # 把对应的eval_episode_info做个汇总, rollouts, eval_episodes/rollout, agents, info
        # one_episode_info[tid]格式为：[(dict1, dict2, dict3), (dict1, dict2, dict3), ...] 每个代表一个step的信息，每个dict代表一个agent的信息（里面的数是mean）
        agents_info = [{} for _ in range(self.num_agents)]
        for agent_id in range(self.num_agents):
            for k in self.one_episode_info[tid][0][agent_id].keys():
                dict_mean = {k: np.mean([info[agent_id][k] for info in self.one_episode_info[tid]])}
                # 对于transship的信息，需要记录pos/neg部分分别求平均
                if 'transship' in k:
                    dict_mean.update({k + '_pos': np.mean([max(info[agent_id][k], 0) for info in self.one_episode_info[tid]]),
                                 k + '_neg': np.mean([- min(info[agent_id][k], 0) for info in self.one_episode_info[tid]])})
                agents_info[agent_id].update(dict_mean)
        # 若为测试集，则存储所有的信息到csv里
        if self.usage == 'test':
            test_info_dir = os.path.join(self.run_dir, 'test_info')
            if not os.path.exists(test_info_dir):
                os.makedirs(test_info_dir)
            with open(os.path.join(test_info_dir, f'test_info_{tid}.csv'), 'w') as f:
                episode_info_to_save = self.one_episode_info[tid]
                columns = episode_info_to_save[0][0].keys()
                f.write('time_index,agent_id,' + ','.join(columns) + '\n')
                for time_index, all_agent_info in enumerate(episode_info_to_save):
                    for agent_id, info in enumerate(all_agent_info):
                        f.write(str(time_index) + ',' + str(agent_id) + ',' + ','.join([str(info[k]) for k in columns]) + '\n')

        self.eval_episode_info[tid].append(agents_info)
        self.one_episode_rewards[tid] = []
        self.one_episode_info[tid] = []

    def eval_log(self, eval_episode):
        """Log evaluation information."""
        # 已消耗时间
        time_consumed = time.time() - self.start
        time_consumed_str = '{}min'.format(round(time_consumed / 60, 2))
        self.eval_episode_rewards = np.concatenate(
            [rewards for rewards in self.eval_episode_rewards if rewards]
        )
        eval_env_infos = {
            "eval_average_episode_rewards": np.average(self.eval_episode_rewards, axis=(0)).flatten(),
            "eval_max_episode_rewards": np.max(self.eval_episode_rewards, axis=(0)).flatten(),
        }
        # 更新eval_env_infos其他信息
        for k in self.eval_episode_info[0][0][0].keys():
            eval_env_infos['eval_average_' + k] = np.array([np.mean([info[agent_id][k]
                                                            for rollout_info in self.eval_episode_info
                                                            for info in rollout_info])
                                                            for agent_id in range(self.num_agents)])
        eval_env_infos['eval_average_transship_pos_fillrate'] = (eval_env_infos['eval_average_' + 'transship' + '_pos'] /
                                                                 np.maximum(eval_env_infos['eval_average_' + 'transship_intend' + '_pos'], 1e-5))
        eval_env_infos['eval_average_transship_neg_fillrate'] = (eval_env_infos['eval_average_' + 'transship' + '_neg'] /
                                                                    np.maximum(eval_env_infos['eval_average_' + 'transship_intend' + '_neg'], 1e-5))
        self.log_env_agent(eval_env_infos)
        eval_avg_rew = eval_env_infos["eval_average_episode_rewards"]
        print("Evaluation average episode reward is {}.\n".format(eval_avg_rew))
        self.log_file.write(
            ",".join(map(str, [self.total_num_steps, time_consumed_str, eval_avg_rew])) + "\n"
        )
        # 如果是测试集，则把所有信息都记录到txt里
        if self.usage == 'test':
            self.log_file.write("average_system_cost: " + str(np.mean(eval_env_infos['eval_average_episode_rewards'])) + '\n')
            self.log_file.write(str(eval_env_infos) + '\n')
        self.log_file.flush()
        return eval_avg_rew
