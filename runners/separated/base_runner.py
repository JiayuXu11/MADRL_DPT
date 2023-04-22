    
import time
import os
import numpy as np
from itertools import chain
import torch
from tensorboardX import SummaryWriter
from utils.separated_buffer import SeparatedReplayBuffer
from utils.util import update_linear_schedule,del_folder

def _t2n(x):
    return x.detach().cpu().numpy()

class Runner(object):
    def __init__(self, config):

        self.all_args = config['all_args']
        self.envs = config['envs']
        self.eval_envs = config['eval_envs']
        self.device = config['device']
        self.num_agents = config['num_agents']

        # parameters
        self.env_name = self.all_args.env_name
        self.algorithm_name = self.all_args.algorithm_name
        self.experiment_name = self.all_args.experiment_name
        # self.use_centralized_V = self.all_args.use_centralized_V
        # self.instant_info_sharing = self.all_args.instant_info_sharing or self.all_args.central_controller
        self.delayed_info_sharing = self.all_args.delayed_info_sharing
        self.central_controller = self.all_args.central_controller
        self.use_obs_instead_of_state = self.all_args.use_obs_instead_of_state
        self.num_env_steps = self.all_args.num_env_steps
        self.episode_length = self.all_args.episode_length
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads
        self.use_linear_lr_decay = self.all_args.use_linear_lr_decay
        self.hidden_size_critic = self.all_args.hidden_size_critic
        self.hidden_size = self.all_args.hidden_size
        self.use_render = self.all_args.use_render
        self.recurrent_N = self.all_args.recurrent_N
        self.use_single_network = self.all_args.use_single_network
        # interval
        self.save_interval = self.all_args.save_interval
        self.use_eval = self.all_args.use_eval
        self.eval_interval = self.all_args.eval_interval
        self.log_interval = self.all_args.log_interval
        self.n_warmup_evaluations = self.all_args.n_warmup_evaluations
        self.n_no_improvement_thres = self.all_args.n_no_improvement_thres
        self.num_involver=self.all_args.num_involver
        # dir
        self.model_dir = self.all_args.model_dir

        # eval or test mean demand
        self.demand_mean_val=self.get_mean_demand(self.all_args.eval_dir)
        self.demand_mean_test=self.get_mean_demand(self.all_args.test_dir)


        if self.use_render:
            self.run_dir = config["run_dir"]
            self.gif_dir = str(self.run_dir / 'gifs')
            if not os.path.exists(self.gif_dir):
                os.makedirs(self.gif_dir)
        else:
            self.run_dir = config["run_dir"]
            self.log_dir = str(self.run_dir / 'logs')
            
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
            # self.log_dir=self.log_dir
            self.writter = SummaryWriter(self.log_dir.replace('\\','/'))
            
            self.save_dir = str(self.run_dir / 'models')
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)


        if self.all_args.algorithm_name == "happo":
            from algorithms.happo_trainer import HAPPO as TrainAlgo
            from algorithms.happo_policy import HAPPO_Policy as Policy
        elif self.all_args.algorithm_name == "heuristic1":
            from algorithms.happo_trainer import HAPPO as TrainAlgo
            from algorithms.happo_policy import HAPPO_Policy as Policy
        elif self.all_args.algorithm_name == "heuristic2":
            from algorithms.happo_trainer import HAPPO as TrainAlgo
            from algorithms.heuristic2_policy import Heuristic_Policy as Policy
        else:
            raise NotImplementedError

        print("critic_observation_space: ", self.envs.observation_space_critic)
        print("actor_observation_space: ", self.envs.observation_space)
        print("action_space: ", self.envs.action_space)

        self.policy = []
        for agent_id in range(self.num_agents):
            critic_observation_space = self.envs.observation_space_critic[agent_id]
            policy_observation_space = self.envs.observation_space[agent_id]
            # policy network
            po = Policy(self.all_args,
                        policy_observation_space,
                        critic_observation_space,
                        self.envs.action_space[agent_id],
                        device = self.device)
            self.policy.append(po)

        if self.model_dir is not None:
            self.restore()

        self.trainer = []
        self.buffer = []
        for agent_id in range(self.num_agents):
            # algorithm
            tr = TrainAlgo(self.all_args, self.policy[agent_id], device = self.device)
            
            # buffer
            critic_observation_space = self.envs.observation_space_critic[agent_id]
            policy_observation_space = self.envs.observation_space[agent_id]
            bu = SeparatedReplayBuffer(self.all_args,
                                       policy_observation_space,
                                       critic_observation_space,
                                       self.envs.action_space[agent_id])
            self.buffer.append(bu)
            self.trainer.append(tr)

        if self.model_dir is not None:
            self.restore_normalizer()
    # 给需求路径，算平均需求
    def get_mean_demand(self,dir):
        path = [dir+'/{}/'.format(i) for i in range(self.agent_num)]
        files_0 = os.listdir(path[0])
        n_eval = len(files_0)
        sum_demand=0
        for i in range(n_eval):
            eval_data_i=[]
            for j in range(len(path)):
                files=os.listdir(path[j])
                data = []
                with open(path[j] + files[i], "rb") as f:
                    lines = f.readlines()
                    for line in lines[:self.episode_length]:
                        sum_demand+=int(line)
        mean_demand=sum_demand/n_eval/len(path)/self.episode_length
        return mean_demand

    def run(self):
        raise NotImplementedError

    def warmup(self):
        raise NotImplementedError

    def collect(self, step):
        raise NotImplementedError

    def insert(self, data):
        raise NotImplementedError
    
    @torch.no_grad()
    def compute(self):
        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_rollout()
            next_value = self.trainer[agent_id].policy.get_values(self.buffer[agent_id].share_obs[-1], 
                                                                self.buffer[agent_id].rnn_states_critic[-1],
                                                                self.buffer[agent_id].masks[-1])
            next_value = _t2n(next_value)
            self.buffer[agent_id].compute_returns(next_value, self.trainer[agent_id].value_normalizer)

    def train(self):
        train_infos = []
        # random update order

        action_dim=self.buffer[0].actions.shape[-1]
        factor = np.ones((self.episode_length, self.n_rollout_threads, 1), dtype=np.float32)

        for agent_id in torch.randperm(self.num_agents):
            self.trainer[agent_id].prep_training()
            self.buffer[agent_id].update_factor(factor)
            available_actions = None if self.buffer[agent_id].available_actions is None \
                else self.buffer[agent_id].available_actions[:-1].reshape(-1, *self.buffer[agent_id].available_actions.shape[2:])
            
            old_actions_logprob, _ =self.trainer[agent_id].policy.actor.evaluate_actions(self.buffer[agent_id].obs[:-1].reshape(-1, *self.buffer[agent_id].obs.shape[2:]),
                                                            self.buffer[agent_id].rnn_states[0:1].reshape(-1, *self.buffer[agent_id].rnn_states.shape[2:]),
                                                            self.buffer[agent_id].actions.reshape(-1, *self.buffer[agent_id].actions.shape[2:]),
                                                            self.buffer[agent_id].masks[:-1].reshape(-1, *self.buffer[agent_id].masks.shape[2:]),
                                                            available_actions,
                                                            self.buffer[agent_id].active_masks[:-1].reshape(-1, *self.buffer[agent_id].active_masks.shape[2:]))
            train_info = self.trainer[agent_id].train(self.buffer[agent_id])

            new_actions_logprob, _ =self.trainer[agent_id].policy.actor.evaluate_actions(self.buffer[agent_id].obs[:-1].reshape(-1, *self.buffer[agent_id].obs.shape[2:]),
                                                            self.buffer[agent_id].rnn_states[0:1].reshape(-1, *self.buffer[agent_id].rnn_states.shape[2:]),
                                                            self.buffer[agent_id].actions.reshape(-1, *self.buffer[agent_id].actions.shape[2:]),
                                                            self.buffer[agent_id].masks[:-1].reshape(-1, *self.buffer[agent_id].masks.shape[2:]),
                                                            available_actions,
                                                            self.buffer[agent_id].active_masks[:-1].reshape(-1, *self.buffer[agent_id].active_masks.shape[2:]))

            factor = factor*_t2n(torch.prod(torch.exp(new_actions_logprob-old_actions_logprob),dim=-1).reshape(self.episode_length,self.n_rollout_threads,1))
            train_infos.append(train_info)      
            self.buffer[agent_id].after_update()

        return train_infos

    def save(self):
        for agent_id in range(self.num_agents):
            if self.use_single_network:
                policy_model = self.trainer[agent_id].policy.model
                torch.save(policy_model.state_dict(), str(self.save_dir) + "/model_agent" + str(agent_id) + ".pt")
            else:
                policy_actor = self.trainer[agent_id].policy.actor
                torch.save(policy_actor.state_dict(), str(self.save_dir) + "/actor_agent" + str(agent_id) + ".pt")
                policy_critic = self.trainer[agent_id].policy.critic
                torch.save(policy_critic.state_dict(), str(self.save_dir) + "/critic_agent" + str(agent_id) + ".pt")
            if self.trainer[agent_id].value_normalizer:
                torch.save(self.trainer[agent_id].value_normalizer.state_dict(), str(self.save_dir) + "/value_normalizer" +  ".pt")

    def restore(self):
        for agent_id in range(self.num_agents):
            if self.use_single_network:
                policy_model_state_dict = torch.load(str(self.model_dir) + '/model_agent' + str(agent_id) + '.pt')
                self.policy[agent_id].model.load_state_dict(policy_model_state_dict)
            else:
                policy_actor_state_dict = torch.load(str(self.model_dir) + '/actor_agent' + str(agent_id) + '.pt')
                self.policy[agent_id].actor.load_state_dict(policy_actor_state_dict)
                policy_critic_state_dict = torch.load(str(self.model_dir) + '/critic_agent' + str(agent_id) + '.pt')
                self.policy[agent_id].critic.load_state_dict(policy_critic_state_dict)
            
    def restore_normalizer(self):
        for agent_id in range(self.num_agents):
            # a=self.trainer[agent_id].value_normalizer.state_dict()
            if self.trainer[agent_id].value_normalizer and os.path.exists(str(self.model_dir) + '/value_normalizer' +  '.pt'):
                value_normalizer_state_dict = torch.load(str(self.model_dir) + '/value_normalizer' +  '.pt')
                self.trainer[agent_id].value_normalizer.load_state_dict(value_normalizer_state_dict)

    def log_train(self, train_infos, total_num_steps): 
        for agent_id in range(self.num_agents):
            for k, v in train_infos[agent_id].items():
                agent_k = "agent%i/" % agent_id + k
                self.writter.add_scalars(agent_k, {agent_k: v}, total_num_steps)

    def log_env(self, env_infos, total_num_steps):
        for k, v in env_infos.items():
            if len(v) > 0:
                self.writter.add_scalars(k, {k: np.mean(v)}, total_num_steps)

    # 清除tensorboard数据
    def clear_tensorboard(self,test_tf):
        self.writter.flush()
        self.writter.close()
        dir_name='test_{}' if test_tf else 'eval_{}'
        for i in range(self.num_agents):
            log_dir=self.log_dir
            dir=os.path.join(log_dir,dir_name.format(str(i)))
            del_folder(dir)
        self.writter=SummaryWriter(log_dir)

