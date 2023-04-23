import time
import numpy as np
from functools import reduce
import torch
from itertools import chain
from runners.separated.base_runner import Runner
import pandas as pd

def _t2n(x):
    return x.detach().cpu().numpy()

class CRunner(Runner):
    """Runner class to perform training, evaluation. See parent class for details."""
    def __init__(self, config):
        super(CRunner, self).__init__(config)

    def run(self):

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        rewards_log = []
        inv_log = []
        actions_log = []
        demand_log = []
        overall_reward= []
        best_reward = float('-inf')
        best_bw = []
        record = 0

        for episode in range(episodes):
            
            if episode % self.eval_interval == 0 and self.use_eval:
                re, dict_write = self.eval()
                dict_write.update({'return':re})
                self.writter.add_scalars('cost_graph', dict_write, episode)
                print("Eval average cost: ", re, " Eval cost composition: ", dict_write)
                print("Best Eval average cost(before): ", best_reward)
                # test_reward,_=self.eval(test_tf=True)
                # print("Best Eval average reward: ", best_reward, " Best Test average reward: ", test_reward)
                if(re > best_reward):
                    if episode > 0:
                        self.save()
                        print("A better model is saved!")
                    
                    best_reward = re
                    best_dict_write = dict_write
                    record = 0
                elif(episode > self.n_warmup_evaluations):
                    record += 1
                    if(record == self.n_no_improvement_thres):
                        print("Training finished because of no imporvement for " + str(self.n_no_improvement_thres) + " evaluations")
                        self.model_dir=str(self.run_dir / 'models')
                        self.restore()
                        test_reward,dict_write_test=self.eval(test_tf=True)
                        print("Best Eval average cost: ", best_reward, " Eval cost composition: ", best_dict_write, " Best Test average reward: ", test_reward, " Test cost composition: ", dict_write_test)
                        return best_reward,best_dict_write, test_reward, dict_write_test
                # break

            self.warmup(train = True, normalize = self.all_args.norm_input, test_tf = False)
            if self.use_linear_lr_decay:
                for agent in range(self.all_args.num_agents):
                    self.trainer[agent].policy.lr_decay(episode, episodes)

            # values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env = self.collect(0)
            for step in range(self.episode_length):


                if self.all_args.algorithm_name == 'happo':
                    # Sample actions
                    values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env = self.collect(step)
                    # Obser reward and next obs
                    obs, obs_critic, rewards, dones, infos = self.envs.step(actions_env)
                elif self.all_args.algorithm_name == 'heuristic1':
                    inv, demand, orders = self.envs.get_property()
                    actions_env=[]
                    for i in range(len(inv)):
                        inv_,demand_,orders_= inv[i], demand[i], orders[i]
                        actions_=[]
                        for j in range(len(inv_)):
                            inv_pred=max(inv_[j]+orders_[j][0]-demand_[j],0)
                            inv_pred=max(inv_pred+orders_[j][1]-demand_[j],0)
                            inv_pred=max(inv_pred+orders_[j][2]-demand_[j],0)
                            inv_pred=max(inv_pred+orders_[j][3]-demand_[j],0)
                            order_heu=max(demand_[j]-inv_pred,0)
                            # order_heu=order_heu if order_heu>5 else 0
                            actions_.append(order_heu)
                        actions_env.append(actions_)
                    obs, rewards, dones, infos = self.envs.step(actions_env,False)
                elif self.all_args.algorithm_name == 'heuristic2':
                    def get_order_revenue_list(revenue_per, holding_cost, backorder_cost, shipping_cost, demand_pred):
                        revenue_list=[]
                        revenue_list.append(-demand_pred*backorder_cost)
                        for k in range(1,30):
                            revenue_k=(k*demand_pred*revenue_per-sum(range(k))*demand_pred*holding_cost-shipping_cost)/k
                            revenue_list.append(revenue_k)
                        return revenue_list

                    inv, demand, orders = self.envs.get_property()
                    actions_env=[]
                    
                    for env_idx in range(self.n_rollout_threads):
                        actions_=[]
                        for agent in range(self.num_agents):
                            inv = self.buffer[agent].obs[step][env_idx][0]
                            demand = self.buffer[agent].obs[step][env_idx][1]
                            orders = self.buffer[agent].obs[step][env_idx][4:]

                            inv_pred=max(inv+orders[0]-demand,0)
                            inv_pred=max(inv_pred+orders[1]-demand,0)
                            inv_pred=max(inv_pred+orders[2]-demand,0)
                            inv_pred=max(inv_pred+orders[3]-demand,0)
                            order_best=np.argmax(get_order_revenue_list(1., 0.2, 0.5, 5, demand))*demand
                            order_tf=1.5*max(demand-inv_pred,0)>0.2*max(order_best-demand,0)
                            order_heu=max(order_best-inv_pred,0) if order_tf else 0
                            # order_heu=order_heu if order_heu>5 else 0
                            actions_.append([order_heu,0])
                        actions_env.append(actions_)
                    actions_env=np.array(actions_env)
                
                        
                    obs, obs_critic, rewards, dones, infos = self.envs.step(actions_env,False)


                

                
                available_actions = np.array([[None for agent_id in range(self.num_agents)] for info in infos])

                rewards_log.append(rewards)

                inv, demand, orders = self.envs.get_property()

                import copy
                inv_log.append(copy.deepcopy(inv))   # 不知道为什么这个会变，就给他用copy锁了
                demand_log.append(demand)
                actions_log.append(copy.deepcopy(orders))

                data = obs, obs_critic, rewards, dones, infos, available_actions, \
                       values, actions, action_log_probs, \
                       rnn_states, rnn_states_critic 
                
                # insert data into buffer
                self.insert(data)

            # compute return and update network
            if self.all_args.training_bf:
                self.compute()
                train_infos = self.train()
                for i,train_info in enumerate(train_infos):
                    self.writter.add_scalars('training process_{}'.format(i), train_info, episode)
            
            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads           

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print("\n Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                        .format(self.algorithm_name,
                                self.experiment_name,
                                episode,
                                episodes,
                                total_num_steps,
                                self.num_env_steps,
                                int(total_num_steps / (end - start))))

                threads_rew = [[] for i in range(self.n_rollout_threads)]
                threads_inv = [[] for i in range(self.n_rollout_threads)]
                threads_act = [[] for i in range(self.n_rollout_threads)]
                threads_demand = [[] for i in range(self.n_rollout_threads)]
                for i in range(len(rewards_log)):
                    for j in range(self.n_rollout_threads):
                        threads_rew[j].append(rewards_log[i][j])
                        threads_inv[j].append(inv_log[i][j])
                        threads_act[j].append(actions_log[i][j])
                        threads_demand[j].append(demand_log[i][j])

                # thread_0=np.c_[np.array(threads_demand[0]),np.array(threads_inv[0]),np.array(threads_act[0]).reshape((200,15)),np.array(threads_rew[0]).reshape((200,3))]
                
                # thread_0=pd.DataFrame(thread_0)
                # thread_0.to_csv('thread0.csv')

                overall_reward.append(np.mean(threads_rew))
                if(len(overall_reward)<6):
                    smooth_reward = overall_reward
                else:
                    smooth_reward = []
                    for i in range(len(overall_reward)-5):
                        smooth_reward.append(np.mean(overall_reward[i:i+10]))
                
                for t in range(min(len(threads_rew),5)):
                    rew = [[] for i in range(self.num_involver)]
                    inv = [[] for i in range(self.num_involver)]
                    act = [[] for i in range(self.num_involver)]
                    for i in range(len(threads_rew[t])):
                        for j in range(self.num_involver):
                            rew[j].append(threads_rew[t][i][j])
                            inv[j].append(threads_inv[t][i][j])
                            act[j].append(threads_act[t][i][j])
                    rew = [round(np.mean(l), 2) for l in rew]
                    inv = [round(np.mean(l), 2) for l in inv]
                    act = [round(np.mean(l), 2) for l in act]
                    print("Reward for thread " + str(t+1) + ": " + str(rew) + " " + str(round(np.mean(rew),2))+"  Inventory: " + str(inv)+"  Order: " + str(act) + " Demand: " + str(np.mean(threads_demand[t], 0)))
                rewards_log = []
                inv_log = []
                actions_log = []
                demand_log = []

        print("Training finished because of finish all the episodes: " + episodes + " episodes")
        self.model_dir=str(self.run_dir / 'models')
        self.restore()
        test_reward,dict_write_test=self.eval(test_tf=True)
        print("Best Eval average cost: ", best_reward, " Eval cost composition: ", best_dict_write, " Best Test average reward: ", test_reward, " Test cost composition: ", dict_write_test)
        return best_reward,best_dict_write, test_reward, dict_write_test

    def warmup(self,train = True, normalize = True, test_tf = False):
        # reset env
        obs, obs_critic = self.envs.reset(train , normalize, test_tf)
        # replay buffer
        
        for agent_id in range(self.num_agents): 
            policy_observation_space = np.array(list(obs[:, agent_id])).copy()
            self.buffer[agent_id].share_obs[0] = np.array(list(obs_critic[:, agent_id]))
            self.buffer[agent_id].obs[0] = policy_observation_space
            self.buffer[agent_id].available_actions[0] = None
            # self.buffer[agent_id].rnn_states = np.zeros_like(self.buffer[agent_id].rnn_states)
            # self.buffer[agent_id].rnn_states_critic = np.zeros_like(self.buffer[agent_id].rnn_states_critic)

    @torch.no_grad()
    def collect(self, step):
        value_collector=[]
        action_collector=[]
        temp_actions_env = []
        action_log_prob_collector=[]
        rnn_state_collector=[]
        rnn_state_critic_collector=[]

        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_rollout()
            value, action, action_log_prob, rnn_state, rnn_state_critic \
                = self.trainer[agent_id].policy.get_actions(self.buffer[agent_id].share_obs[step],
                                                self.buffer[agent_id].obs[step],
                                                self.buffer[agent_id].rnn_states[step],
                                                self.buffer[agent_id].rnn_states_critic[step],
                                                self.buffer[agent_id].masks[step],
                                                self.buffer[agent_id].available_actions[step])
            # value, action, action_log_prob, rnn_state, rnn_state_critic \
            #     = self.trainer[agent_id].policy.get_actions(self.buffer[agent_id].share_obs[step],
            #                                     self.buffer[agent_id].obs[step],
            #                                     self.buffer[agent_id].rnn_states[step],
            #                                     self.buffer[agent_id].rnn_states_critic[step],
            #                                     self.buffer[agent_id].masks[step],
            #                                     self.buffer[agent_id].available_actions[step],deterministic=True)

            value_collector.append(_t2n(value))
            action_collector.append(_t2n(action))

            if self.envs.action_space[agent_id].__class__.__name__ == 'MultiDiscrete':
                for i in range(self.envs.action_space[agent_id].shape):
                    uc_action_env = np.eye(self.envs.action_space[agent_id].high[i] + 1)[action[:, i].cpu().detach()]
                    if i == 0:
                        action_env = uc_action_env
                    else:
                        action_env = np.concatenate((action_env, uc_action_env), axis=1)
            elif self.envs.action_space[agent_id].__class__.__name__ == 'Discrete':
                action_env = np.squeeze(np.eye(self.envs.action_space[agent_id].n)[action.cpu().detach()], 1)
            elif self.envs.action_space[agent_id].__class__.__name__ == 'Box':
                action_env = np.array(action.cpu().detach())
            else:
                raise NotImplementedError
            
            temp_actions_env.append(action_env)

            action_log_prob_collector.append(_t2n(action_log_prob))
            rnn_state_collector.append(_t2n(rnn_state))
            rnn_state_critic_collector.append(_t2n(rnn_state_critic))

            
        
        # [self.envs, agents, dim]
        actions_env = []
        for i in range(self.n_rollout_threads):
            one_hot_action_env = []
            for temp_action_env in temp_actions_env:
                one_hot_action_env.append(temp_action_env[i])
                if self.all_args.central_controller:
                    one_hot_action_env = temp_action_env[i].reshape(self.all_args.num_involver,-1)
                    # print(np.where(one_hot_action_env >= 1))
            actions_env.append(one_hot_action_env)

        values = np.array(value_collector).transpose(1, 0, 2)
        actions = np.array(action_collector).transpose(1, 0, 2)
        action_log_probs = np.array(action_log_prob_collector).transpose(1, 0, 2)
        rnn_states = np.array(rnn_state_collector).transpose(1, 0, 2, 3)
        rnn_states_critic = np.array(rnn_state_critic_collector).transpose(1, 0, 2, 3)

        
        return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env

    def insert(self, data):
        obs, share_obs_critic, rewards, dones, infos, available_actions, \
        values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        dones_env = np.all(dones, axis=1)
        if self.all_args.central_controller:
            dones = dones[:,[0]]
        rnn_states[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, *self.buffer[0].rnn_states_critic.shape[2:]), dtype=np.float32)

        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        active_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        active_masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
        active_masks[dones_env == True] = np.ones(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        bad_masks = np.array([[[1.0] for agent_id in range(self.num_agents)] for info in infos])
        
        for agent_id in range(self.num_agents):
            policy_observation_space = np.array(list(obs[:, agent_id]))
            share_obs_critic_ = np.array(list(share_obs_critic[:, agent_id]))
            
            self.buffer[agent_id].insert(share_obs_critic_, policy_observation_space, rnn_states[:,agent_id],
                    rnn_states_critic[:,agent_id],actions[:,agent_id], action_log_probs[:,agent_id],
                    values[:,agent_id], rewards[:,agent_id], masks[:,agent_id])

    def log_train(self, train_infos, total_num_steps):
        for agent_id in range(self.num_agents):
            train_infos[agent_id]["average_step_rewards"] = np.mean(self.buffer[agent_id].rewards)
            for k, v in train_infos[agent_id].items():
                agent_k = "agent%i/" % agent_id + k
                self.writter.add_scalars(agent_k, {agent_k: v}, total_num_steps)

    

    @torch.no_grad()
    def eval(self, test_tf=False):
        
        # 把之前的tensorboard数据清除掉，不然会同时显示，啥也看不清
        self.clear_tensorboard(test_tf)
        overall_reward = []
        eval_num = self.eval_envs.get_eval_num()

        penalty_cost_all=0
        ordering_cost_all=0
        holding_cost_all=0
        shipping_cost_all=0
        shipping_cost_pure=0
        ordering_times=0

        for eval_index in range(eval_num):
            eval_obs, eval_obs_critic = self.eval_envs.reset(test_tf,self.all_args.norm_input)
            
            eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_rnn_states_critic = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size_critic), dtype=np.float32)
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

            rewards_log=[]
            eval_values_steps=[[] for i in range(self.episode_length)]

            for eval_step in range(self.episode_length):
                
                temp_actions_env = []
                eval_values=[]
                for agent_id in range(self.num_agents):
                    self.trainer[agent_id].policy.hist_demand=self.eval_envs.get_hist_demand()[0][agent_id]
                    self.trainer[agent_id].prep_rollout()
                    eval_value,eval_actions,_,temp_rnn_state,temp_rnn_state_critic\
                        =self.trainer[agent_id].policy.get_actions(eval_obs_critic[:,agent_id],
                                                        eval_obs[:,agent_id],
                                                        eval_rnn_states[:,agent_id],
                                                        eval_rnn_states_critic[:,agent_id],
                                                        eval_masks[:,agent_id],
                                                        None,
                                                        deterministic=True)
                    eval_values.append(float(eval_value[0][0].detach().cpu()))


                    # eval_actions, temp_rnn_state = \
                    #     self.trainer[agent_id].policy.act(eval_obs[:,agent_id],
                    #                             eval_rnn_states[:,agent_id],
                    #                             eval_masks[:,agent_id],
                    #                             None,
                    #                             deterministic=True)
                    eval_rnn_states[:,agent_id]=_t2n(temp_rnn_state)
                    eval_rnn_states_critic[:,agent_id]=_t2n(temp_rnn_state_critic)
                    action = eval_actions.detach().cpu().numpy()

                    if self.envs.action_space[agent_id].__class__.__name__ == 'MultiDiscrete':
                        for i in range(self.envs.action_space[agent_id].shape):
                            uc_action_env = np.eye(self.envs.action_space[agent_id].high[i] + 1)[action[:, i]]
                            # print(np.where(uc_action_env >= 1))
                            if i == 0:
                                action_env = uc_action_env
                            else:
                                action_env = np.concatenate((action_env, uc_action_env), axis=1)
                                # print(np.where(action_env >= 1))
                    elif self.envs.action_space[agent_id].__class__.__name__ == 'Discrete':
                        action_env = np.squeeze(np.eye(self.envs.action_space[agent_id].n)[action], 1)
                    else:
                        action_env=action
                        # raise NotImplementedError

                    temp_actions_env.append(action_env)

                eval_values_steps[eval_step]=eval_values
                #eval_actions = np.array(eval_actions_collector).transpose(1,0,2)
                eval_actions_env = []
                for i in range(self.n_eval_rollout_threads):
                    eval_one_hot_action_env = []
                    for eval_temp_action_env in temp_actions_env:
                        eval_one_hot_action_env.append(eval_temp_action_env[i])
                        if self.all_args.central_controller:
                            eval_one_hot_action_env = eval_temp_action_env[i].reshape(self.all_args.num_involver,-1)
                            # print(np.where(eval_one_hot_action_env >= 1))
                    eval_actions_env.append(eval_one_hot_action_env)

                # Obser reward and next obs
                eval_obs,eval_obs_critic, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions_env)

                eval_available_actions = None

                overall_reward.append(np.mean(eval_rewards))
                rewards_log.append(eval_rewards[0])

                eval_dones_env = np.all(eval_dones, axis=1)

                eval_rnn_states[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
                eval_rnn_states_critic[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size_critic), dtype=np.float32)
                eval_masks = np.ones((self.all_args.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
                eval_masks[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

                # 统计不同cost
                for a in range(self.num_agents):
                    penalty_cost_all+=eval_infos[0][a]['penalty_cost']
                    ordering_cost_all+=eval_infos[0][a]['ordering_cost']
                    holding_cost_all+=eval_infos[0][a]['holding_cost']
                    shipping_cost_all+=eval_infos[0][a]['shipping_cost_all']
                    shipping_cost_pure+=eval_infos[0][a]['shipping_cost_pure']
                    if eval_step == self.episode_length-1:
                        ordering_times += eval_infos[0][a]['ordering_times']

                # 写入tensorboard
                if eval_index<3:
                    for agent_id in range(self.num_involver):
                        self.writter.add_scalars(main_tag='{eval_or_test}_{eval_index}/Execution_{agent_id}'.format(eval_or_test='test' if test_tf else 'eval',eval_index=eval_index,agent_id=agent_id),
                                                tag_scalar_dict=eval_infos[0][agent_id],global_step= eval_step)
            
            rewards_log=np.array(rewards_log).reshape((self.episode_length,self.num_involver))
            if eval_index<3:
                eval_returns=np.zeros((self.episode_length + 1,self.num_involver), dtype=np.float32)
                eval_returns[-1] = 0
                for step in reversed(range(rewards_log.shape[0])):
                    eval_returns[step] = eval_returns[step + 1] * self.all_args.gamma  + rewards_log[step]
                    for agent_id in range(self.num_agents):
                                est_V=eval_values_steps[step][agent_id]
                                if self.trainer[agent_id]._use_popart or self.trainer[agent_id]._use_valuenorm:
                                    est_V=self.trainer[agent_id].value_normalizer.denormalize(np.array([est_V]))[0]
                                self.writter.add_scalars(main_tag='{eval_or_test}_{eval_index}/Value_{agent_id}'.format(eval_or_test='test' if test_tf else 'eval',eval_index=eval_index,agent_id=agent_id),
                                                        tag_scalar_dict={'est_V': est_V,
                                                                        'real_V': eval_returns[step][agent_id],},
                                                                        global_step= step)
        num_all= eval_num*self.episode_length*self.num_agents
        dict_cost={"shipping_cost_all":shipping_cost_all/num_all,"shipping_cost_pure":shipping_cost_pure/num_all,"holding_cost":holding_cost_all/num_all,"ordering_cost":ordering_cost_all/num_all,"penalty_cost":penalty_cost_all/num_all,"ordering_times":ordering_times/eval_num/self.num_agents}
        all_demand_mean=self.demand_mean_test if test_tf else self.demand_mean_val
        norm_reward_drift = self.all_args.reward_norm_multiplier*all_demand_mean if 'norm' in self.all_args.reward_type else 0 
        return np.mean(overall_reward)-norm_reward_drift, dict_cost