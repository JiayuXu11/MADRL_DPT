import numpy as np
from . import generator
import os 
import random
#====================================================================================
# Define all the exogenous parameters you need in your supply chain environment here.
# They may include:
# 1. Number of actors
# 2. Dimention of observation space and action space
# 3. Cost coefficients
# 4. File path of your evaluation demand data
# 5. Other parameters
H = [0.2, 0.2, 0.2, 0.2, 0.2]  # holding cost
R = [3.0, 3.0, 3.0, 3.0, 3.0] 
P = [3.5, 3.5, 3.5, 3.5, 3.5]  # penalty for unsatisfied demand
C = [2 ,2 ,2 ,2 ,2]  # purchasing cost
S = [0.5, 0.5, 0.5, 0.5, 0.5]  # unit shipping cost
# RT = 0.1 # revenue per transshipping unit

DISTANCE = np.array([[0,820,1411,770,872],[820,0,2404,624,420],[1411,2404,0,1785,2187],[770,624,1785,0,557],[872,420,2187,557,0]])  # 一个二维数组，表示不同仓库之间的distance


S_I = 10
S_O = 10

DEMAND_MAX = 20





CRITIC_OBS_DIM = 13
EPISODE_LEN = 200
# ALPHA=0.7


FIXED_COST = 5

# EVAL_PTH = ["./eval_data/merton/0/", "./eval_data/merton/1/", "./eval_data/merton/2/"]
# TEST_PTH = ["./test_data/merton/0/", "./test_data/merton/1/", "./test_data/merton/2/"]




#====================================================================================


class Env(object):

    def __init__(self,args):
        
        #============================================================================
        # Define the member variables you need here.
        # The following three memeber variables must be defined
        self.agent_num = args.num_involver
        self.lead_time = args.lead_time
        self.demand_info_for_critic=args.demand_info_for_critic

        # actor_obs
        self.instant_info_sharing=args.instant_info_sharing
        self.obs_transship=args.obs_transship
        self.actor_obs_step =args.actor_obs_step
        self.obs_dim = self.get_obs_dim(self.instant_info_sharing, self.actor_obs_step)
        # critic_obs
        self.use_centralized_V = args.use_centralized_V
        self.obs_critic_dim = self.get_critic_obs_dim(self.use_centralized_V, True)

        ACTION_DIM_DICT = {'discrete':(DEMAND_MAX*2+1)*(DEMAND_MAX+1),'multi_discrete':[DEMAND_MAX*3+1,DEMAND_MAX*2+1],'continue':2, 'central_multi_discrete':[DEMAND_MAX*3+1,DEMAND_MAX*2+1]*self.agent_num, 'central_discrete':[(DEMAND_MAX*2+1)*(DEMAND_MAX+1)]*self.agent_num}
        self.action_type = args.action_type
        self.action_dim = ACTION_DIM_DICT[self.action_type]
        
        # transship顺序
        self.shipping_order = self.get_shipping_order(DISTANCE)
        self.shipping_cost_matrix = self.phi(DISTANCE)

        self.alpha = args.alpha
        self.ratio_transsship = args.ratio_transship
        self.gamma = args.gamma


        # transship收益分配
        self.transship_revenue_method = args.transship_revenue_method
        self.constant_transship_revenue = args.constant_transship_revenue
        self.ratio_transship_revenue = args.ratio_transship_revenue

        
        self.inventory = []
        self.order = []
        self.record_act_sta = [[] for i in range(self.agent_num)]
        self.eval_episode_len = EPISODE_LEN
        self.episode_max_steps = EPISODE_LEN

        self.looking_len = round(1./(1.-self.gamma +1e-10))

        # info 
        self.inventory_start = []
        self.shortage=[]
        self.reward_selfish=[]
        self.reward_selfish_cum=[]
        self.reward=[]
        self.reward_cum=[]

        self.reward_type=args.reward_type
        self.reward_norm_multiplier = args.reward_norm_multiplier
        #============================================================================ 
        self.eval_path = [args.eval_dir+'/{}/'.format(i) for i in range(self.agent_num)]
        self.test_path = [args.test_dir+'/{}/'.format(i) for i in range(self.agent_num)]
        self.n_eval, self.eval_data = self.get_eval_data() # Get demand data for evaluation
        self.eval_index = 0 # Counter for evaluation

        self.n_test, self.test_data = self.get_test_data() # Get demand data for evaluation
        self.test_index = 0 # Counter for evaluation

    def get_shipping_order(self, matrix):
        indices_sorted_by_value = np.argsort(matrix, axis=None)
        # reshape the sorted indices to match the original matrix shape
        indices_sorted_by_value = np.unravel_index(indices_sorted_by_value, matrix.shape)

        # exclude diagonal elements and symmetric elements
        indices_sorted_by_value = list(zip(indices_sorted_by_value[0], indices_sorted_by_value[1]))
        indices_sorted_by_value = [index for index in indices_sorted_by_value if index[0] < index[1]]
        return indices_sorted_by_value

    # distance->transshipping price    
    def phi(self, distance):
        return 0.0005*distance
    
    def get_eval_data(self):
        """
        - Need to be implemented
        - Load local demand data for evaluation
        - Inputs:
            - Modify the inputs as you need
        - Outputs:
            - n_eval: int, number of demand sequences (also number of episodes in one evaluation)
            - eval_data: list, demand data for evaluation
        """
        files_0 = os.listdir(self.eval_path[0])
        n_eval = len(files_0)
        eval_data=[]
    
        for i in range(n_eval):
            eval_data_i=[]
            for j in range(len(self.eval_path)):
                files=os.listdir(self.eval_path[j])
                data = []
                with open(self.eval_path[j] + files[i], "rb") as f:
                    lines = f.readlines()
                    for line in lines:
                        data.append(int(line))
                eval_data_i.append(data)
            eval_data.append(eval_data_i)
        # print(np.array(eval_data).shape)
        return n_eval, eval_data

    def get_test_data(self):
        """
        - Need to be implemented
        - Load local demand data for testuation
        - Inputs:
            - Modify the inputs as you need
        - Outputs:
            - n_test: int, number of demand sequences (also number of episodes in one testuation)
            - test_data: list, demand data for testuation
        """
        files_0 = os.listdir(self.test_path[0])
        n_test = len(files_0)
        test_data=[]
    
        for i in range(n_test):
            test_data_i=[]
            for j in range(len(self.test_path)):
                files=os.listdir(self.test_path[j])
                data = []
                with open(self.test_path[j] + files[i], "rb") as f:
                    lines = f.readlines()
                    for line in lines:
                        data.append(int(line))
                test_data_i.append(data)
            test_data.append(test_data_i)
        # print(np.array(test_data).shape)
        return n_test, test_data

    def get_training_data(self):
        """
        - Need to be implemented
        - Load one-episode simulated or local demand data for training
        - Inputs:
            - Modify the inputs as you need
        - Outputs:
            - demand_list: list, one-episode demand data for training
        """
        demand_list = [generator.merton(EPISODE_LEN, DEMAND_MAX) for _ in range(self.agent_num)]
        return demand_list

    def get_obs_dim(self, info_sharing, obs_step):
        base_dim = 2 + self.lead_time
        transship_dim = 2 
        step_dim = 1 if obs_step else 0

        transship_dim_dict = {'no_transship':0,'self_transship':1,'all_transship':self.agent_num}

        if info_sharing:
            return base_dim*self.agent_num + step_dim
        
        return base_dim+transship_dim_dict[self.obs_transship]*transship_dim + step_dim
    
    def get_critic_obs_dim(self, info_sharing, obs_step):
        demand_info_num = len(self.demand_info_for_critic)
        demand_info_num = demand_info_num + (4 if 'quantile' in self.demand_info_for_critic else 0 )
        demand_dim = demand_info_num*self.agent_num if info_sharing else demand_info_num*1
        return self.get_obs_dim(info_sharing, obs_step) + demand_dim


    def reset(self, train = True, normalize = True, test_tf = False):

        #============================================================================
        # Reset all the member variables that need to be reset at the beginning of an episode here
        # Note that self.eval_index should not be reset here
        # The following one must be reset
        self.step_num = 0
        self.action_history = [[] for i in range(self.agent_num)]
        self.train = train
        self.normalize = normalize
        self.inventory = [S_I for i in range(self.agent_num)]
        self.order = [[S_O for i in range(self.lead_time)] for j in range(self.agent_num)]
        self.transship_request = [0 for i in range(self.agent_num)]
        self.transship_intend = [0 for i in range(self.agent_num)]
        self.transship_matrix = np.zeros((self.agent_num, self.agent_num))

        # info
        self.inventory_start = [S_I for i in range(self.agent_num)]
        self.demand_fulfilled = [0 for i in range(self.agent_num)]
        self.shortage = [0 for i in range(self.agent_num)]
        self.reward_selfish = [0 for i in range(self.agent_num)]
        self.reward_selfish_cum = [0 for i in range(self.agent_num)]
        self.reward = [0 for i in range(self.agent_num)]
        self.reward_cum = [0 for i in range(self.agent_num)]
        self.shipping_cost_pure = [0 for i in range(self.agent_num)]
        self.shipping_cost_all = [0 for i in range(self.agent_num)]
        self.ordering_cost = [0 for i in range(self.agent_num)]
        self.penalty_cost = [0 for i in range(self.agent_num)]
        self.holding_cost = [0 for i in range(self.agent_num)]
        self.ordering_times = [0 for i in range(self.agent_num)]


        #============================================================================

        if(train == True):
            self.demand_list = self.get_training_data() # Get demand data for training
        elif test_tf:
            self.demand_list = self.test_data[self.test_index] # Get demand data for testuation
            self.test_index += 1
            if(self.test_index == self.n_test):
                self.test_index = 0
        else:
            self.demand_list = self.eval_data[self.eval_index] # Get demand data for evaluation
            self.eval_index += 1
            if(self.eval_index == self.n_eval):
                self.eval_index = 0

        # 统计需求的mean和std        
        self.set_demand_statistics()
        

        sub_agent_obs = self.get_step_obs(self.instant_info_sharing, self.actor_obs_step) 
        critic_obs = self.get_step_obs_critic(self.use_centralized_V, True)
        
        return sub_agent_obs, critic_obs
    
    def get_info(self):
        infos=[]
        for agent_id in range(self.agent_num):
            info_dict={}
            info_dict['start_inventory'] = self.inventory_start[agent_id]
            info_dict['end_inventory']=self.inventory[agent_id]
            info_dict['demand'] = self.demand_list[agent_id][self.step_num-1]
            info_dict['order'] = self.order[agent_id][-1]
            info_dict['transship']=self.transship_request[agent_id]
            info_dict['transship_intend']=self.transship_intend[agent_id]
            info_dict['demand_fulfilled'] = self.demand_fulfilled[agent_id]
            info_dict['shortage'] = self.shortage[agent_id]
            info_dict['reward_selfish'] = self.reward_selfish[agent_id]
            info_dict['reward_selfish_cum'] = self.reward_selfish_cum[agent_id]
            info_dict['reward'] = self.reward[agent_id]
            info_dict['reward_cum'] = self.reward_cum[agent_id]
            info_dict['shipping_cost_all'] = self.shipping_cost_all[agent_id]
            info_dict['shipping_cost_pure'] = self.shipping_cost_pure[agent_id]
            info_dict['penalty_cost'] = self.penalty_cost[agent_id]
            info_dict['holding_cost'] = self.holding_cost[agent_id]
            info_dict['ordering_times'] = self.ordering_times[agent_id]
            info_dict['ordering_cost'] = self.ordering_cost[agent_id]
            
            infos.append(info_dict)
        return infos

    def step(self, actions):
        
        action = self.action_map(actions) # Map outputs of MADRL to actual ordering actions

        # reward = self.state_update(action) # System state update
        reward = self.state_update_transship_revenue_sharing(action)

        sub_agent_obs = self.get_step_obs(self.instant_info_sharing, self.actor_obs_step) # Get step obs
        sub_agent_reward = self.get_processed_rewards(reward) # Get processed rewards

        if(self.step_num > self.episode_max_steps):
            sub_agent_done = [True for i in range(self.agent_num)]
        else:
            sub_agent_done = [False for i in range(self.agent_num)]
        sub_agent_info = self.get_info()
        critic_obs = self.get_step_obs_critic(self.use_centralized_V, True)
        self.transship_matrix = np.zeros((self.agent_num, self.agent_num))
        return [sub_agent_obs, critic_obs, sub_agent_reward, sub_agent_done, sub_agent_info]
    

    def get_eval_num(self):
        return self.n_eval

    # 不需要其实
    def get_eval_bw_res(self):
        """"
        - Need to be implemented
        - Get the ordering fluctuation measurement for each actor/echelon during evaluation. The results will be printed out after each evaluation during training. 
        - Inputs:
            - Modify the inputs as you need
        - Outputs:
            - eval_bw_res: list, ordering fluctuation measurement for each actor/echelon
        """
        eval_bw_res = []

        return eval_bw_res
    
    def get_demand(self):
        return [self.demand_list[i][self.step_num-1] for i in range(self.agent_num)]
    
    def get_hist_demand(self):
        return [self.demand_list[i][:self.step_num] for i in range(self.agent_num)]
    
    def get_orders(self):
        """"
        - Need to be implemented
        - Get actual ordering actions for all actors. Will be printed out during the training.
        - Inputs:
            - Modify the inputs as you need
        - Outputs:
            - current_orders: list, actual ordering actions for all actors
        """
        
        # return self.order
        return [self.order[i][-1] for i in range(self.agent_num)]
    
    def get_inventory(self):
        """"
        - Need to be implemented
        - Get inventory levels for all actors. Will be printed out during the training.
        - Inputs:
            - Modify the inputs as you need
        - Outputs:
            - current_inventory: list, inventory levels for all actors
        """
        
        return self.inventory
    
    def action_map(self, actions):
        order_amounts=[]
        transship_amounts=[]

        if self.action_type == 'discrete' or self.action_type == 'central_discrete':
            action = [np.argmax(i) for i in actions]
            for i in range(len(action)):
                order_amount=action[i]//(DEMAND_MAX+1)
                order_amounts.append(order_amount)
                transship_amount=max(action[i]%(DEMAND_MAX+1)-DEMAND_MAX/2,-self.inventory[i]-self.order[i][0])
                transship_amounts.append(transship_amount)
                
        elif self.action_type == 'multi_discrete'or self.action_type == 'central_multi_discrete':
            action = [(np.argmax(i[:self.action_dim[0]]),np.argmax(i[self.action_dim[0]:])) for i in actions]
            for i in range(len(action)):
                order_amount=action[i][0]
                order_amounts.append(order_amount)
                transship_amount=max(action[i][1]-DEMAND_MAX,-self.inventory[i]-self.order[i][0])
                transship_amounts.append(transship_amount)

        elif self.action_type == 'continue':
            action = actions
            for i in range(len(action)):
                order_amounts.append(action[i][0])
                transship_amounts.append(max(action[i][1],-self.inventory[i]-self.order[i][0]))

        else:
            raise Exception("wrong action_type")
        
        self.transship_intend=transship_amounts.copy()

        for a1,a2 in self.shipping_order:
            if a1<self.agent_num and a2<self.agent_num:
                # 表示一个想要货，一个想出货
                if transship_amounts[a1]*transship_amounts[a2]<0:
                    tran_amount = min(abs(transship_amounts[a1]),abs(transship_amounts[a2]))
                    self.transship_matrix[a1][a2]=tran_amount if transship_amounts[a1]>0 else -tran_amount
                    self.transship_matrix[a2][a1]=-self.transship_matrix[a1][a2]
                    transship_amounts[a1]-=self.transship_matrix[a1][a2]
                    transship_amounts[a2]-=self.transship_matrix[a2][a1]

        # 最后几天订的货，因为leadtime原因也到不了
        if not self.actor_obs_step and (self.step_num > EPISODE_LEN-self.lead_time-1):
            order_amounts = [0 for _ in range(self.agent_num)]
        transship_amounts= [sum(self.transship_matrix[i]) for i in range(self.agent_num)]
        mapped_actions=[k for k in zip(order_amounts,transship_amounts)]

        return mapped_actions
    

    def get_step_obs(self, info_sharing, obs_step):
        """
        - Need to be implemented
        - Get step obs (obs for each step)
        - Inputs:
            - Modify the inputs as you need
        - Outputs:
            - sub_agent_obs: list, a list for obs of all actors, shape for obs of each actor: (self.obs_dim, )
        """
        sub_agent_obs = []

        order_all = []
        for k in range(self.agent_num):
            order_all+=self.order[k]

        for i in range(self.agent_num):

            if info_sharing:
                base_arr = np.array(self.inventory + [self.demand_list[k][self.step_num-1] for k in range(self.agent_num)])
                order_arr = np.array(order_all)
            else:
                base_arr = np.array([self.inventory[i],self.demand_list[i][self.step_num-1]])
                order_arr = np.array(self.order[i])
            if self.obs_transship == 'no_transship' or info_sharing:
                transship_arr = np.array([])
            elif self.obs_transship == 'all_transship':
                transship_arr = np.array(self.transship_intend + self.transship_request)
            elif self.obs_transship == 'self_transship':
                transship_arr = np.array([self.transship_intend[i],self.transship_request[i]])
            else:
                raise Exception('wrong obs_transship')
            
            if obs_step:
                step_arr = np.array([self.step_num])
            else:
                step_arr = np.array([])
            if(self.normalize):
                arr = np.concatenate([base_arr*2/DEMAND_MAX-1.,order_arr/DEMAND_MAX-1.,transship_arr/DEMAND_MAX,step_arr*2/EPISODE_LEN-1])
            else:
                arr = np.concatenate([base_arr,order_arr,transship_arr,step_arr])
            sub_agent_obs.append(arr)

        return sub_agent_obs
    
    
    # 统计需求均值与标准差，以供critic network使用
    def set_demand_statistics(self):
        if self.step_num<self.episode_max_steps:  
            self.demand_mean=[np.mean([demand[idx] for idx in range(self.step_num,self.episode_max_steps)]) for demand in self.demand_list]
            self.demand_std=[np.std([demand[idx] for idx in range(self.step_num,self.episode_max_steps)]) for demand in self.demand_list]
            self.demand_mean_dy=[np.mean([demand[idx] for idx in range(self.step_num,min(self.looking_len+self.step_num,self.episode_max_steps))]) for demand in self.demand_list]
            self.demand_std_dy=[np.std([demand[idx] for idx in range(self.step_num,min(self.looking_len+self.step_num,self.episode_max_steps))]) for demand in self.demand_list]
            self.demand_q5=[np.quantile([demand[idx] for idx in range(self.step_num,min(self.looking_len+self.step_num,self.episode_max_steps))],0.05) for demand in self.demand_list]
            self.demand_q25=[np.quantile([demand[idx] for idx in range(self.step_num,min(self.looking_len+self.step_num,self.episode_max_steps))],0.25) for demand in self.demand_list]
            self.demand_q50=[np.quantile([demand[idx] for idx in range(self.step_num,min(self.looking_len+self.step_num,self.episode_max_steps))],0.5) for demand in self.demand_list]
            self.demand_q75=[np.quantile([demand[idx] for idx in range(self.step_num,min(self.looking_len+self.step_num,self.episode_max_steps))],0.75) for demand in self.demand_list]
            self.demand_q95=[np.quantile([demand[idx] for idx in range(self.step_num,min(self.looking_len+self.step_num,self.episode_max_steps))],0.95) for demand in self.demand_list]
            
    # critic network 专属obs
    def get_step_obs_critic(self, info_sharing, obs_step):
        actor_agent_obs = self.get_step_obs(info_sharing, obs_step)
        self.set_demand_statistics()
        sub_agent_obs = []
        for i in range(self.agent_num):
            actor_arr = actor_agent_obs[i]
            if info_sharing:
                demand_mean_arr = (self.demand_mean if 'all_mean' in self.demand_info_for_critic else []) + (self.demand_mean_dy if 'mean' in self.demand_info_for_critic else [])
                demand_mean_arr = np.array(demand_mean_arr)
                demand_quantile_arr = (self.demand_q5+self.demand_q25+self.demand_q50+self.demand_q75+self.demand_q95) if 'quantile' in self.demand_info_for_critic else [] 
                demand_quantile_arr = np.array(demand_quantile_arr)
                demand_std_arr = (self.demand_std if 'all_std' in self.demand_info_for_critic else []) + (self.demand_std_dy if 'std' in self.demand_info_for_critic else [])
                demand_std_arr = np.array(demand_std_arr)
            else:
                demand_mean_arr = np.array([self.demand_mean[i],self.demand_mean_dy[i]])
                demand_std_arr = np.array([self.demand_std[i] + self.demand_std_dy[i]])
            if(self.normalize):
                arr = np.concatenate([actor_arr,demand_mean_arr*2/DEMAND_MAX-1., demand_std_arr/DEMAND_MAX-1., demand_quantile_arr*2/DEMAND_MAX-1])
            else:
                arr = np.concatenate([actor_arr,demand_mean_arr, demand_std_arr, demand_quantile_arr])
            sub_agent_obs.append(arr)

        return sub_agent_obs
    
    def get_processed_rewards(self, reward):
        """
        - Need to be implemented
        - Get processed rewards for all actors
        - Inputs:
            - reward: list, reward directly from the state update (typically each actor's on-period cost)
            - Modify the inputs as you need
        - Outputs:
            - processed_rewards: list, a list for rewards of all actors
        """
        processed_rewards = []
        # if(self.train):
        #     processed_rewards = [[self.alpha*i + (1-self.alpha)*np.mean(reward)] for i in reward]
        # else:
        #     processed_rewards = [[i] for i in reward]
        processed_rewards = [[self.alpha*i + (1-self.alpha)*np.mean(reward)] for i in reward]
        self.reward=[self.alpha*i + (1-self.alpha)*np.mean(reward) for i in reward]
        self.reward_cum=[self.reward_cum[i]+self.reward[i] for i in range(self.agent_num)]
        return processed_rewards
    
    def state_update(self, action):
        """
        - Need to be implemented
        - Update system state and record some states that you may need in other fuctions like get_eval_bw_res, get_orders, etc.
        - Inputs:
            - action: list, processed actions for each actor
            - Modify the inputs as you need
        - Outputs:
            - rewards: list, rewards for each actors (typically one-period costs for all actors)
        """
        all_transship_revenue,transship_volume = self.get_transship_revenue(action) 
        cur_demand = [self.demand_list[0][self.step_num], self.demand_list[1][self.step_num],self.demand_list[2][self.step_num]]
        rewards=[]
        self.step_num+=1
        for i in range(self.agent_num):
            self.order[i].append(action[i][0])
            self.transship_request[i]=action[i][1]
            inv_start=self.inventory[i]+self.order[i][0]+self.transship_request[i]
            self.inventory_start[i]=inv_start
            reward= -C[i]*(sum(action[i]))-S[i]*max(self.transship_request[i],0)-H[i]*max(inv_start-cur_demand[i],0)+P[i]*min(inv_start-cur_demand[i],0)-FIXED_COST*(1 if action[i][0]>0 else 0)
            
            # transship 收益分配
            transship_revenue = 0
            if self.transship_revenue_method == 'constant':
                transship_revenue = -self.constant_transship_revenue*self.transship_request[i]
            elif self.transship_revenue_method == 'ratio':
                if self.transship_request[i]>0:
                    transship_revenue = -(1-self.ratio_transship_revenue)*all_transship_revenue[i]
                else:
                    transship_revenue = -(self.transship_request[i]/transship_volume) * sum(all_transship_revenue)*(1-self.ratio_transship_revenue)
            else:
                raise Exception('wrong transship revenue aloocated method')
            reward+=transship_revenue

            # 最后一天将仓库内剩余货品按成本价折算
            if self.step_num > EPISODE_LEN-1:
                reward=reward+(H[i]+C[i])*max(inv_start-cur_demand[i],0)
            rewards.append(reward)

            self.reward_selfish_cum[i]+=reward
            self.reward_selfish[i]=reward
            
            self.shortage[i]=cur_demand[i]-inv_start
            self.inventory[i]=max(inv_start-cur_demand[i],0.)
            self.action_history[i].append(action[i])
            self.order[i]=self.order[i][1:]
            
        return rewards
    
    # transship的收益
    def get_transship_revenue(self,action):
        cur_demand = [self.demand_list[0][self.step_num], self.demand_list[1][self.step_num],self.demand_list[2][self.step_num]]
        transship_revenue=[0 for i in range(self.agent_num)]
        transship_volume = 1e-10
        for i in range(self.agent_num):
            ts = action[i][1]
            if ts<=0:
                transship_revenue[i] = 0
                continue
            transship_volume+=ts
            inv_start_without_transship=self.inventory[i]+self.order[i][0]
            shortage_wihout_transship = max(cur_demand[i]-inv_start_without_transship , 0)
            transship_revenue[i] = max((-C[i]+P[i])*min(shortage_wihout_transship,ts) - S[i]*ts - H[i]*max(ts-shortage_wihout_transship,0),0)   
        return transship_revenue , transship_volume      
    
    # 考虑了transship供需关系，及带来的未来收益的分配模式
    def state_update_transship_revenue_sharing(self, action):
        """
        - Need to be implemented
        - Update system state and record some states that you may need in other fuctions like get_eval_bw_res, get_orders, etc.
        - Inputs:
            - action: list, processed actions for each actor
            - Modify the inputs as you need
        - Outputs:
            - rewards: list, rewards for each actors (typically one-period costs for all actors)
        """
        all_transship_revenue,transship_volume = self.get_transship_revenue(action) 
        cur_demand = [self.demand_list[0][self.step_num], self.demand_list[1][self.step_num],self.demand_list[2][self.step_num]]
        rewards_after=[]
        rewards_before = []
        Vs = []
        V_befores =[]
        self.step_num+=1

        # 计算transship前后收益
        for i in range(self.agent_num):
            self.order[i].append(action[i][0])
            self.transship_request[i]=action[i][1]
            inv_start_before = self.inventory[i]+self.order[i][0]
            inv_start=self.inventory[i]+self.order[i][0]+self.transship_request[i]
            self.inventory_start[i]=inv_start

            revenue_demand =cur_demand[i]*R[i] if 'reward' in self.reward_type else 0
            norm_drift =cur_demand[i]*self.reward_norm_multiplier if 'norm' in self.reward_type else 0 

            
            # 纯运费
            self.shipping_cost_pure[i] = sum([self.shipping_cost_matrix[i][j]*self.transship_matrix[i][j] if self.transship_matrix[i][j]>0 else 0 for j in range(self.agent_num)])
            # 运费+买卖货的费用
            self.shipping_cost_all[i] = self.shipping_cost_pure[i] + C[i]*(action[i][1])

            self.ordering_cost[i] = C[i]*(action[i][0])+FIXED_COST*(1 if action[i][0]>0 else 0)
            self.ordering_times[i]+=(1 if action[i][0]>0 else 0)
            self.penalty_cost[i] = -P[i]*min(inv_start-cur_demand[i],0)
            self.holding_cost[i] = H[i]*max(inv_start-cur_demand[i],0)
            # transship 后的reward 
            reward= -self.ordering_cost[i]-self.shipping_cost_all[i]-self.holding_cost[i]-self.penalty_cost[i]+revenue_demand+norm_drift
            
            # transship前的reward
            reward_before= -C[i]*(action[i][0])-H[i]*max(inv_start_before-cur_demand[i],0)+P[i]*min(inv_start_before-cur_demand[i],0)-FIXED_COST*(1 if action[i][0]>0 else 0)+revenue_demand+norm_drift
            
            self.demand_fulfilled[i] =  min(inv_start,cur_demand[i])
            self.shortage[i]=cur_demand[i]-inv_start
            self.inventory[i]=max(inv_start-cur_demand[i],0.)
            self.action_history[i].append(action[i])
            self.order[i]=self.order[i][1:]

            # transship 前的未来收益，可能要考虑gamma
            V  = 0 
            # transship 后的未来收益
            V_before = 0 
            Vs.append(V)
            V_befores.append(V_before)

            # 最后一天将仓库内剩余货品按成本价折算
            if self.step_num > EPISODE_LEN-1:
                reward=reward+(C[i])*max(inv_start-cur_demand[i],0)
                reward_before=reward_before+(C[i])*max(inv_start-cur_demand[i],0)
            rewards_after.append(reward)
            rewards_before.append(reward_before)

        transship_revenue = np.array(rewards_after) + self.gamma * np.array(Vs) - np.array(rewards_before) + self.gamma * np.array(V_befores)
        transship_revenue_sum = np.sum(transship_revenue) 
        transship_intend_p = sum([t if t>0 else 0 for t in self.transship_intend])
        transship_intend_n = sum([-t if t<0 else 0 for t in self.transship_intend])
        ratio_pn = transship_intend_n/(transship_intend_p+transship_intend_n+1e-10)

        rewards=[]
        # 把transship收益分了
        for i in range(self.agent_num):
            reward = rewards_after[i]

             # transship 收益分配
            transship_reallocate = 0
            if self.transship_revenue_method == 'constant':
                transship_reallocate = -self.constant_transship_revenue*self.transship_request[i]
            elif self.transship_revenue_method == 'ratio':
                if self.transship_request[i]>0:
                    transship_reallocate = -(1-self.ratio_transship_revenue)*all_transship_revenue[i]
                else:
                    transship_reallocate = -(self.transship_request[i]/transship_volume) * sum(all_transship_revenue)*(1-self.ratio_transship_revenue)
            elif self.transship_revenue_method == 'market_ratio':
                if self.transship_intend[i] >= 0:
                    # volume_ratio = self.transship_intend[i]/transship_intend_p
                    volume_ratio = self.transship_request[i]/transship_volume
                    revenue_allocated = transship_revenue_sum*ratio_pn*volume_ratio
                elif self.transship_intend[i]<0:
                    # volume_ratio = -self.transship_intend[i]/transship_intend_n
                    volume_ratio = -self.transship_request[i]/transship_volume
                    revenue_allocated = transship_revenue_sum*(1-ratio_pn)*volume_ratio

                transship_reallocate = revenue_allocated - transship_revenue[i]
            else:
                raise Exception('wrong transship revenue allocated method')
            self.shipping_cost_all[i]-=transship_reallocate
            reward+=transship_reallocate
            rewards.append(reward)
            self.reward_selfish_cum[i]+=reward
            self.reward_selfish[i]=reward
            
        return rewards
    