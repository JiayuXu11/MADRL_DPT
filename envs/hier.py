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
P = [4.0, 4.0, 4.0, 4.0, 4.0]  # penalty for unsatisfied demand
C = [2 ,2 ,2 ,2 ,2]  # purchasing cost
D = [[0.         ,5.94820641, 6.32083547, 4.01651129, 3.40052813], 
    [5.94820641, 0.        , 5.50879987 ,1.54098246, 4.89310332], 
    [6.32083547, 5.50879987, 0.         ,4.38603051 ,1.57469625], 
    [4.01651129 ,1.54098246 ,4.38603051 ,0.         ,6.84649239], 
    [3.40052813 ,4.89310332 ,1.57469625 ,6.84649239, 0.        ]]  # 一个二维数组，表示不同仓库之间的distance

# S = [0.5, 0.5, 0.5, 0.5, 0.5]  # unit shipping cost
# RT = 0.1 # revenue per transshipping unit

S_I = 10
S_O = 10

AGENT_NUM = 3
DEMAND_MAX = 20

ACTION_DIM_DICT = {'central_discrete':[(DEMAND_MAX*2+1)]*3}



CRITIC_OBS_DIM = 13
EPISODE_LEN = 200
# ALPHA=0.7


FIXED_COST = 5

EVAL_PTH = ["./eval_data/merton/0/", "./eval_data/merton/1/", "./eval_data/merton/2/"]
TEST_PTH = ["./test_data/merton/0/", "./test_data/merton/1/", "./test_data/merton/2/"]






#====================================================================================

def get_eval_data():
    """
    - Need to be implemented
    - Load local demand data for evaluation
    - Inputs:
        - Modify the inputs as you need
    - Outputs:
        - n_eval: int, number of demand sequences (also number of episodes in one evaluation)
        - eval_data: list, demand data for evaluation
    """
    files_0 = os.listdir(EVAL_PTH[0])
    n_eval = len(files_0)
    eval_data=[]
  
    for i in range(n_eval):
        eval_data_i=[]
        for j in range(len(EVAL_PTH)):
            files=os.listdir(EVAL_PTH[j])
            data = []
            with open(EVAL_PTH[j] + files[i], "rb") as f:
                lines = f.readlines()
                for line in lines:
                    data.append(int(line))
            eval_data_i.append(data)
        eval_data.append(eval_data_i)
    # print(np.array(eval_data).shape)
    return n_eval, eval_data

def get_test_data():
    """
    - Need to be implemented
    - Load local demand data for testuation
    - Inputs:
        - Modify the inputs as you need
    - Outputs:
        - n_test: int, number of demand sequences (also number of episodes in one testuation)
        - test_data: list, demand data for testuation
    """
    files_0 = os.listdir(TEST_PTH[0])
    n_test = len(files_0)
    test_data=[]
  
    for i in range(n_test):
        test_data_i=[]
        for j in range(len(TEST_PTH)):
            files=os.listdir(TEST_PTH[j])
            data = []
            with open(TEST_PTH[j] + files[i], "rb") as f:
                lines = f.readlines()
                for line in lines:
                    data.append(int(line))
            test_data_i.append(data)
        test_data.append(test_data_i)
    # print(np.array(test_data).shape)
    return n_test, test_data

def get_training_data():
    """
    - Need to be implemented
    - Load one-episode simulated or local demand data for training
    - Inputs:
        - Modify the inputs as you need
    - Outputs:
        - demand_list: list, one-episode demand data for training
    """
    demand_list = [generator.merton(EPISODE_LEN, DEMAND_MAX), generator.merton(EPISODE_LEN, DEMAND_MAX),generator.merton(EPISODE_LEN, DEMAND_MAX)]
    return demand_list


class Env(object):

    def __init__(self,args):

        #============================================================================
        # Define the member variables you need here.
        # The following three memeber variables must be defined
        self.agent_num = AGENT_NUM
        self.lead_time = args.lead_time
        # actor_obs
        self.instant_info_sharing=args.instant_info_sharing
        self.obs_transship=args.obs_transship
        self.actor_obs_step =args.actor_obs_step
        self.obs_dim = self.get_obs_dim(self.instant_info_sharing, self.actor_obs_step)
        # critic_obs
        self.use_centralized_V = args.use_centralized_V
        self.obs_critic_dim = self.get_critic_obs_dim(self.use_centralized_V, True)

        self.action_type = args.action_type
        self.action_dim = ACTION_DIM_DICT[self.action_type]
        

        self.alpha = 0
        self.ratio_transsship = args.ratio_transship
        self.gamma = args.gamma

        # transship收益分配
        self.transship_revenue_method = args.transship_revenue_method
        self.constant_transship_revenue = args.constant_transship_revenue
        self.ratio_transship_revenue = args.ratio_transship_revenue

        # 嵌套网络
        self.nested_network, self.shipping_cost_matrix =self.get_nested_network()
        
        self.inventory = []
        self.order = []
        self.record_act_sta = [[] for i in range(AGENT_NUM)]
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
        #============================================================================ 

        self.n_eval, self.eval_data = get_eval_data() # Get demand data for evaluation
        self.eval_index = 0 # Counter for evaluation

        self.n_test, self.test_data = get_test_data() # Get demand data for evaluation
        self.test_index = 0 # Counter for evaluation

    def get_obs_dim(self, info_sharing, obs_step):
        base_dim = 2 + self.lead_time
        transship_dim = 2 
        step_dim = 1 if obs_step else 0

        transship_dim_dict = {'no_transship':0,'self_transship':1,'all_transship':self.agent_num}

        if info_sharing:
            return base_dim*AGENT_NUM + step_dim
        
        return base_dim+transship_dim_dict[self.obs_transship]*transship_dim + step_dim
    
    def get_critic_obs_dim(self, info_sharing, obs_step):
        demand_dim = 4*self.agent_num if info_sharing else 4
        return self.get_obs_dim(info_sharing, obs_step) + demand_dim


    def reset(self, train = True, normalize = True, test_tf = False):

        #============================================================================
        # Reset all the member variables that need to be reset at the beginning of an episode here
        # Note that self.eval_index should not be reset here
        # The following one must be reset
        self.step_num = 0
        self.action_history = [[] for i in range(AGENT_NUM)]
        self.train = train
        self.normalize = normalize
        self.inventory = [S_I for i in range(AGENT_NUM)]
        self.order = [[S_O for i in range(self.lead_time)] for j in range(AGENT_NUM)]
        self.transship_request = [0 for i in range(AGENT_NUM)]
        self.transship_intend = [0 for i in range(AGENT_NUM)]

        # info
        self.inventory_start = [S_I for i in range(AGENT_NUM)]
        self.shortage = [0 for i in range(AGENT_NUM)]
        self.reward_selfish = [0 for i in range(AGENT_NUM)]
        self.reward_selfish_cum = [0 for i in range(AGENT_NUM)]
        self.reward = [0 for i in range(AGENT_NUM)]
        self.reward_cum = [0 for i in range(AGENT_NUM)]


        #============================================================================

        if(train == True):
            self.demand_list = get_training_data() # Get demand data for training
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
        self.demand_mean=[np.mean([demand[idx] for idx in range(self.step_num,self.episode_max_steps)]) for demand in self.demand_list]
        self.demand_std=[np.std([demand[idx] for idx in range(self.step_num,self.episode_max_steps)]) for demand in self.demand_list]

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
            info_dict['self_implement']=self.transship_request[agent_id]  # 表示自己交付的需求量
            # info_dict['transship_intend']=self.transship_intend[agent_id]
            info_dict['shortage'] = self.shortage[agent_id]
            info_dict['reward_selfish'] = self.reward_selfish[agent_id]
            info_dict['reward_selfish_cum'] = self.reward_selfish_cum[agent_id]
            info_dict['reward'] = self.reward[agent_id]
            info_dict['reward_cum'] = self.reward_cum[agent_id]
            infos.append(info_dict)
        return infos

    def step(self, actions):
        
        action = self.action_map(actions) # Map outputs of MADRL to actual ordering actions

        reward = self.state_update(action) # System state update
        # reward = self.state_update_transship_revenue_sharing(action)

        sub_agent_obs = self.get_step_obs(self.instant_info_sharing, self.actor_obs_step) # Get step obs
        sub_agent_reward = self.get_processed_rewards(reward) # Get processed rewards

        if(self.step_num > self.episode_max_steps):
            sub_agent_done = [True for i in range(self.agent_num)]
        else:
            sub_agent_done = [False for i in range(self.agent_num)]
        sub_agent_info = self.get_info()
        critic_obs = self.get_step_obs_critic(self.use_centralized_V, True)
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
        return [self.demand_list[0][self.step_num-1],self.demand_list[1][self.step_num-1],self.demand_list[2][self.step_num-1]]
    
    def get_hist_demand(self):
        return [self.demand_list[0][:self.step_num],self.demand_list[1][:self.step_num],self.demand_list[2][:self.step_num]]
    
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
        return [self.order[0][-1],self.order[1][-1],self.order[2][-1]]
    
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
    
    # 根据 D 距离矩阵，生成嵌套网络，以及不同地点的交付成本
    def get_nested_network(self):
        pass

    # 返回一个二维数组，根据嵌套网络，逐级实现需求交付
    def get_allocation(self):
        pass

    def action_map(self, actions):
        order_amounts=[]
        transship_amounts=[]

        if self.action_type == 'discrete' or self.action_type == 'central_discrete':
            action = [np.argmax(i) for i in actions]
            for i in range(len(action)):
                order_amount=action[i]
                order_amounts.append(order_amount)
        # allocated_amounts[i][j] 表示仓库i接收了来自仓库j的库存数
        allocated_amounts=self.get_allocation()
        
        mapped_actions=[k for k in zip(order_amounts,allocated_amounts)]

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
            self.demand_mean_dy=[np.mean([demand[idx] for idx in range(self.step_num,min(self.looking_len+self.step_num,self.episode_max_steps))]) for demand in self.demand_list]
            self.demand_std_dy=[np.std([demand[idx] for idx in range(self.step_num,min(self.looking_len+self.step_num,self.episode_max_steps))]) for demand in self.demand_list]

    # critic network 专属obs
    def get_step_obs_critic(self, info_sharing, obs_step):
        actor_agent_obs = self.get_step_obs(info_sharing, obs_step)
        self.set_demand_statistics()
        sub_agent_obs = []
        for i in range(self.agent_num):
            actor_arr = actor_agent_obs[i]
            if info_sharing:
                demand_mean_arr = np.array(self.demand_mean  + self.demand_mean_dy)
                demand_std_arr = np.array(self.demand_std + self.demand_std_dy)
            else:
                demand_mean_arr = np.array([self.demand_mean[i],self.demand_mean_dy[i]])
                demand_std_arr = np.array([self.demand_std[i] + self.demand_std_dy[i]])
            if(self.normalize):
                arr = np.concatenate([actor_arr,demand_mean_arr*2/DEMAND_MAX-1., demand_std_arr/DEMAND_MAX-1.])
            else:
                arr = np.concatenate([actor_arr,demand_mean_arr, demand_std_arr])
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
        self.reward_cum=[self.reward_cum[i]+self.reward[i] for i in range(AGENT_NUM)]
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
        cur_demand = [self.demand_list[0][self.step_num], self.demand_list[1][self.step_num],self.demand_list[2][self.step_num]]
        rewards=[]
        self.step_num+=1
        for i in range(self.agent_num):
            self.order[i].append(action[i][0])
            allocate_amount = sum(action[i][1])
            inv_start=self.inventory[i]+self.order[i][0]+allocate_amount
            self.inventory_start[i]=inv_start
            shipping_cost = sum([i*j for i,j in zip(self.shipping_cost_matrix[i],action[i][1])])
            reward= -C[i]*(sum(action[i][0]))-shipping_cost-H[i]*max(inv_start-cur_demand[i],0)+P[i]*min(inv_start-cur_demand[i],0)
        
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
    