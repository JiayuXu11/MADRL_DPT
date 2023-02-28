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
P = [0.5, 0.5, 0.5, 0.5, 0.5]  # penalty for unsatisfied demand
R = [3 ,3 ,3 ,3 ,3]  # revenue per unit
C = [2 ,2 ,2 ,2 ,2]  # purchasing cost
S = [0.5, 0.5, 0.5, 0.5, 0.5]  # unit shipping cost

S_I = 10
S_O = 10
AGENT_NUM = 3
DEMAND_MAX = 20
ACTION_DIM = 2
LEAD_TIME = 4
OBS_DIM = 4+LEAD_TIME
EPISODE_LEN = 200
ALPHA=0.8


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

    def __init__(self):

        #============================================================================
        # Define the member variables you need here.
        # The following three memeber variables must be defined
        self.agent_num = AGENT_NUM
        self.obs_dim = OBS_DIM
        self.action_dim = ACTION_DIM 

        self.inventory = []
        self.order = []
        self.record_act_sta = [[] for i in range(AGENT_NUM)]
        self.eval_episode_len = EPISODE_LEN
        self.episode_max_steps = EPISODE_LEN

        #============================================================================ 

        self.n_eval, self.eval_data = get_eval_data() # Get demand data for evaluation
        self.eval_index = 0 # Counter for evaluation

        self.n_test, self.test_data = get_test_data() # Get demand data for evaluation
        self.test_index = 0 # Counter for evaluation

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
        self.order = [[S_O for i in range(LEAD_TIME)] for j in range(AGENT_NUM)]
        self.transship_request = [0 for i in range(AGENT_NUM)]


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

        sub_agent_obs = self.get_reset_obs(normalize) # Get reset obs
            
        return sub_agent_obs

    def step(self, actions, one_hot=False):
        
        if(one_hot):
            action_ = [np.argmax(i) for i in actions]
        else:
            action_ = actions
        
        action = self.action_map(action_,one_hot) # Map outputs of MADRL to actual ordering actions
        reward = self.state_update(action) # System state update
        sub_agent_obs = self.get_step_obs(action) # Get step obs
        sub_agent_reward = self.get_processed_rewards(reward) # Get processed rewards

        if(self.step_num > self.episode_max_steps):
            sub_agent_done = [True for i in range(self.agent_num)]
        else:
            sub_agent_done = [False for i in range(self.agent_num)]
        sub_agent_info = [[] for i in range(self.agent_num)]

        return [sub_agent_obs, sub_agent_reward, sub_agent_done, sub_agent_info]
    
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
    
    def action_map(self, action,one_hot=True):
        order_amounts=[]
        transship_amounts=[]
        for i in range(len(action)):
            if one_hot:
                order_amount=action[i]//(2*DEMAND_MAX+1)
                order_amounts.append(order_amount)
                transship_amount=max(action[i]%(2*DEMAND_MAX+1)-DEMAND_MAX,-self.inventory[i]-self.order[i][0])
                transship_amounts.append(transship_amount)
            else:
                order_amounts.append(action[i][0])
                transship_amounts.append(max(action[i][1],-self.inventory[i]-self.order[i][0]))
        
        # 撮合transship
        i=0
        while(sum(transship_amounts) != 0):
            if sum(transship_amounts) > 0:
                if (transship_amounts[i] > 0):
                    transship_amounts[i] += -1
            elif sum(transship_amounts) < 0:
                if (transship_amounts[i] < 0):
                    transship_amounts[i] += 1
            i+=1
            i=0 if i > self.agent_num-1 else i

        mapped_actions=[k for k in zip(order_amounts,transship_amounts)]

        return mapped_actions
    
    def get_reset_obs(self,normalize):
        """
        - Need to be implemented
        - Get reset obs (initial obs)
        - Inputs:
            - Modify the inputs as you need
        - Outputs:
            - sub_agent_obs: list, a list for obs of all actors, shape for obs of each actor: (self.obs_dim, )
        """
        sub_agent_obs = []
        for i in range(self.agent_num):
            if(normalize):
                arr = np.array([self.inventory[i],self.demand_list[i][self.step_num-1], self.transship_request[i], self.step_num]+self.order[i])/(DEMAND_MAX)
            else:
                arr = np.array([self.inventory[i],self.demand_list[i][self.step_num-1], self.transship_request[i], self.step_num]+self.order[i])
            arr = np.reshape(arr, (self.obs_dim,))
            sub_agent_obs.append(arr)
        return sub_agent_obs
    
    def get_step_obs(self,action):
        """
        - Need to be implemented
        - Get step obs (obs for each step)
        - Inputs:
            - Modify the inputs as you need
        - Outputs:
            - sub_agent_obs: list, a list for obs of all actors, shape for obs of each actor: (self.obs_dim, )
        """
        sub_agent_obs = []
        for i in range(self.agent_num):
            if(self.normalize):
                arr = np.array([self.inventory[i],self.demand_list[i][self.step_num-1], self.transship_request[i], self.step_num]+ self.order[i])/(DEMAND_MAX)
            else:
                arr = np.array([self.inventory[i],self.demand_list[i][self.step_num-1], self.transship_request[i], self.step_num]+ self.order[i])
            arr = np.reshape(arr, (self.obs_dim,))
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
        if(self.train):
            processed_rewards = [[ALPHA*i + (1-ALPHA)*np.mean(reward)] for i in reward]
        else:
            processed_rewards = [[i] for i in reward]

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
            self.transship_request[i]=action[i][1]
            inv_start=self.inventory[i]+self.order[i][0]+self.transship_request[i]
            reward= -C[i]*(sum(action[i]))-S[i]*max(self.transship_request[i],0)+R[i]*min(inv_start,cur_demand[i])-H[i]*max(inv_start-cur_demand[i],0)+P[i]*min(inv_start-cur_demand[i],0)-FIXED_COST*(1 if action[i][0]>0 else 0)
            rewards.append(reward)
            self.inventory[i]=max(inv_start-cur_demand[i],0.)
            self.action_history[i].append(action[i])
            self.order[i]=self.order[i][1:]
        return rewards