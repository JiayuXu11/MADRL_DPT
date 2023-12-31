import numpy as np
from . import generator
import os 
import random
import chardet


DISTANCE = [np.loadtxt('envs/distance/distance_matrix_0.txt', delimiter=',').astype(int),
            np.loadtxt('envs/distance/distance_matrix_1.txt', delimiter=',').astype(int),
            np.loadtxt('envs/distance/distance_matrix_2.txt', delimiter=',').astype(int),
            np.loadtxt('envs/distance/distance_matrix_3.txt', delimiter=',').astype(int),]

S_I = 10
S_O = 10

DEMAND_MAX = 20

EPISODE_LEN = 200



#====================================================================================


class Env(object):

    def __init__(self,args):
        
        #============================================================================
        # Define the member variables you need here.
        # The following three memeber variables must be defined
        self.agent_num = args.num_involver
        self.lead_time = args.lead_time
        self.demand_info_for_critic=args.demand_info_for_critic
        self.if_transship = args.if_transship
        self.train_path = args.train_dir
        self.SKU_id = args.SKU_id
        self.demand_max_for_clip = args.demand_max_for_clip
         
        # cost parameter
        self.H = args.H  # holding cost
        self.R = args.R  # selling price per unit (only used for reward)
        self.P = args.P  # penalty cost
        self.C = args.C  # ordering cost
        self.FIXED_COST = args.FIXED_COST 
        self.shipping_cost_per_distance = args.shipping_cost_per_distance

        self.generator_method = args.generator_method

        # Whether to consider the issue of different distances or not
        self.homo_distance=args.homo_distance
        self.mini_pooling = args.mini_pooling
                
        # 设置distance
        self.distance = DISTANCE[0 if self.homo_distance else args.distance_index][:self.agent_num,:self.agent_num]

        # Cash on delivery or payment in advance.
        self.pay_first= args.pay_first

        # actor_obs
        self.instant_info_sharing=args.instant_info_sharing
        self.adjusted_info_sharing = args.adjusted_info_sharing
        self.obs_transship=args.obs_transship
        self.actor_obs_step =args.actor_obs_step
        self.critic_obs_step = args.critic_obs_step
        self.obs_dim = self.get_obs_dim(self.instant_info_sharing, self.adjusted_info_sharing, self.actor_obs_step)
        # critic_obs
        self.use_centralized_V = args.use_centralized_V
        self.obs_critic_dim = self.get_critic_obs_dim(self.use_centralized_V, self.critic_obs_step)

        self.demand_for_action_dim = args.demand_for_action_dim if args.demand_for_action_dim else [DEMAND_MAX]*self.agent_num
        self.action_type = args.action_type

        if self.action_type == 'discrete':
            self.action_dim = [(self.demand_for_action_dim[i]*2+1)*(self.demand_for_action_dim[i]+1) for i in range(self.agent_num)]
        elif self.action_type == 'multi_discrete':
            self.action_dim = [[self.demand_for_action_dim[i]*3+1,self.demand_for_action_dim[i]*2+1] for i in range(self.agent_num)]
        elif self.action_type == 'central_multi_discrete':
            self.action_dim = []
            for i in range(self.agent_num):
                self.action_dim += [self.demand_for_action_dim[i]*3+1,self.demand_for_action_dim[i]*2+1]
            self.action_dim = [self.action_dim]
        elif self.action_type == 'central_discrete':
            self.action_dim = [[(self.demand_for_action_dim[i]*2+1)*(self.demand_for_action_dim[i]+1) for i in range(self.agent_num) ]]
        # self.action_dim = ACTION_DIM_DICT[self.action_type]
        
        # transship order
        self.shipping_order = self.get_shipping_order(self.distance)
        self.shipping_cost_matrix = self.phi(self.distance)

        self.alpha = args.alpha
        self.ratio_transsship = args.ratio_transship
        self.gamma = args.gamma


        # transship revenue allocation
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
        return self.shipping_cost_per_distance*distance
    
    def get_training_data(self):
        """
        - Need to be implemented
        - Load one-episode simulated or local demand data for training
        - Inputs:
            - Modify the inputs as you need
        - Outputs:
            - demand_list: list, one-episode demand data for training
        """
        if(self.generator_method=='merton'):
            demand_list = [np.array(generator.merton(2*EPISODE_LEN,DEMAND_MAX).demand_list) for _ in range(self.agent_num)]
        elif(self.generator_method=='poisson'):
            demand_list = [generator.poisson(2*EPISODE_LEN,DEMAND_MAX/2,DEMAND_MAX).demand_list for _ in range(self.agent_num)]
        elif(self.generator_method=='normal'):
            demand_list = [generator.normal(2*EPISODE_LEN,DEMAND_MAX/2,DEMAND_MAX/4,DEMAND_MAX).demand_list for _ in range(self.agent_num)]
        elif(self.generator_method=='uniform'):
            demand_list = [generator.uniform(2*EPISODE_LEN,DEMAND_MAX).demand_list for _ in range(self.agent_num)]
        elif(self.generator_method=='shanshu_arima'):
             demand_list = [generator.shanshu_arima(self.SKU_id,i,2*EPISODE_LEN,self.demand_max_for_clip).demand_list for i in range(self.agent_num)]
        elif(self.generator_method=='random_fragment'):
            demand_list = [generator.random_fragment(i,2*EPISODE_LEN,self.train_path,1000*DEMAND_MAX,random.randint(0,500)).demand_list for i in range(self.agent_num)]
        elif(self.generator_method=='random_resample'):
            demand_list = [generator.random_resample(i,2*EPISODE_LEN,self.train_path,1000*DEMAND_MAX).demand_list for i in range(self.agent_num)]
        return demand_list


    def get_obs_dim(self, info_sharing, adjusted_info_sharing, obs_step):
        base_dim = 2 + self.lead_time
        transship_dim = 2 
        step_dim = 1 if obs_step else 0

        transship_dim_dict = {'no_transship':0,'self_transship':1,'all_transship':self.agent_num}

        if info_sharing:
            return base_dim*self.agent_num + step_dim
        if adjusted_info_sharing:
            return 3*self.agent_num + step_dim + self.lead_time + transship_dim_dict[self.obs_transship]*transship_dim  # 3包含demand，inventory，intransit_sum
        
        return base_dim+transship_dim_dict[self.obs_transship]*transship_dim + step_dim
    
    def get_critic_obs_dim(self, info_sharing, obs_step):
        # demand_info_num = len(self.demand_info_for_critic)
        demand_info_num = (5 if 'quantile' in self.demand_info_for_critic else 0 )+(self.lead_time if 'LT_all' in self.demand_info_for_critic else 0 )+(1 if 'LT_mean' in self.demand_info_for_critic else 0 )
        demand_dim = demand_info_num*self.agent_num if info_sharing else demand_info_num*1
        patial_info_sharing_dim = self.agent_num*2 if not info_sharing else 0
        # The input of the critic does not include the current demand
        obs_diff = self.agent_num if info_sharing else 1
        return self.get_obs_dim(info_sharing, False, obs_step) + demand_dim + patial_info_sharing_dim - obs_diff


    def reset(self, train = True, normalize = True, demand_list_set = None):

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

        else:
            self.demand_list = demand_list_set

        sub_agent_obs = self.get_step_obs(self.instant_info_sharing, self.adjusted_info_sharing,self.actor_obs_step) 
        critic_obs = self.get_step_obs_critic(self.use_centralized_V, self.critic_obs_step)
        
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



        sub_agent_obs = self.get_step_obs(self.instant_info_sharing, self.adjusted_info_sharing, self.actor_obs_step) # Get step obs
        sub_agent_reward = self.get_processed_rewards(reward) # Get processed rewards

        if(self.step_num > self.episode_max_steps):
            sub_agent_done = [True for i in range(self.agent_num)]
        else:
            sub_agent_done = [False for i in range(self.agent_num)]

        sub_agent_info = self.get_info()


        critic_obs = self.get_step_obs_critic(self.use_centralized_V, self.critic_obs_step)


        self.transship_matrix = np.zeros((self.agent_num, self.agent_num))

        return [sub_agent_obs, critic_obs, sub_agent_reward, sub_agent_done, sub_agent_info]
    

    
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
            action = [i[0] for i in actions]
            for i in range(self.agent_num):
                order_amount=action[i]//(self.demand_for_action_dim[i]+1)
                order_amounts.append(order_amount)
                transship_amount=max(action[i]%(+1)-self.demand_for_action_dim[i]/2,-self.inventory[i]-self.order[i][0])
                transship_amounts.append(transship_amount)
                
        elif self.action_type == 'multi_discrete'or self.action_type == 'central_multi_discrete':
            # action = [(np.argmax(i[:self.action_dim[0]]),np.argmax(i[self.action_dim[0]:])) for i in actions]
            action = [(i[0],i[1]) for i in actions]
            for i in range(self.agent_num):
                order_amount=action[i][0]
                order_amounts.append(order_amount)
                transship_amount=max(action[i][1]-self.demand_for_action_dim[i],-self.inventory[i]-self.order[i][0])
                transship_amounts.append(transship_amount)

        elif self.action_type == 'continue':
            action = actions
            for i in range(self.agent_num):
                order_amounts.append(action[i][0])
                transship_amounts.append(max(action[i][1],-self.inventory[i]-self.order[i][0]))

        else:
            raise Exception("wrong action_type")
        
        if self.if_transship:
            self.transship_intend=transship_amounts.copy()

            if self.homo_distance:
                self.transship_merge_homo(transship_amounts)
            elif self.mini_pooling["flag"]:
                self.transship_merge_mp(transship_amounts, threshold=self.mini_pooling['threshold'], how=self.mini_pooling['how'])
            else:
                for a1,a2 in self.shipping_order:
                    if a1<self.agent_num and a2<self.agent_num:
                        # Represents one party requesting goods and another party wanting to sell goods.
                        if transship_amounts[a1]*transship_amounts[a2]<0:
                            tran_amount = min(abs(transship_amounts[a1]),abs(transship_amounts[a2]))
                            self.transship_matrix[a1][a2]=tran_amount if transship_amounts[a1]>0 else -tran_amount
                            self.transship_matrix[a2][a1]=-self.transship_matrix[a1][a2]
                            transship_amounts[a1]-=self.transship_matrix[a1][a2]
                            transship_amounts[a2]-=self.transship_matrix[a2][a1]

            transship_amounts= [sum(self.transship_matrix[i]) for i in range(self.agent_num)]
        else:
            transship_amounts = [0] * self.agent_num
        mapped_actions=[k for k in zip(order_amounts,transship_amounts)]

        return mapped_actions
    
    # Matching mechanism under homo distance.
    def transship_merge_homo(self,transship_amounts):

        if self.ratio_transsship:
            transship_pos=sum([t if t>0 else 0 for t in transship_amounts])
            transship_neg=sum([t if t<0 else 0 for t in transship_amounts])
            if sum(transship_amounts) < 0:
                ratio = -transship_pos/transship_neg
                for i in range(len(transship_amounts)):
                    transship_amounts[i]= round(ratio*transship_amounts[i],0) if transship_amounts[i]<0 else transship_amounts[i]
            elif sum(transship_amounts) > 0:
                ratio = -transship_neg/transship_pos
                for i in range(len(transship_amounts)):
                    transship_amounts[i]= round(ratio*transship_amounts[i],0) if transship_amounts[i]>0 else transship_amounts[i]
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
        
        # transship_matrix
        for a1 in range(self.agent_num):
            for a2 in range(self.agent_num):
                if transship_amounts[a1]*transship_amounts[a2]<0:
                    tran_amount = min(abs(transship_amounts[a1]),abs(transship_amounts[a2]))
                    self.transship_matrix[a1][a2]=tran_amount if transship_amounts[a1]>0 else -tran_amount
                    self.transship_matrix[a2][a1]=-self.transship_matrix[a1][a2]
                    transship_amounts[a1]-=self.transship_matrix[a1][a2]
                    transship_amounts[a2]-=self.transship_matrix[a2][a1]

    # hierarchical mpooling mechanism
    def transship_merge_mp(self, transship_amounts, threshold, how):
        def mini_pooling(distance_matrix, transship_amounts, thres):
            n = len(transship_amounts)
            possible_pairs = []

            # Find possible transship pairs
            for i in range(n):
                for j in range(n):
                    if i != j and transship_amounts[i] < 0 and transship_amounts[j] > 0:
                        possible_pairs.append((i, j, distance_matrix[i][j]))

            # Sort pairs by distance
            possible_pairs.sort(key=lambda x: x[2])

            # Filter pairs within the distance threshold
            mini_pool_pairs = [pair[:2] for pair in possible_pairs if pair[2] - possible_pairs[0][2] <= thres]

            if not mini_pool_pairs:
                return False
            
            return mini_pool_pairs
        
        while True:
            mini_pool_pairs = mini_pooling(self.distance, transship_amounts, threshold)
            if not mini_pool_pairs:
                break
            while mini_pool_pairs:

                if how == 'even':
                    curr_pair = random.choice(mini_pool_pairs)
                elif how == 'ratio':
                    weights = [transship_amounts[pair[1]] for pair in mini_pool_pairs]
                    curr_pair = random.choice(mini_pool_pairs, weights=weights, k=1)
                self.transship_matrix[curr_pair[0]][curr_pair[1]] -= 1
                self.transship_matrix[curr_pair[1]][curr_pair[0]] += 1
                transship_amounts[curr_pair[0]] += 1
                transship_amounts[curr_pair[1]] -= 1
                if transship_amounts[curr_pair[0]] == 0:
                    mini_pool_pairs = list(filter(lambda x: curr_pair[0] not in x, mini_pool_pairs))
                if transship_amounts[curr_pair[1]] == 0:
                    mini_pool_pairs = list(filter(lambda x: curr_pair[1] not in x, mini_pool_pairs))
                
            
            
        
    def get_step_obs(self, info_sharing, adjusted_info_sharing, obs_step, demand_today=True):
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
        order_avg = []
        for k in range(self.agent_num):
            order_all+= self.order[k]
            order_avg+= [np.mean(self.order[k])]

        for i in range(self.agent_num):

            if info_sharing:
                if i==0:
                    base_arr = np.array(self.inventory + ([(self.demand_list[k][self.step_num-1] if self.step_num>0 else self.demand_for_action_dim[k]/2) for k in range(self.agent_num)] if demand_today else []))
                    order_arr = np.array(order_all)
            elif adjusted_info_sharing:
                if i==0:
                    base_arr = np.array(self.inventory + ([(self.demand_list[k][self.step_num-1] if self.step_num>0 else self.demand_for_action_dim[k]/2) for k in range(self.agent_num)] if demand_today else []))
                order_arr = np.array(self.order[i] + order_avg)
            else:
                base_arr = np.array([self.inventory[i],(self.demand_list[i][self.step_num-1] if self.step_num>0 else self.demand_for_action_dim[i]/2) ]) if demand_today else np.array([self.inventory[i]])
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
                arr = np.concatenate([base_arr*2/self.demand_for_action_dim[i]-1.,order_arr/self.demand_for_action_dim[i]-1.,transship_arr/self.demand_for_action_dim[i],step_arr*2/EPISODE_LEN-1])
            else:
                arr = np.concatenate([base_arr,order_arr,transship_arr,step_arr])
            sub_agent_obs.append(arr)

        return sub_agent_obs
    
    def del_and_insert(self, arr, del_num, insert_num):
        # Binary search for the position of the number to be deleted
        left, right = 0, len(arr) - 1
        while left <= right:
            mid = (left + right) // 2
            if arr[mid] == del_num:
                arr[mid] = insert_num
                i = mid
                while i > 0 and arr[i - 1] > insert_num:
                    arr[i], arr[i - 1] = arr[i - 1], arr[i]
                    i -= 1
                i = mid
                while i < len(arr) - 1 and arr[i + 1] < insert_num:
                    arr[i], arr[i + 1] = arr[i + 1], arr[i]
                    i += 1
                break
            elif arr[mid] < del_num:
                left = mid + 1
            else:
                right = mid - 1
        return arr

    # Compute the quantile of the demand for use by the critic network
    def set_demand_statistics(self):
        if self.step_num ==0:
            self.demand_dy_sorted = [np.sort(demand[self.step_num+self.lead_time:self.looking_len+self.step_num+self.lead_time]) for demand in self.demand_list] 
            self.demand_LT = [[demand[idx] for idx in range(self.step_num,self.step_num+self.lead_time)] for demand in self.demand_list]
            self.demand_LT_mean = [np.mean(dl) for dl in self.demand_LT]

        else:  

            insert_index=self.looking_len+self.step_num+self.lead_time-1

            self.demand_LT = [[demand[idx] for idx in range(self.step_num,self.step_num+self.lead_time)] for demand in self.demand_list]
            
            

            for agent in range(self.agent_num):
                del_num = self.demand_list[agent][self.step_num+self.lead_time-1]
                insert_num = self.demand_list[agent][insert_index]
                self.demand_dy_sorted[agent]=self.del_and_insert(self.demand_dy_sorted[agent],del_num,insert_num)

                del_num_LT = self.demand_list[agent][self.step_num-1]
                insert_num_LT = self.demand_LT[agent][-1]
                self.demand_LT_mean[agent]=self.demand_LT_mean[agent]+(-del_num_LT+insert_num_LT)/self.lead_time

        self.demand_q5=[demand[int(0.05*(len(demand)-1))] for demand in self.demand_dy_sorted]
        self.demand_q25=[demand[int(0.25*(len(demand)-1))] for demand in self.demand_dy_sorted]
        self.demand_q50=[demand[int(0.5*(len(demand)-1))] for demand in self.demand_dy_sorted]
        self.demand_q75=[demand[int(0.75*(len(demand)-1))] for demand in self.demand_dy_sorted]
        self.demand_q95=[demand[int(0.95*(len(demand)-1))] for demand in self.demand_dy_sorted]

        
        
        

        if self.use_centralized_V:
            self.demand_LT_all_helper=[]
            for lt_d in self.demand_LT:
                self.demand_LT_all_helper +=lt_d

    # If no transshipment is made, after the lead time, how much inventory is left and what is the average shortage.
    def get_LT_inv_shortage(self):

        self.LT_inv=self.inventory[:]
        self.LT_avg_shortage=[0 for _ in range(self.agent_num)]
        for agent in range(self.agent_num):
            for lt in range(self.lead_time):
                inv = self.LT_inv[agent]+self.order[agent][lt]-self.demand_list[agent][lt+self.step_num]
                self.LT_inv[agent]=max(inv,0)
                self.LT_avg_shortage[agent]+=-min(inv,0)
            self.LT_avg_shortage[agent]=self.LT_avg_shortage[agent]/self.lead_time

        self.LT_inv_shortage=self.LT_inv+self.LT_avg_shortage


    # Critic network exclusive observations.
    def get_step_obs_critic(self, info_sharing, obs_step):
        actor_agent_obs = self.get_step_obs(info_sharing, False, obs_step, False)

        self.set_demand_statistics()

        self.get_LT_inv_shortage()


        sub_agent_obs = []

        for i in range(self.agent_num):
            actor_arr = actor_agent_obs[i]
            if info_sharing :
                if i==0:

                    demand_quantile_arr = (self.demand_q5+self.demand_q25+self.demand_q50+self.demand_q75+self.demand_q95) if 'quantile' in self.demand_info_for_critic else [] 
                    demand_quantile_arr = np.array(demand_quantile_arr)

                    demand_LT_arr = (self.demand_LT_all_helper if 'LT_all' in self.demand_info_for_critic else []) + (self.demand_LT_mean if 'LT_mean' in self.demand_info_for_critic else [])
                    demand_LT_arr = np.array(demand_LT_arr)
                    other_actor_arr = np.array([])
            else:

                demand_quantile_arr = [self.demand_q5[i],self.demand_q25[i],self.demand_q50[i],self.demand_q75[i],self.demand_q95[i]] if 'quantile' in self.demand_info_for_critic else [] 
                demand_quantile_arr = np.array(demand_quantile_arr)

                demand_LT_arr = (self.demand_LT[i] if 'LT_all' in self.demand_info_for_critic else []) + ([self.demand_LT_mean[i]] if 'LT_mean' in self.demand_info_for_critic else [])
                demand_LT_arr = np.array(demand_LT_arr)
                other_actor_arr = self.LT_inv_shortage
                other_actor_arr = np.array(other_actor_arr)


            if(self.normalize):
                arr = np.concatenate([actor_arr, demand_quantile_arr*2/self.demand_for_action_dim[i]-1, demand_LT_arr*2/self.demand_for_action_dim[i]-1, other_actor_arr*2/self.demand_for_action_dim[i]-1])
            else:
                arr = np.concatenate([actor_arr, demand_quantile_arr, demand_LT_arr,other_actor_arr])
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

        processed_rewards = [[self.alpha*i + (1-self.alpha)*np.mean(reward)] for i in reward]
        self.reward=[self.alpha*i + (1-self.alpha)*np.mean(reward) for i in reward]
        self.reward_cum=[self.reward_cum[i]+self.reward[i] for i in range(self.agent_num)]
        return processed_rewards
    
    def get_transship_revenue(self,action):
        cur_demand = [self.demand_list[i][self.step_num] for i in range(self.agent_num)]
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
            transship_revenue[i] = max((-self.C+self.P)*min(shortage_wihout_transship,ts) - self.S*ts - self.H*max(ts-shortage_wihout_transship,0),0)   
        return transship_revenue , transship_volume      
    
    def state_update_transship_revenue_sharing(self, action):
        """
        - Need to be implemented
        - Update system state and record some states
        - Inputs:
            - action: list, processed actions for each actor
            - Modify the inputs as you need
        - Outputs:
            - rewards: list, rewards for each actors (typically one-period costs for all actors)
        """
        cur_demand = [self.demand_list[i][self.step_num] for i in range(self.agent_num)]
        rewards_after=[]
        rewards_before = []

        self.step_num+=1

        # Calculate the profits before and after transshipment.
        for i in range(self.agent_num):
            self.order[i].append(action[i][0])
            self.transship_request[i]=action[i][1]
            inv_start_before = self.inventory[i]+self.order[i][0]
            inv_start=self.inventory[i]+self.order[i][0]+self.transship_request[i]
            self.inventory_start[i]=inv_start

            revenue_demand =cur_demand[i]*self.R if 'reward' in self.reward_type else 0
            norm_drift =cur_demand[i]*self.reward_norm_multiplier if 'norm' in self.reward_type else 0 

            
            # pure shipping cost
            self.shipping_cost_pure[i] = sum([self.shipping_cost_matrix[i][j]*self.transship_matrix[i][j] if self.transship_matrix[i][j]>0 else 0 for j in range(self.agent_num)])
            # Shipping cost + buying/selling cost
            self.shipping_cost_all[i] = self.shipping_cost_pure[i] + self.C*(action[i][1])

            self.ordering_cost[i] = (self.C*(action[i][0])+self.FIXED_COST*(1 if action[i][0]>0 else 0)) if self.pay_first else (self.C*(self.order[i][0])+self.FIXED_COST*(1 if self.order[i][0]>0 else 0))
            self.ordering_times[i]+=(1 if action[i][0]>0 else 0) if self.pay_first else (1 if self.order[i][0]>0 else 0)
            self.penalty_cost[i] = -self.P*min(inv_start-cur_demand[i],0)
            self.holding_cost[i] = self.H*max(inv_start-cur_demand[i],0)
            # reward after transshipment 
            reward= -self.ordering_cost[i]-self.shipping_cost_all[i]-self.holding_cost[i]-self.penalty_cost[i]+revenue_demand+norm_drift
            
            # reward before transshipment 
            reward_before= -self.ordering_cost[i]-self.H*max(inv_start_before-cur_demand[i],0)+self.P*min(inv_start_before-cur_demand[i],0)+revenue_demand+norm_drift
            
            self.demand_fulfilled[i] =  min(inv_start,cur_demand[i])
            self.shortage[i]=cur_demand[i]-inv_start
            self.inventory[i]=max(inv_start-cur_demand[i],0.)
            self.action_history[i].append(action[i])
            self.order[i]=self.order[i][1:]

            # On the last day, convert the remaining inventory in the warehouse to cost price
            if self.step_num > EPISODE_LEN-1 and (not self.train):
                reward=reward+(self.C)*max(inv_start-cur_demand[i],0)
                reward_before=reward_before+(self.C)*max(inv_start-cur_demand[i],0)
            rewards_after.append(reward)
            rewards_before.append(reward_before)

        rewards=[]
        # allocation of transshipment revnue
        for i in range(self.agent_num):
            reward = rewards_after[i]

            transship_reallocate = 0
            if self.transship_revenue_method == 'constant':
                transship_reallocate = -self.constant_transship_revenue*self.transship_request[i]
            else:
                raise Exception('wrong transship revenue allocated method')
            self.shipping_cost_all[i]-=transship_reallocate
            reward+=transship_reallocate
            rewards.append(reward)
            self.reward_selfish_cum[i]+=reward
            self.reward_selfish[i]=reward
            
        return rewards
    