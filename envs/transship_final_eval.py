import numpy as np
from . import generator
import os 
import random
import chardet


DISTANCE = [
    np.array(
       [[   0, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000],
       [1000,    0, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000],
       [1000, 1000,    0, 1000, 1000, 1000, 1000, 1000, 1000, 1000],
       [1000, 1000, 1000,    0, 1000, 1000, 1000, 1000, 1000, 1000],
       [1000, 1000, 1000, 1000,    0, 1000, 1000, 1000, 1000, 1000],
       [1000, 1000, 1000, 1000, 1000,    0, 1000, 1000, 1000, 1000],
       [1000, 1000, 1000, 1000, 1000, 1000,    0, 1000, 1000, 1000],
       [1000, 1000, 1000, 1000, 1000, 1000, 1000,    0, 1000, 1000],
       [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000,    0, 1000],
       [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000,    0]]),
    np.array(
       [[   0, 1017, 1660,  913, 1205, 1266,  757, 1544, 1947, 1421],
       [1017,    0, 1242, 1313, 1734,  642, 1200, 1647, 1885, 2000],
       [1660, 1242,    0, 1258, 1573,  749, 1292,  990,  948, 1806],
       [ 913, 1313, 1258,    0,  566, 1207,  300,  808, 1246,  836],
       [1205, 1734, 1573,  566,    0, 1618,  682,  879, 1314,  412],
       [1266,  642,  749, 1207, 1618,    0, 1154, 1311, 1459, 1885],
       [ 757, 1200, 1292,  300,  682, 1154,    0,  946, 1372,  944],
       [1544, 1647,  990,  808,  879, 1311,  946,    0,  591, 1045],
       [1947, 1885,  948, 1246, 1314, 1459, 1372,  591,    0, 1445],
       [1421, 2000, 1806,  836,  412, 1885,  944, 1045, 1445,    0]]),
    np.array(
       [[   0, 1701,  636,  821,  874, 1362, 1449,  857,  300,  587],
       [1701,    0, 1364, 1801, 1360, 1138,  838, 1102, 1707, 2000],
       [ 636, 1364,    0, 1062,  869, 1248, 1062,  601,  622,  974],
       [ 821, 1801, 1062,    0,  693, 1127, 1762, 1021,  872,  747],
       [ 874, 1360,  869,  693,    0,  751, 1387,  654,  915, 1024],
       [1362, 1138, 1248, 1127,  751,    0, 1425,  911, 1398, 1524],
       [1449,  838, 1062, 1762, 1387, 1425,    0,  994, 1435, 1785],
       [ 857, 1102,  601, 1021,  654,  911,  994,    0,  871, 1147],
       [ 300, 1707,  622,  872,  915, 1398, 1435,  871,    0,  612],
       [ 587, 2000,  974,  747, 1024, 1524, 1785, 1147,  612,    0]]),
    np.array(
       [[   0,  576,  300, 1125, 1083,  976, 1129, 1417,  676,  868],
       [ 576,    0,  449, 1249, 1306, 1169, 1481, 1836,  849, 1206],
       [ 300,  449,    0, 1092, 1189, 1071, 1204, 1545,  650,  933],
       [1125, 1249, 1092,    0, 2000, 1917,  898, 1581,  605,  767],
       [1083, 1306, 1189, 2000,    0,  302, 1694, 1511, 1571, 1536],
       [ 976, 1169, 1071, 1917,  302,    0, 1667, 1552, 1479, 1487],
       [1129, 1481, 1204,  898, 1694, 1667,    0,  853,  875,  431],
       [1417, 1836, 1545, 1581, 1511, 1552,  853,    0, 1431,  997],
       [ 676,  849,  650,  605, 1571, 1479,  875, 1431,    0,  617],
       [ 868, 1206,  933,  767, 1536, 1487,  431,  997,  617,    0]])
 ]
# DISTANCE = np.array([[0,820,1411,770,872],[820,0,2404,624,420],[1411,2404,0,1785,2187],[770,624,1785,0,557],[872,420,2187,557,0]])  # 一个二维数组，表示不同仓库之间的distance

S_I = 10
S_O = 10

DEMAND_MAX = 20

EPISODE_LEN = 200

FIXED_COST = 5


#====================================================================================


class Env(object):

    def __init__(self,args):
        
        #============================================================================
        # Define the member variables you need here.
        # The following three memeber variables must be defined
        self.agent_num = args.num_involver
        self.lead_time = args.lead_time
        self.demand_info_for_critic=args.demand_info_for_critic
        
        # cost parameter
        self.H = args.H  # holding cost
        self.R = args.R  # selling price per unit (only used for reward)
        self.P = args.P  # penalty cost
        self.C = args.C  # ordering cost
        self.shipping_cost_per_distance = args.shipping_cost_per_distance

        self.generator_method = args.generator_method

        # 考不考虑distance不同问题
        self.homo_distance=args.homo_distance
        self.mini_pooling = args.mini_pooling
                
        # 设置distance
        self.distance = DISTANCE[0 if self.homo_distance else args.distance_index][:self.agent_num,:self.agent_num]

        # 货到付款， 还是先钱后货
        self.pay_first= args.pay_first

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
        self.shipping_order = self.get_shipping_order(self.distance)
        self.shipping_cost_matrix = self.phi(self.distance)

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
            demand_list = [np.array(generator.merton(EPISODE_LEN,DEMAND_MAX).demand_list) for _ in range(self.agent_num)]
            # demand_list = [generator.merton(EPISODE_LEN, DEMAND_MAX), generator.merton(EPISODE_LEN, DEMAND_MAX),generator.merton(EPISODE_LEN, DEMAND_MAX)]
        elif(self.generator_method=='poisson'):
            demand_list = [generator.poisson(EPISODE_LEN,DEMAND_MAX/2,DEMAND_MAX).demand_list for _ in range(self.agent_num)]
            # demand_list=[generator.poisson(EPISODE_LEN,DEMAND_MAX/2,DEMAND_MAX),generator.poisson(EPISODE_LEN,DEMAND_MAX/2,DEMAND_MAX),generator.poisson(EPISODE_LEN,DEMAND_MAX/2,DEMAND_MAX)]
        elif(self.generator_method=='normal'):
            demand_list = [generator.normal(EPISODE_LEN,DEMAND_MAX/2,DEMAND_MAX/4,DEMAND_MAX).demand_list for _ in range(self.agent_num)]
            # demand_list=[generator.normal(EPISODE_LEN,DEMAND_MAX/2,DEMAND_MAX/4,DEMAND_MAX),generator.normal(EPISODE_LEN,DEMAND_MAX/2,DEMAND_MAX/4,DEMAND_MAX),generator.normal(EPISODE_LEN,DEMAND_MAX/2,DEMAND_MAX/4,DEMAND_MAX)]
        elif(self.generator_method=='uniform'):
            demand_list = [generator.uniform(EPISODE_LEN,DEMAND_MAX).demand_list for _ in range(self.agent_num)]
            # demand_list=[generator.uniform(EPISODE_LEN,DEMAND_MAX),generator.uniform(EPISODE_LEN,DEMAND_MAX),generator.uniform(EPISODE_LEN,DEMAND_MAX)]
        elif(self.generator_method=='shanshu'):
             demand_list = [generator.shanshu(EPISODE_LEN,DEMAND_MAX,i).demand_list for i in range(self.agent_num)]
            #  demand_list=[generator.shanshu(EPISODE_LEN,DEMAND_MAX,0),generator.shanshu(EPISODE_LEN,DEMAND_MAX,1),generator.shanshu(EPISODE_LEN,DEMAND_MAX,2)]
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
        # demand_info_num = len(self.demand_info_for_critic)
        demand_info_num = (5 if 'quantile' in self.demand_info_for_critic else 0 )+(self.lead_time if 'LT_all' in self.demand_info_for_critic else 0 )+(1 if 'LT_mean' in self.demand_info_for_critic else 0 )
        demand_dim = demand_info_num*self.agent_num if info_sharing else demand_info_num*1
        patial_info_sharing_dim = self.agent_num*2
        # critic的输入不包含当期需求
        obs_diff = self.agent_num if info_sharing else 1
        return self.get_obs_dim(info_sharing, obs_step) + demand_dim + patial_info_sharing_dim - obs_diff


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
                order_amount=action[i]//(DEMAND_MAX+1)
                order_amounts.append(order_amount)
                transship_amount=max(action[i]%(DEMAND_MAX+1)-DEMAND_MAX/2,-self.inventory[i]-self.order[i][0])
                transship_amounts.append(transship_amount)
                
        elif self.action_type == 'multi_discrete'or self.action_type == 'central_multi_discrete':
            # action = [(np.argmax(i[:self.action_dim[0]]),np.argmax(i[self.action_dim[0]:])) for i in actions]
            action = [(i[0],i[1]) for i in actions]
            for i in range(self.agent_num):
                order_amount=action[i][0]
                order_amounts.append(order_amount)
                transship_amount=max(action[i][1]-DEMAND_MAX,-self.inventory[i]-self.order[i][0])
                transship_amounts.append(transship_amount)

        elif self.action_type == 'continue':
            action = actions
            for i in range(self.agent_num):
                order_amounts.append(action[i][0])
                transship_amounts.append(max(action[i][1],-self.inventory[i]-self.order[i][0]))

        else:
            raise Exception("wrong action_type")
        
        self.transship_intend=transship_amounts.copy()

        if self.homo_distance:
            self.transship_merge_homo(transship_amounts)
        elif self.mini_pooling["flag"]:
            self.transship_merge_mp(transship_amounts, threshold=self.mini_pooling['threshold'], how=self.mini_pooling['how'])
        else:
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
    
    # homo distance下的撮合机制 
    def transship_merge_homo(self,transship_amounts):
        # 撮合transship
        # ## 1. 按比例分配,和下面那个挨个砍一刀一起用才是完整版。效果不好，感觉还不如挨个砍一刀。
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
        # 2. 若仍未撮合成功，则挨个砍一刀
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
        
        # 转换为transship_matrix格式
        for a1 in range(self.agent_num):
            for a2 in range(self.agent_num):
                if transship_amounts[a1]*transship_amounts[a2]<0:
                    tran_amount = min(abs(transship_amounts[a1]),abs(transship_amounts[a2]))
                    self.transship_matrix[a1][a2]=tran_amount if transship_amounts[a1]>0 else -tran_amount
                    self.transship_matrix[a2][a1]=-self.transship_matrix[a1][a2]
                    transship_amounts[a1]-=self.transship_matrix[a1][a2]
                    transship_amounts[a2]-=self.transship_matrix[a2][a1]

    # mini_pooling 撮合机制 
    def transship_merge_mp(self, transship_amounts, threshold, how):
        # 撮合transship
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
                
            
            
            # transship_pos=sum([t if t>0 else 0 for t in transship_amounts])
            # transship_neg=sum([t if t<0 else 0 for t in transship_amounts])
        # # 撮合transship
        # # ## 1. 按比例分配,和下面那个挨个砍一刀一起用才是完整版。效果不好，感觉还不如挨个砍一刀。
        # if self.ratio_transsship:
        #     transship_pos=sum([t if t>0 else 0 for t in transship_amounts])
        #     transship_neg=sum([t if t<0 else 0 for t in transship_amounts])
        #     if sum(transship_amounts) < 0:
        #         ratio = -transship_pos/transship_neg
        #         for i in range(len(transship_amounts)):
        #             transship_amounts[i]= round(ratio*transship_amounts[i],0) if transship_amounts[i]<0 else transship_amounts[i]
        #     elif sum(transship_amounts) > 0:
        #         ratio = -transship_neg/transship_pos
        #         for i in range(len(transship_amounts)):
        #             transship_amounts[i]= round(ratio*transship_amounts[i],0) if transship_amounts[i]>0 else transship_amounts[i]
        # # 2. 若仍未撮合成功，则挨个砍一刀
        # i=0
        # while(sum(transship_amounts) != 0):
        #     if sum(transship_amounts) > 0:
        #         if (transship_amounts[i] > 0):
        #             transship_amounts[i] += -1
        #     elif sum(transship_amounts) < 0:
        #         if (transship_amounts[i] < 0):
        #             transship_amounts[i] += 1
        #     i+=1
        #     i=0 if i > self.agent_num-1 else i
        
        # 转换为transship_matrix格式
        # for a1 in range(self.agent_num):
        #     for a2 in range(self.agent_num):
        #         if transship_amounts[a1]*transship_amounts[a2]<0:
        #             tran_amount = min(abs(transship_amounts[a1]),abs(transship_amounts[a2]))
        #             self.transship_matrix[a1][a2]=tran_amount if transship_amounts[a1]>0 else -tran_amount
        #             self.transship_matrix[a2][a1]=-self.transship_matrix[a1][a2]
        #             transship_amounts[a1]-=self.transship_matrix[a1][a2]
        #             transship_amounts[a2]-=self.transship_matrix[a2][a1]
        
    def get_step_obs(self, info_sharing, obs_step, demand_today=True):
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
                if i==0:
                    base_arr = np.array(self.inventory + ([(self.demand_list[k][self.step_num-1] if self.step_num>0 else DEMAND_MAX/2) for k in range(self.agent_num)] if demand_today else []))
                    order_arr = np.array(order_all)
            else:
                base_arr = np.array([self.inventory[i],(self.demand_list[i][self.step_num-1] if self.step_num>0 else DEMAND_MAX/2) ]) if demand_today else np.array([self.inventory[i]])
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
    
    def del_and_insert(self, arr, del_num, insert_num):
    # 二分查找要删除的数字的位置
        left, right = 0, len(arr) - 1
        while left <= right:
            mid = (left + right) // 2
            if arr[mid] == del_num:
                # 找到要删除的数字，将其替换为插入的数字
                arr[mid] = insert_num
                # 从插入数字的位置开始向前遍历，直到找到一个比当前位置小的数或者到达数组的开头
                i = mid
                while i > 0 and arr[i - 1] > insert_num:
                    arr[i], arr[i - 1] = arr[i - 1], arr[i]
                    i -= 1
                # 从插入数字的位置开始向后遍历，直到找到一个比当前位置大的数或者到达数组的结尾
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

    # 统计需求quantile，以供critic network使用
    def set_demand_statistics(self):
        if self.step_num ==0:
            self.demand_dy_sorted = [np.sort(demand[min(self.step_num+self.lead_time,self.episode_max_steps):min(self.looking_len+self.step_num+self.lead_time,self.episode_max_steps)]) for demand in self.demand_list] 
            self.demand_LT = [[(demand[idx] if idx<self.episode_max_steps else 0 )for idx in range(self.step_num,self.step_num+self.lead_time)] for demand in self.demand_list]
            self.demand_LT_mean = [np.mean(dl) for dl in self.demand_LT]

        elif self.step_num<self.episode_max_steps:  

            insert_index=self.looking_len+self.step_num+self.lead_time-1

            self.demand_LT = [[(demand[idx] if idx<self.episode_max_steps else 0 )for idx in range(self.step_num,self.step_num+self.lead_time)] for demand in self.demand_list]
            
            

            for agent in range(self.agent_num):
                del_num = self.demand_list[agent][min(self.step_num+self.lead_time-1,self.episode_max_steps-1)]
                insert_num = self.demand_list[agent][insert_index] if insert_index<self.episode_max_steps else self.demand_q50[agent]
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

    # 假如不transship，lead time时间后，还剩多少库存，平均缺货多少
    def get_LT_inv_shortage(self):

        self.LT_inv=self.inventory[:]
        self.LT_avg_shortage=[0 for _ in range(self.agent_num)]
        for agent in range(self.agent_num):
            for lt in range(self.lead_time):
                inv = self.LT_inv[agent]+self.order[agent][lt]-self.demand_list[agent][min(lt+self.step_num,self.episode_max_steps-1)]
                self.LT_inv[agent]=max(inv,0)
                self.LT_avg_shortage[agent]+=-min(inv,0)
            self.LT_avg_shortage[agent]=self.LT_avg_shortage[agent]/self.lead_time

        self.LT_inv_shortage=self.LT_inv+self.LT_avg_shortage


    # critic network 专属obs
    def get_step_obs_critic(self, info_sharing, obs_step):
        actor_agent_obs = self.get_step_obs(info_sharing, obs_step, False)

        self.set_demand_statistics()

        self.get_LT_inv_shortage()


        sub_agent_obs = []

        for i in range(self.agent_num):
            actor_arr = actor_agent_obs[i]
            if info_sharing :
                if i==0:
                    # demand_mean_arr = (self.demand_mean if 'all_mean' in self.demand_info_for_critic else []) + (self.demand_mean_dy if 'mean' in self.demand_info_for_critic else [])
                    # demand_mean_arr = np.array(demand_mean_arr)
                    demand_quantile_arr = (self.demand_q5+self.demand_q25+self.demand_q50+self.demand_q75+self.demand_q95) if 'quantile' in self.demand_info_for_critic else [] 
                    demand_quantile_arr = np.array(demand_quantile_arr)
                    # demand_std_arr = (self.demand_std if 'all_std' in self.demand_info_for_critic else []) + (self.demand_std_dy if 'std' in self.demand_info_for_critic else [])
                    # demand_std_arr = np.array(demand_std_arr)
                    demand_LT_arr = (self.demand_LT_all_helper if 'LT_all' in self.demand_info_for_critic else []) + (self.demand_LT_mean if 'LT_mean' in self.demand_info_for_critic else [])
                    demand_LT_arr = np.array(demand_LT_arr)
                    other_actor_arr = self.LT_inv_shortage
                    other_actor_arr = np.array(other_actor_arr)
            else:
                # demand_mean_arr = ([self.demand_mean[i]] if 'all_mean' in self.demand_info_for_critic else []) + ([self.demand_mean_dy[i]] if 'mean' in self.demand_info_for_critic else [])
                # demand_mean_arr = np.array(demand_mean_arr)
                demand_quantile_arr = [self.demand_q5[i],self.demand_q25[i],self.demand_q50[i],self.demand_q75[i],self.demand_q95[i]] if 'quantile' in self.demand_info_for_critic else [] 
                demand_quantile_arr = np.array(demand_quantile_arr)
                # demand_std_arr = ([self.demand_std[i]] if 'all_std' in self.demand_info_for_critic else []) + ([self.demand_std_dy[i]] if 'std' in self.demand_info_for_critic else [])
                # demand_std_arr = np.array(demand_std_arr)
                demand_LT_arr = (self.demand_LT[i] if 'LT_all' in self.demand_info_for_critic else []) + ([self.demand_LT_mean[i]] if 'LT_mean' in self.demand_info_for_critic else [])
                demand_LT_arr = np.array(demand_LT_arr)
                other_actor_arr = self.LT_inv_shortage
                other_actor_arr = np.array(other_actor_arr)


            if(self.normalize):
                # arr = np.concatenate([actor_arr,demand_mean_arr*2/DEMAND_MAX-1., demand_std_arr/DEMAND_MAX-1., demand_quantile_arr*2/DEMAND_MAX-1, demand_LT_arr*2/DEMAND_MAX-1, other_actor_arr*2/DEMAND_MAX-1])
                arr = np.concatenate([actor_arr, demand_quantile_arr*2/DEMAND_MAX-1, demand_LT_arr*2/DEMAND_MAX-1, other_actor_arr*2/DEMAND_MAX-1])
            else:
                # arr = np.concatenate([actor_arr,demand_mean_arr, demand_std_arr, demand_quantile_arr, demand_LT_arr])
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
        # if(self.train):
        #     processed_rewards = [[self.alpha*i + (1-self.alpha)*np.mean(reward)] for i in reward]
        # else:
        #     processed_rewards = [[i] for i in reward]
        processed_rewards = [[self.alpha*i + (1-self.alpha)*np.mean(reward)] for i in reward]
        self.reward=[self.alpha*i + (1-self.alpha)*np.mean(reward) for i in reward]
        self.reward_cum=[self.reward_cum[i]+self.reward[i] for i in range(self.agent_num)]
        return processed_rewards
    
    
    # transship的收益
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
        cur_demand = [self.demand_list[i][self.step_num] for i in range(self.agent_num)]
        rewards_after=[]
        rewards_before = []

        self.step_num+=1

        # 计算transship前后收益
        for i in range(self.agent_num):
            self.order[i].append(action[i][0])
            self.transship_request[i]=action[i][1]
            inv_start_before = self.inventory[i]+self.order[i][0]
            inv_start=self.inventory[i]+self.order[i][0]+self.transship_request[i]
            self.inventory_start[i]=inv_start

            revenue_demand =cur_demand[i]*self.R if 'reward' in self.reward_type else 0
            norm_drift =cur_demand[i]*self.reward_norm_multiplier if 'norm' in self.reward_type else 0 

            
            # 纯运费
            self.shipping_cost_pure[i] = sum([self.shipping_cost_matrix[i][j]*self.transship_matrix[i][j] if self.transship_matrix[i][j]>0 else 0 for j in range(self.agent_num)])
            # 运费+买卖货的费用
            self.shipping_cost_all[i] = self.shipping_cost_pure[i] + self.C*(action[i][1])

            self.ordering_cost[i] = (self.C*(action[i][0])+FIXED_COST*(1 if action[i][0]>0 else 0)) if self.pay_first else (self.C*(self.order[i][0])+FIXED_COST*(1 if self.order[i][0]>0 else 0))
            self.ordering_times[i]+=(1 if action[i][0]>0 else 0) if self.pay_first else (1 if self.order[i][0]>0 else 0)
            self.penalty_cost[i] = -self.P*min(inv_start-cur_demand[i],0)
            self.holding_cost[i] = self.H*max(inv_start-cur_demand[i],0)
            # transship 后的reward 
            reward= -self.ordering_cost[i]-self.shipping_cost_all[i]-self.holding_cost[i]-self.penalty_cost[i]+revenue_demand+norm_drift
            
            # transship前的reward
            reward_before= -self.ordering_cost[i]-self.H*max(inv_start_before-cur_demand[i],0)+self.P*min(inv_start_before-cur_demand[i],0)+revenue_demand+norm_drift
            
            self.demand_fulfilled[i] =  min(inv_start,cur_demand[i])
            self.shortage[i]=cur_demand[i]-inv_start
            self.inventory[i]=max(inv_start-cur_demand[i],0.)
            self.action_history[i].append(action[i])
            self.order[i]=self.order[i][1:]

            # 最后一天将仓库内剩余货品按成本价折算
            if self.step_num > EPISODE_LEN-1:
                reward=reward+(self.C)*max(inv_start-cur_demand[i],0)
                reward_before=reward_before+(self.C)*max(inv_start-cur_demand[i],0)
            rewards_after.append(reward)
            rewards_before.append(reward_before)

        rewards=[]
        # 把transship收益分了
        for i in range(self.agent_num):
            reward = rewards_after[i]

             # transship 收益分配
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
    