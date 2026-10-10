import copy
import numpy as np
from scipy.optimize import linprog
from harl.envs.multi_lt_transship.dist_mechanism.base_mechanism import BaseMechanism

class BestDistribution(BaseMechanism):
    '''
    偷看未来需求数据，与到货数据，用线性规划求解最优转运量
    '''
    def __init__(self, distance_matrix, allocate_args=None, others_info = None):
        super().__init__(distance_matrix, allocate_args, others_info)
        self.pre_ema = np.zeros((self.env.num_envs, self.env.num_agents), dtype=np.float32)

    def predict_demand(self, agent_idx, env_idx=0):
        """
        预测指定agent的需求
        Args:
            agent_idx: int, agent的索引
            env_idx: int, environment的索引
        Returns:
            demand: float, 当前需求
            demand_mean_est: float, 预测的平均需求
        """
        # Get lead time mean from LT data
        lead_time_mean = np.mean(self.env.LT[env_idx, agent_idx, :])
        alpha = self.env.args.get('alpha', 0.3)

        if self.env.step_index[env_idx] == 0:
            return self.env.demand[env_idx, agent_idx, 0], self.env.demand[env_idx, agent_idx, 0]

        demand = self.env.demand[env_idx, agent_idx, self.env.step_index[env_idx]]
        demand_mean_est = alpha * demand + (1 - alpha) * self.pre_ema[env_idx, agent_idx]
        self.pre_ema[env_idx, agent_idx] = demand_mean_est
        return demand, demand_mean_est

    def get_future_arrival(self, order_amounts, look_ahead_days, env_idx=0):
        '''
        预估未来look_ahead_days天的到货情况（基于在途库存）
        注意：best mechanism 偷看的是未来需求，而不是到货
        到货仍然使用父类的 est_future_arrival 方法

        Args:
            order_amounts: ndarray, 当前订单量
            look_ahead_days: int, 预测的天数
            env_idx: int, environment的索引
        Returns:
            arrival_list: dict, 未来每个agent每一天的到货量
        '''
        arrival_list = {}
        for agent_idx in range(self.agent_num):
            # Use parent class method to estimate future arrivals
            arrival_list[agent_idx] = self.est_future_arrival(
                agent_idx,
                order_amounts[agent_idx],
                look_ahead_days,
                env_idx
            )

        return arrival_list

    def get_transship_matrix(self, transship_intentions, env_idx=0):
        '''
        transship_intentions: list, 每个agent是否愿意参与transship
        env_idx: int, environment的索引
        '''
        # 参与transshipment的agent索引
        active_agents = [i for i in range(len(transship_intentions)) if transship_intentions[i] == 1]

        # 如果没有agent愿意参与transshipment，那么就不需要转运
        if len(active_agents) <= 1:
            return np.zeros((self.agent_num, self.agent_num))

        # 初始库存
        order_amounts = self.env.order_today[env_idx, :]
        next_start_inventory = self.env.inventory[env_idx, :]
        next_start_inventory = [next_start_inventory[i] for i in active_agents]

        # 预估的到货list
        arrival_list = []
        look_ahead_days = 10000

        # 确定look ahead days的范围 - 使用LT数据的平均值
        for agent_idx in range(self.agent_num):
            lt_agent = np.mean(self.env.LT[env_idx, agent_idx, :])
            look_ahead_days = int(min(look_ahead_days, lt_agent))

        look_ahead_days = max(1, look_ahead_days)  # 至少预测1天

        # 假设未来的需求就是这个了（偷看真实需求）
        next_demand = []
        current_step = self.env.step_index[env_idx]
        for agent_idx in range(self.agent_num):
            # 直接使用未来的真实需求数据
            future_demands = []
            for t in range(look_ahead_days):
                future_step = current_step + t
                if future_step < self.env.episode_length:
                    future_demands.append(self.env.demand[env_idx, agent_idx, future_step])
                else:
                    # 如果超出范围，用最后一个已知需求
                    future_demands.append(self.env.demand[env_idx, agent_idx, self.env.episode_length - 1])
            next_demand.append(future_demands)
        next_demand = [next_demand[i] for i in active_agents]

        # 获取未来到货（偷看真实到货）
        arrival_list_dict = self.get_future_arrival(order_amounts, look_ahead_days, env_idx)
        arrival_list = [arrival_list_dict[i] for i in active_agents]

        # 创建优化变量
        I = active_agents  # 只考虑愿意参与的agent
        J = active_agents  # 只考虑愿意参与的agent
        T = list(range(look_ahead_days))

        N = len(I)  # Number of agents
        T_len = len(T)  # Number of time periods

        # Variable indices
        tran_indices = {}
        index = 0
        for t in range(T_len):
            for i in range(N):
                for j in range(N):
                    tran_indices[(i, j, t)] = index
                    index += 1

        s_inv_indices = {}
        for t in range(T_len):
            for i in range(N):
                s_inv_indices[(i, t)] = index
                index += 1

        inv_max_0_indices = {}
        for t in range(T_len):
            for i in range(N):
                inv_max_0_indices[(i, t)] = index
                index += 1

        inv_indices = {}
        for t in range(T_len):
            for i in range(N):
                inv_indices[(i, t)] = index
                index += 1

        shortage_indices = {}
        for t in range(T_len):
            for i in range(N):
                shortage_indices[(i, t)] = index
                index += 1

        # Total number of variables
        V = index
        bounds = []

        # Bounds for tran variables (non-negative)
        for _ in range(N*N*(T_len)):
            bounds.append((0, None))

        # Bounds for s_inv variables (unbounded)
        for _ in range(N*(T_len)):
            bounds.append((None, None))

        # Bounds for inv_max_0 variables (non-negative)
        for _ in range(N*(T_len)):
            bounds.append((0, None))

        # Bounds for inv variables (unbounded)
        for _ in range(N*T_len):
            bounds.append((None, None))

        # Bounds for shortage variables (non-negative)
        for _ in range(N*T_len):
            bounds.append((0, None))

        # equality constraints
        A_eq = []
        b_eq = []

        for i in range(N):
            # model.addConstr(s_inv[i, 0] == next_start_inventory[i])
            row = np.zeros(V)
            idx = s_inv_indices[(i, 0)]
            row[idx] = 1
            A_eq.append(row)
            b_eq.append(next_start_inventory[i])

        for t in range(T_len):
            for i in range(N):
                # reactive:
                # model.addConstr(inv[i, t] == s_inv[i, t] - tran.sum(i, "*", t) + tran.sum("*", i, t) - next_demand[i][t])
                # proactive:
                # model.addConstr(inv[i, t] == s_inv[i, t] - next_demand[i][t]
                row = np.zeros(V)
                row[s_inv_indices[(i, t)]] = 1
                row[inv_indices[(i, t)]] = -1
                if self.env.reactive_tf:
                    for j in range(N):
                        row[tran_indices[(i, j, t)]] = -1
                        row[tran_indices[(j, i, t)]] = 1
                A_eq.append(row)
                b_eq.append(next_demand[i][t])
                # reactive:
                # model.addConstr(s_inv[i, t+1] == inv[i, t] + arrival_list[i][t])
                # proactive:
                # model.addConstr(s_inv[i, t+1] == inv[i, t] - tran.sum(i, "*", t) + tran.sum("*", i, t) + arrival_list[i][t])
                if t < T_len - 1:
                    row = np.zeros(V)
                    row[inv_indices[(i, t)]] = 1
                    row[s_inv_indices[(i, t+1)]] = -1
                    if not self.env.reactive_tf:
                        for j in range(N):
                            row[tran_indices[(i, j, t)]] = -1
                            row[tran_indices[(j, i, t)]] = 1
                    A_eq.append(row)
                    b_eq.append(-arrival_list[i][t])


        A_ub = []
        b_ub = []
        for i in range(N):
            # proactive:
            # model.addConstr(tran.sum(i, "*", t) <= inv_max_0[i, t])
            if not self.env.reactive_tf:
                row = np.zeros(V)
                for j in range(N):
                    row[tran_indices[(i, j, 0)]] = 1
                row[inv_max_0_indices[(i, 0)]] = -1
                A_ub.append(row)
                b_ub.append(0)


        for i in range(N):
            for t in range(T_len):
                # model.addConstr(inv_max_0[i, t] >= inv[i, t])
                row = np.zeros(V)
                row[inv_max_0_indices[(i, t)]] = -1
                row[inv_indices[(i, t)]] = 1
                A_ub.append(row)
                b_ub.append(0)
                # model.addConstr(shortage[i, t] >= -inv[i, t])
                row = np.zeros(V)
                row[inv_indices[(i, t)]] = -1
                row[shortage_indices[(i, t)]] = -1
                A_ub.append(row)
                b_ub.append(0)

        c = np.zeros(V)
        # Holding costs for inv_max_0 variables
        for t in range(T_len):
            for i in range(N):
                idx = inv_max_0_indices[(i, t)]
                i_agent = I[i]
                c[idx] = self.env.H[i_agent]

        # Shortage costs for shortage variables
        for t in range(T_len):
            for i in range(N):
                idx = shortage_indices[(i, t)]
                i_agent = I[i]
                c[idx] = self.env.B[i_agent]

        # Transportation costs for tran variables
        for t in range(T_len):
            for i in range(N):
                for j in range(N):
                    idx = tran_indices[(i, j, t)]
                    i_agent = I[i]
                    j_agent = J[j]
                    c[idx] = self.env.distance_matrix[i_agent][j_agent] * self.env.shipping_cost_per_distance

        A_eq = np.array(A_eq)
        b_eq = np.array(b_eq)
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs', options={'time_limit': 60.0})

        # 初始化transship_amounts，只更新参与transshipment的agent
        transship_amounts = [0 for _ in range(self.agent_num)]

        if result.success:
            x = result.x
            for (i, j, t), idx in tran_indices.items():
                if t == 0 and i!=j:
                    i_agent = I[i]
                    j_agent = J[j]
                    transship_amount = round(x[idx] - 0.1)

                    # 最大转运量阈值
                    cur_demand = self.env.demand[env_idx, i_agent, self.env.step_index[env_idx]]
                    tran_max_thres = max(0, self.env.inventory[env_idx, i_agent]
                                      if self.env.reactive_tf else
                                      self.env.inventory[env_idx, i_agent] - cur_demand)
                    # 检查是否超过库存可接受范围
                    if transship_amounts[i_agent] - transship_amount >= -tran_max_thres:
                        transship_amounts[i_agent] -= transship_amount
                        transship_amounts[j_agent] += transship_amount
                    else:
                        # 如果超过可用库存，则只转运可用的最大数量
                        available_amount = tran_max_thres + transship_amounts[i_agent]
                        transship_amounts[i_agent] = -tran_max_thres
                        transship_amounts[j_agent] += available_amount
        else:
            print("Optimization failed:", result.message)

        transship_matrix = self.find_min_distance_matrix_alternative(transship_amounts)
        return transship_matrix
