import numpy as np
import copy
from scipy.optimize import linprog
from harl.envs.multi_lt_transship.dist_mechanism.base_mechanism import BaseMechanism

class AnupindiDistribution(BaseMechanism):
    '''
    该类用于复现Anupindi等人的分配机制
    '''
    def get_transship_matrix(self, transship_intentions, env_idx=0):
        '''
        Generates the transshipment matrix.
        Args:
            transship_intentions: list, indicates whether each agent is willing to participate in transshipment
            env_idx: int, the environment index to process

        Returns:
            transship_matrix: np.array, the transshipment matrix where (i,j) represents the amount agent j transships to agent i
        '''
        # Get current demand for all agents in this environment
        cur_demand = self.env.demand[env_idx, :, self.env.step_index[env_idx]]
        transship_matrix = np.zeros((self.agent_num, self.agent_num))

        if self.env.reactive_tf and self.env.transship_type in ['force_to_transship', 'half_autonomy']:
            extra_inventory = [max(0, self.env.inventory[env_idx, agent_id] - cur_demand[agent_id])
                               if transship_intentions[agent_id]==1 else 0
                               for agent_id in range(self.agent_num)]
            extra_demand = [max(0, -self.env.inventory[env_idx, agent_id] + cur_demand[agent_id])
                               if transship_intentions[agent_id]==1 else 0
                               for agent_id in range(self.agent_num)]
        elif self.env.transship_type in ['force_to_transship', 'half_autonomy'] and not self.allocate_args.get('clairvoyant_tf', False):
            # Estimate future arrivals using parent class method
            # Get next period arrival (1 day ahead) for each agent
            # est_future_arrival = np.array([


            # ])
            # Use current demand as estimate for next period demand
            demand_est = self.env.demand_est[env_idx, :, self.env.step_index[env_idx]]

            # Use vectorized computation
            inventory_arr = self.env.inventory[env_idx, :]
            # future_diff = np.minimum(0, -demand_est + est_future_arrival)
            extra_inventory_arr = np.maximum(0, inventory_arr - demand_est)
            extra_demand_arr = np.maximum(0, -inventory_arr + demand_est)

            # Apply transship_intentions mask
            extra_inventory = [extra_inventory_arr[i] if transship_intentions[i]==1 else 0 for i in range(self.agent_num)]
            extra_demand = [extra_demand_arr[i] if transship_intentions[i]==1 else 0 for i in range(self.agent_num)]
        elif self.allocate_args.get('clairvoyant_tf', False):
            # For clairvoyant mode, we can "peek" at future demand (but still use est_future_arrival for arrivals)
            next_step = min(self.env.step_index[env_idx] + 1, self.env.episode_length - 1)
            # Use parent class method to estimate future arrivals
            arrival_list = np.array([
                self.est_future_arrival(agent_idx, 0, 1, env_idx)[0] if 1 > 0 else 0
                for agent_idx in range(self.agent_num)
            ])
            next_demand = self.env.demand[env_idx, :, next_step]

            extra_inventory = [max(0, self.env.inventory[env_idx, agent_id] - cur_demand[agent_id] +
                               min(0, - next_demand[agent_id] + arrival_list[agent_id]))
                               if transship_intentions[agent_id]==1 else 0
                                for agent_id in range(self.agent_num)]
            extra_demand = [max(0, -self.env.inventory[env_idx, agent_id] + cur_demand[agent_id]
                                        + next_demand[agent_id] - arrival_list[agent_id])
                                        if transship_intentions[agent_id]==1 else 0
                                        for agent_id in range(self.agent_num)]
        elif self.env.transship_type == 'pure_autonomy':
            extra_demand = np.maximum(transship_intentions, 0)
            extra_inventory = -np.minimum(transship_intentions, 0)
        else:
            extra_inventory = [0 for _ in range(self.agent_num)]
            extra_demand = [0 for _ in range(self.agent_num)]


        self.extra_inventory = extra_inventory
        self.extra_demand = extra_demand
        num_vars = self.agent_num * self.agent_num

        # Objective coefficients c (negated because linprog performs minimization)
        c = []
        for i in range(self.agent_num):
            for j in range(self.agent_num):
                rev = (self.env.P[j]
                       + self.env.H[i]
                       + self.env.B[j]
                       - self.env.C[i]
                       - self.distance_matrix[i,j] * self.env.shipping_cost_per_distance)
                c.append(-rev)  # Negate for minimization

        # Bounds on variables (all variables >= 0)
        bounds = [(0, None)] * num_vars

        # Constraints: A_ub x <= b_ub
        A_ub = []
        b_ub = []

        # Column sum constraints (transship in quantities)
        for j in range(self.agent_num):
            A_row = [0.0] * num_vars
            for i in range(self.agent_num):
                idx = i * self.agent_num + j
                A_row[idx] = 1.0
            A_ub.append(A_row)
            b_ub.append(extra_demand[j])

        # Row sum constraints (transship out quantities)
        for i in range(self.agent_num):
            A_row = [0.0] * num_vars
            for j in range(self.agent_num):
                idx = i * self.agent_num + j
                A_row[idx] = 1.0
            A_ub.append(A_row)
            b_ub.append(extra_inventory[i])

        # Suppress output by setting options
        options = {'disp': False}

        # Solve the LP using linprog with HiGHS
        res = linprog(c=c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs', options=options)

        if res.success:
            # Extract solution
            Q_sol = np.array(res.x).reshape((self.agent_num, self.agent_num))
            transship_matrix = Q_sol.T - Q_sol
            fun = -res.fun  # Negate to get the original maximized value

            # Get dual variables (shadow prices)
            duals = res.ineqlin.marginals  # Corrected line
            # The order of duals corresponds to the constraints in A_ub
            # First the column sum constraints, then the row sum constraints
            in_price = -duals[:self.agent_num]
            out_price = -duals[self.agent_num:2*self.agent_num]
            self.prices = list(out_price) + list(in_price)

            # 初始化transship_amounts，只更新参与transshipment的agent
            transship_amounts = [0 for _ in range(self.agent_num)]

            # 检查transship_matrix是否满足库存限制，并更正为整数
            for i in range(self.agent_num):
                for j in range(self.agent_num):
                    if i != j and transship_matrix[i, j] < 0:
                        transship_amount = round(-transship_matrix[i, j] - 0.1)

                        # 最大转运量阈值
                        tran_max_thres = max(0, self.env.inventory[env_idx, i])
                        # 检查是否超过库存可接受范围
                        if transship_amounts[i] - transship_amount >= -tran_max_thres:
                            transship_amounts[i] -= transship_amount
                            transship_amounts[j] += transship_amount

                        else:
                            # 如果超过可用库存，则只转运可用的最大数量
                            available_amount = tran_max_thres + transship_amounts[i]
                            transship_amounts[i] = -tran_max_thres
                            transship_amounts[j] += available_amount

        else:
            print("Optimization was stopped with status:", res.message)

        transship_matrix = self.find_min_distance_matrix_alternative(transship_amounts)

        self.fun = res.fun
        return transship_matrix

    def get_payment(self, transship_matrix, payment_arr=None):
        '''
        Generates the payment for each agent based on the transshipment matrix.
        Positive values indicate payments to be made, negative values indicate payments to be received.
        '''
        payment_type = self.allocate_args.get('payment_type', 'dual')
        if payment_type != 'dual':
            # 使用父类的get_payment方法
            return super().get_payment(transship_matrix)
        else:
            out_price = self.prices[0:self.agent_num]
            in_price = self.prices[self.agent_num:]
            transship_qty = np.sum(transship_matrix, axis=1)
            shipping_cost = np.sum(self.env.shipping_cost_matrix * np.maximum(transship_matrix, 0))

            # Calculate revenue allocation
            # rev_allocation = np.multiply(out_price, np.maximum(-transship_qty, 0)) + np.multiply(in_price, np.maximum(transship_qty, 0))

            rev_allocation_new = np.multiply(out_price, self.extra_inventory) + np.multiply(in_price, self.extra_demand)


            # Calculate revenue without payment
            rev_no_pay = (np.multiply([self.env.P[i] + self.env.B[i] for i in range(self.agent_num)], np.maximum(transship_qty, 0)) +
                        np.multiply([self.env.H[i] - self.env.C[i] for i in range(self.agent_num)], np.maximum(-transship_qty, 0)))
            true_rev_all = sum(rev_no_pay) - shipping_cost
            rev_allocation_new = (true_rev_all / sum(rev_allocation_new)) * rev_allocation_new if sum(rev_allocation_new) != 0 else rev_allocation_new
            # payment = rev_no_pay - rev_allocation
            payment = rev_no_pay - rev_allocation_new

            return payment
