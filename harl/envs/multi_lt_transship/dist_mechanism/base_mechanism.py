import numpy as np



class BaseMechanism:
    def __init__(self, distance_matrix, allocate_args=None, others_info = None):
        self.distance_matrix = distance_matrix
        self.agent_num = distance_matrix.shape[0]
        self.allocate_args = allocate_args
        self.shipping_order = None

        self.env = others_info
        self.revenue_allocation_method = self.env.revenue_allocation_method
        self.constant_transshipment_price = self.env.constant_transshipment_price
        self.price_arr = np.ones((self.env.num_envs, self.agent_num), dtype=np.float32) * self.constant_transshipment_price# shape: (E, 1)
        self.shipping_cost_matrix = self.env.shipping_cost_matrix

    def get_transship_matrix(self, transship_intentions):
        '''
        根据每个人的intentions生成转运矩阵, (i,j)表示j转运给i的量(正数表示j转给i, 负数表示i转给j)
        如果该mechanism是不参考其他agent的intentions而生成转运矩阵的,
        那么这里的transship_intentions就是表达每个人是否愿意接受transship支配, 1表示愿意, 0表示不愿意。
        '''
        return np.zeros((self.agent_num, self.agent_num))

    def get_payment(self, transship_matrix, price_arr=None):
        '''
        根据转运矩阵生成每个agent的payment: [agent0_payment, agent1_payment, ...]
        正数表示agent需要支付, 负数表示agent需要收到payment
        price_arr: ndarray, 每个agent的拿货价格
        '''
        shipping_cost_list = (self.shipping_cost_matrix * np.maximum(transship_matrix, 0)).sum(axis=1)
        if self.revenue_allocation_method == 'constant':
            payment = transship_matrix.sum(axis=1) * self.constant_transshipment_price + shipping_cost_list
        elif self.revenue_allocation_method == 'agent_specific' and price_arr is not None:
            payment = transship_matrix.sum(axis=1) * price_arr + shipping_cost_list
        elif self.revenue_allocation_method == 'dual':
            from harl.envs.multi_lt_transship.dist_mechanism.anupindi_mechanism import AnupindiDistribution
            anup_mec = AnupindiDistribution(self.distance_matrix, self.allocate_args, self.env)
            transship_matrix = anup_mec.get_transship_matrix(self.transship_intentions)
            payment = anup_mec.get_payment(transship_matrix)

        else:
            raise NotImplementedError
        return payment

    def get_vectorized_payment(self, transship_matrix, price_arr=None):
        '''
        该函数用于生成vectorized的payment
        Args:
            transship_matrix: ndarray, (E, A, A), 转运矩阵
            price_arr: ndarray, (E, A), 每个env下agent的拿货价格
        Returns:
            payment: ndarray, 每个agent的payment
        '''
        payment = (self.shipping_cost_matrix * np.maximum(transship_matrix, 0)).sum(axis=2)
        if self.revenue_allocation_method == 'dual':
            from harl.envs.multi_lt_transship.dist_mechanism.anupindi_mechanism import AnupindiDistribution
            for env_idx in range(transship_matrix.shape[0]):
                anup_mec = AnupindiDistribution(self.distance_matrix, self.allocate_args, self.env)
                _ = anup_mec.get_transship_matrix(transship_matrix[env_idx].sum(axis=1))
                payment[env_idx] = anup_mec.get_payment(transship_matrix[env_idx])
        else:
            if price_arr is None:
                price_arr = self.env._get_price_array_for_payment()
            payment = transship_matrix.sum(axis=2) * price_arr + payment
        return payment

    def get_mechanism_result(self, transship_intentions, **kwargs):
        '''
        生成mechanism的支配结果，包含转运矩阵和payment
        '''
        self.transship_intentions = transship_intentions
        transship_matrix = self.get_transship_matrix(transship_intentions, kwargs.get('env_idx', 0))
        price_arr = kwargs.get('price_arr', None)
        if price_arr is None:
            payment = self.get_payment(transship_matrix)
        else:
            payment = self.get_payment(transship_matrix, price_arr)
        return transship_matrix, payment

    def est_future_arrival(self, agent_idx, order_amount, look_ahead_days, env_idx=0):
        '''
        预估未来look_ahead_days天的到货情况（向量化版本）
        注意：不能使用delivery_schedule，因为它包含未来信息
        只能使用intransit_hist（在途库存历史）来预估

        intransit_hist[env_idx, agent_idx, -i] 表示 i 天前下的订单量
        这些订单会在 lt_avg - (i+1) 天后到货

        Args:
            agent_idx: int, agent的索引
            order_amount: float, 当前订单量
            look_ahead_days: int, 预测的天数
            env_idx: int, environment的索引
        Returns:
            future_arrival: list, 未来每一天的到货量
        '''
        future_arrival = [0 for _ in range(int(look_ahead_days))]

        # 获取该agent的平均lead time
        avg_lt = self.env.lt_avg_agents[agent_idx]

        # 使用intransit_hist来预估在途订单的到货
        # intransit_hist shape: (num_envs, num_agents, max_lt)
        # intransit_hist[env_idx, agent_idx, i] 表示 i 天前下的订单量
        # 这些订单会在 avg_lt - (i+1) 天后到货
        max_lt = self.env.max_lt
        for i in range(max_lt):
            order_qty = self.env.intransit_hist[env_idx, agent_idx, i]
            if order_qty > 0:
                # 该订单是在 max_lt - i 天前下的，会在 avg_lt - (i+1) 天后到货
                days_until_arrival = avg_lt - (max_lt - i) - 1
                if 0 <= days_until_arrival < look_ahead_days:
                    future_arrival[days_until_arrival] += order_qty
                elif days_until_arrival < 0:
                    future_arrival[0] += order_qty

        # 添加当前订单的预期到货（基于平均LT）
        if avg_lt < look_ahead_days and order_amount > 0:
            future_arrival[avg_lt] += order_amount

        return future_arrival

    def find_min_distance_matrix_alternative(self, transship_actual):
        '''
        按照潜在路径（pos -> neg）从小到大排列然后依次 fulfill transship matrix
        '''
        # 初始化transship_matrix
        transship_matrix = np.zeros((self.agent_num, self.agent_num))

        # 记录未满足的供需
        demand = np.array([max(val, 0) for val in transship_actual])
        supply = np.array([max(-val, 0) for val in transship_actual])

        # 找出所有潜在的供需配对，按照距离从小到大排序
        potential_paths = []
        for i in range(self.agent_num):
            for j in range(self.agent_num):
                if i != j and supply[i] > 0 and demand[j] > 0:
                    potential_paths.append((i, j, self.distance_matrix[i, j]))

        # 按距离从小到大排序
        potential_paths.sort(key=lambda x: x[2])

        # 依次fulfill每个供需对
        for i, j, _ in potential_paths:
            if supply[i] == 0 or demand[j] == 0:
                continue

            # 选择可运送的最大值，满足供需约束
            transfer_amount = min(supply[i], demand[j])
            transship_matrix[i][j] = transfer_amount
            transship_matrix[j][i] = -transfer_amount

            # 更新剩余供需
            supply[i] -= transfer_amount
            demand[j] -= transfer_amount

            # 如果供需已经满足，跳过
            if supply[i] == 0:
                continue
            if demand[j] == 0:
                continue

        return -transship_matrix
