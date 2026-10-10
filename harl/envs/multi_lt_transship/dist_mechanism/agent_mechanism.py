import numpy as np
from harl.envs.multi_lt_transship.dist_mechanism.base_mechanism import BaseMechanism

class AgentMechanism(BaseMechanism):
    '''
    该类用于实现mechanism agent的分配结果
    和其他不同，除了要给transship_intentions还要给allocate_ratio(每个agent被sample到的概率)
    '''
    def get_transship_matrix(self, transship_intentions, transship_ratio):
        '''
        该函数用于生成转运矩阵
        Args:
            transship_intentions: list, 每个agent是否愿意参与transship
            transship_ratio: list, 每个agent被sample到的概率

        Returns:
            transship_matrix: np.array, 转运矩阵，(i,j)表示j转运给i的量(正数表示j转给i, 负数表示i转给j)
        '''
        transship_matrix = np.zeros((self.agent_num, self.agent_num))
        # 如果transship_ratio为全0，则直接返回全0的transship_matrix
        if sum(transship_ratio) <= 0.5:
            return transship_matrix
        # 如果transship_intentions刚好匹配，则直接最小距离的分配方案为最优
        if abs(sum(transship_intentions)) <= 1e-5:
            transship_actual = transship_intentions.copy()
            return self.find_min_distance_matrix_alternative(transship_actual)
        transship_ratio = transship_ratio.copy()
        transship_intentions = transship_intentions.copy()
        transship_actual = [0] * self.agent_num
        # 判断是+的多还是-的多
        pos_over_tf = sum(transship_intentions) > 0
        # 少的那一方actual和intention一样，并把transship_ratio中对应的agent的值置为0
        sum_intentions_lack = 0  # 少的那一端的intention的和
        if pos_over_tf:
            for i in range(len(transship_intentions)):
                if transship_intentions[i] <= 0:
                    sum_intentions_lack += transship_intentions[i]
                    transship_actual[i] = transship_intentions[i]
                    transship_intentions[i] = 0
                    transship_ratio[i] = 0

        else:
            for i in range(len(transship_intentions)):
                if transship_intentions[i] >= 0:
                    sum_intentions_lack += transship_intentions[i]
                    transship_actual[i] = transship_intentions[i]
                    transship_intentions[i] = 0
                    transship_ratio[i] = 0

        # Renormalize transship_ratio to ensure it sums to 1
        ratio_sum = sum(transship_ratio)
        if ratio_sum > 0:
            transship_ratio = [r / ratio_sum for r in transship_ratio]
        else:
            # If all ratios are 0, use uniform distribution over non-zero intentions
            valid_indices = [i for i in range(len(transship_intentions)) if transship_intentions[i] != 0]
            if valid_indices:
                uniform_prob = 1.0 / len(valid_indices)
                transship_ratio = [uniform_prob if i in valid_indices else 0 for i in range(len(transship_ratio))]
            else:
                # All intentions are 0, no transshipment needed
                return np.zeros((self.agent_num, self.agent_num))

        # 按照transship_ratio sample生成一个array，这个array，表示每次sample到的agent的id
        # Ensure probabilities are valid
        p_arr = np.array(transship_ratio, dtype=np.float64)
        total_p = p_arr.sum()
        if total_p > 0:
            p_arr = p_arr / total_p
        else:
            # This case should be unreachable; fallback to uniform (to avoid crash)
            p_arr = np.ones(self.agent_num, dtype=np.float64) / self.agent_num
        # np.random.choice requires sum(p)==1
        sample_arr = np.random.choice(self.agent_num, size=1000, replace=True, p=p_arr)
        # 按照sample_arr的顺序，挨个fulfill demand
        for i in sample_arr:
            if pos_over_tf and sum_intentions_lack < 0:
                if transship_intentions[i] > 0:
                    transship_intentions[i] -= 1
                    transship_actual[i] += 1
                    sum_intentions_lack += 1
            elif not pos_over_tf and sum_intentions_lack > 0:
                if transship_intentions[i] < 0:
                    transship_intentions[i] += 1
                    transship_actual[i] -= 1
                    sum_intentions_lack -= 1
            else:
                break
        # 按照最小距离和生成transship_matrix
        # transship_matrix = self.find_min_distance_matrix(transship_actual)
        transship_matrix = self.find_min_distance_matrix_alternative(transship_actual)

        return transship_matrix

    def get_mechanism_result(self, transship_intentions, **kwargs):
        '''
        生成mechanism的支配结果，包含转运矩阵和payment
        '''
        transship_ratio = kwargs['transship_fulfill_ratio']
        transship_matrix = self.get_transship_matrix(transship_intentions, transship_ratio)
        payment = self.get_payment(transship_matrix)
        return transship_matrix, payment
