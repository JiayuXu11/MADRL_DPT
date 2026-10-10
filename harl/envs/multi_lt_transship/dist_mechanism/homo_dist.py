import numpy as np
from harl.envs.multi_lt_transship.dist_mechanism.base_mechanism import BaseMechanism

class HomoDistribution(BaseMechanism):
    '''
    该类用于在不考虑距离情况下的分配计划，砍一刀
    '''
    # homo distance下的撮合机制
    def get_transship_matrix(self, transship_intentions):
        transship_matrix = np.zeros_like(self.distance_matrix)
        transship_intentions = transship_intentions.copy()
        i=0
        while(sum(transship_intentions) != 0):
            if sum(transship_intentions) > 0:
                if (transship_intentions[i] > 0):
                    transship_intentions[i] += -1
            elif sum(transship_intentions) < 0:
                if (transship_intentions[i] < 0):
                    transship_intentions[i] += 1
            i+=1
            i=0 if i > self.agent_num-1 else i

        # 转换为transship_matrix格式
        for a1 in range(self.agent_num):
            for a2 in range(self.agent_num):
                if transship_intentions[a1]*transship_intentions[a2]<0:
                    tran_amount = min(abs(transship_intentions[a1]),abs(transship_intentions[a2]))
                    transship_matrix[a1][a2]=tran_amount if transship_intentions[a1]>0 else -tran_amount
                    transship_matrix[a2][a1]=-transship_matrix[a1][a2]
                    transship_intentions[a1]-=transship_matrix[a1][a2]
                    transship_intentions[a2]-=transship_matrix[a2][a1]

        return transship_matrix
