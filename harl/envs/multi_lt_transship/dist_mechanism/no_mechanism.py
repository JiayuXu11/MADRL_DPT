import numpy as np
import random
from harl.envs.multi_lt_transship.dist_mechanism.base_mechanism import BaseMechanism
class NoMechanism(BaseMechanism):
    '''
    不transship的机制，直接返回0的payment和0的transship_matrix
    '''
    def __init__(self, distance_matrix, allocate_args=None, others_info = None):
        self.agent_num = distance_matrix.shape[0]

    def get_transship_matrix(self, transship_intentions, env_idx=0):
        return np.zeros((self.agent_num, self.agent_num))

    def get_payment(self, transship_matrix, price_arr=None, env_idx=0):
        return [0] * self.agent_num
