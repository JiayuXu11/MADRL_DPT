import numpy as np
import random
import math
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

class merton(object):

    def __init__(self, length, max_demand = 20):

        self.demand_list=[1,2,3,5]
            
        for i in range(len(self.demand_list)):
            self.demand_list[i] = min(max_demand, self.demand_list[i])  

    def __getitem__(self, key = 0):
        return self.demand_list[key]

class stationary_possion(object):

    def __init__(self, length, max_demand=20):
        self.demand_list = np.random.poisson(max_demand/2, length)
    
    def __getitem__(self, key = 0):
        return self.demand_list[key]
