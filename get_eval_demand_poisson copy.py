import os
import numpy as np
from envs.generator import poisson
DEMAND_MAX = 20
EPISODE_LEN = 400
num_eval = 30
# agent_num = 3
np.random.seed(1234567)
for agent in range(10):
    path='test_data/poisson2/{}'.format(agent)
    if not os.path.exists(path):
        os.makedirs(path)
    
    for i in range(num_eval):
        demand_list = poisson(EPISODE_LEN, DEMAND_MAX/2,DEMAND_MAX).demand_list
        with open(path+'/{}.txt'.format(i),'w') as f:
            for demand in demand_list:
                f.write(str(demand))
                f.write('\n')
            f.close()