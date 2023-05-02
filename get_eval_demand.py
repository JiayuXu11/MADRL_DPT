import os
import numpy as np
from envs.generator import merton
DEMAND_MAX = 20
EPISODE_LEN = 400
num_eval = 30
agent_num = 3
np.random.seed(0)
for agent in range(agent_num,5):
    path='eval_data/merton/{}'.format(agent)
    if not os.path.exists(path):
        os.makedirs(path)
    
    for i in range(num_eval):
        demand_list = merton(EPISODE_LEN, DEMAND_MAX)
        with open(path+'/{}.txt'.format(i),'w') as f:
            for demand in demand_list:
                f.write(str(demand))
                f.write('\n')
            f.close()


