import os
from envs.generator_copy import merton
import numpy as np
from matplotlib.pyplot import plot
import chardet
#EVAL_PTH = ["./eval_data/SKU029/0/", "./eval_data/SKU029/1/", "./eval_data/SKU029/2/"]
agent_num = 3
price = 3
EVAL_PTH = ["test_data/merton/{}/".format(i) for i in range(agent_num)]

# EVAL_PTH = ["./shanshu_sampling_test/0/", "./shanshu_sampling_test/1/", "./shanshu_sampling_test/2/"]

def get_eval_data():
    """
    - Need to be implemented
    - Load local demand data for evaluation
    - Inputs:
        - Modify the inputs as you need
    - Outputs:
        - n_eval: int, number of demand sequences (also number of episodes in one evaluation)
        - eval_data: list, demand data for evaluation
    """
    files_0 = os.listdir(EVAL_PTH[0])
    n_eval = len(files_0)
    eval_data=[]
  
    for i in range(n_eval):
        eval_data_i=[]
        for j in range(len(EVAL_PTH)):
            files=os.listdir(EVAL_PTH[j])
            data = []
            with open(EVAL_PTH[j] + files[i],'rb') as f:
                d=f.read()
                encoding = chardet.detect(d)['encoding']
        
            with open(EVAL_PTH[j] + files[i], encoding=encoding) as f:
                lines = f.readlines()
                for line in lines:
                    data.append(float(line))
            eval_data_i.append(data)
        eval_data.append(eval_data_i)
    # print(np.array(eval_data).shape)
    return n_eval, eval_data

n_eval,eval_data=get_eval_data()
all_demand=0
for n in range(n_eval):
    for agent in range(agent_num):
        for day in range(200):
            all_demand+=eval_data[n][agent][day]
mean_demand=all_demand/n_eval/agent_num/200
print(mean_demand)
print(mean_demand*price)
print(mean_demand*2.4)
# a= merton(200, 20).demand_list
# plot(a)
# print(a)
