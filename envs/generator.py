import numpy as np
import random
from random import randint
import math
import os
import chardet
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

""" new packages """
from statsmodels.graphics.api import qqplot
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima.model import ARIMAResults
import pandas as pd

class merton(object):

    def __init__(self, length, max_demand = 20):

        base = int(max_demand/2)
        start = 0
        var = 15
        delta = 0.01
        delta_t = var
        u = 0.5*delta*delta
        a = 0
        b = 0.01
        lamda = var
        
        while(True):
            self.demand_list = []
            self.drump = []
            self.no_drump = []
            self.no_drump.append(start)
            self.demand_list.append(start)
            self.drump.append(0)
            for i in range(length):
                Z = np.random.normal(0, 1)
                N = np.random.poisson(lamda)
                Z_2 = np.random.normal(0, 2)
                M = a*N + b*(N**0.5)*Z_2
                new_X = self.demand_list[-1] + u - 0.5*delta*delta + (delta_t**0.5)*delta*Z + M
                self.demand_list.append(new_X)
            self.demand_list = [int(math.exp(i)*base) for i in self.demand_list]
            if(np.mean(self.demand_list)>0 and np.mean(self.demand_list)<max_demand):
                break
            
        for i in range(len(self.demand_list)):
            self.demand_list[i] = min(max_demand, self.demand_list[i])  

    def __getitem__(self, key = 0):
        return self.demand_list[key]

class stationary_possion(object):

    def __init__(self, length, max_demand=20):
        self.demand_list = np.random.poisson(max_demand/2, length)
    
    def __getitem__(self, key = 0):
        return self.demand_list[key]

class poisson(object):

    def __init__(self, length, mean,max_demand=20):
        self.demand_list = np.random.poisson(mean,length)
        self.demand_list=np.clip(self.demand_list,0,max_demand)
        self.demand_list = [int(i) for i in self.demand_list]
    
    def __getitem__(self, key = 0):
        return self.demand_list[key]


class normal(object):
    def __init__(self, length,mean,var, max_demand=20):
        self.demand_list = np.random.normal(mean, var, length)
        self.demand_list=np.clip(self.demand_list,0,max_demand)
        self.demand_list = [int(i) for i in self.demand_list]
    
    def __getitem__(self, key = 0):
        return self.demand_list[key]
    
class uniform(object):
    def __init__(self, length, max_demand=20):
        self.demand_list = np.random.uniform(0,max_demand, length)
        self.demand_list=np.clip(self.demand_list,0,max_demand)
        self.demand_list = [int(i) for i in self.demand_list]
    
    def __getitem__(self, key = 0):
        return self.demand_list[key]
    
SKU = "SKU029"
agents_idx = ['DC001_SKU029', 'DC002_SKU029', 'DC003_SKU029', 'DC004_SKU029', 'DC005_SKU029', 'DC006_SKU029', 'DC007_SKU029', 'DC008_SKU029', 'DC009_SKU029', 'DC010_SKU029', 'DC011_SKU029', 'DC012_SKU029', 'DC013_SKU029', 'DC014_SKU029', 'DC015_SKU029', 'DC016_SKU029', 'DC017_SKU029', 'DC018_SKU029']

class shanshu(object):

    def __init__(self, length, max_demand = 20, i = 0):
        """ need to be specified in this self that which warehous_sku does this agent belongs to """
        ARIMA_path = os.path.join("envs", "ARIMA_results", "ARIMA-" + agents_idx[i] + ".txt")
        #print(ARIMA_path)
        model_fit = ARIMAResults.load(ARIMA_path)
        simulated_np = model_fit.simulate(length)
        simulated_np_clipped_and_squashed = np.clip(simulated_np, 0, simulated_np.max())*(max_demand/simulated_np.max())
        self.demand_list = np.round(simulated_np_clipped_and_squashed)

    def __getitem__(self, key = 0):
        return self.demand_list[key]

class shanshu_arima(object):

    def __init__(self, SKU_id, agent, length, demand_max_for_clip):

        """ need to be specified in this self that which warehous_sku does this agent belongs to """
        ARIMA_path = os.path.join("envs", "ARIMA_results_" + SKU_id, "ARIMA-" + str(agent) + ".txt")
        #print(ARIMA_path)
        model_fit = ARIMAResults.load(ARIMA_path)
        simulated_np = model_fit.simulate(length)
        simulated_np_clipped_and_squashed = np.clip(simulated_np, 0, demand_max_for_clip[agent])
        self.demand_list = np.round(simulated_np_clipped_and_squashed)

    def __getitem__(self, key = 0):
        return self.demand_list[key]
      
TRAIN_PTH = ["./train_data/shanshu_sampling/0/", "./train_data/shanshu_sampling/1/", "./train_data/shanshu_sampling/2/"]

class shanshu_sampling(object):
    def __init__(self, agent, length, max_demand = 20):
        files_0 = os.listdir(TRAIN_PTH[0])
        n_eval = len(files_0)
        i=randint(0,n_eval-1)
        self.demand_list = []
        files=os.listdir(TRAIN_PTH[agent])
        with open(TRAIN_PTH[agent] + files[i],'rb') as f:
                d=f.read()
                encoding = chardet.detect(d)['encoding']
        with open(TRAIN_PTH[agent] + files[i],  encoding=encoding) as f:
            lines = f.readlines()
            for line in lines:
                self.demand_list.append(int(line))
            

        for i in range(len(self.demand_list)):
            self.demand_list[i] = min(max_demand, self.demand_list[i])  

    def __getitem__(self, key = 0):
        return self.demand_list[key]
    
class random_fragment(object):
    def __init__(self, agent, length, train_path,max_demand,start):
        cur_data = train_path+'/'+str(agent)+'.csv'
        cur_df = pd.read_csv(cur_data)
        cur_df_eval = cur_df.iloc[start:start+length].copy()
        self.demand_list = cur_df_eval['sale'].tolist()

        for i in range(len(self.demand_list)):
            self.demand_list[i] = min(max_demand, self.demand_list[i])  

    def __getitem__(self, key = 0):
        return self.demand_list[key]
    
class random_resample(object):
    def __init__(self, agent, length, train_path,max_demand):
        cur_data = train_path+'/'+str(agent)+'.csv'
        cur_df = pd.read_csv(cur_data)
        random_elements = np.random.choice(cur_df['sale'], size=length, replace=False)
        self.demand_list = random_elements.tolist()

        for i in range(len(self.demand_list)):
            self.demand_list[i] = min(max_demand, self.demand_list[i])  

    def __getitem__(self, key = 0):
        return self.demand_list[key]
