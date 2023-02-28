import os
from tensorboardX import SummaryWriter
from math import sinh,cosh,tanh
import random
from pathlib import Path
log_dir =Path('try_tb')/'logs'
print(log_dir)
writter = SummaryWriter(str(log_dir).replace('\\','/'))

# def del_folder(path):
#     if not os.path.exists(path):
#         print('{} not exist'.format(path))
#         return None
#     ls = os.listdir(path)
#     for i in ls:
#         c_path=os.path.join(path,i)
#         if os.path.isdir(c_path):
#             del_folder(c_path)
#         else:
#             os.remove(c_path)

# for j in range(10):
#     for i in range(5):
#         dict_tb={}
#         dict_tb['sin']=sinh(i)+random.random()
#         dict_tb['cos']=cosh(i)+random.random()
#         if j>0 and i==0:
#             writter.close()
#             del_folder(str(log_dir/'try').replace('\\','/'))
#             writter=SummaryWriter(str(log_dir).replace('\\','/'))
#         writter.add_scalars('try',dict_tb,global_step=i)
#         writter.add_scalars('try',{'tanh':tanh(i)},global_step=i)
#         writter.add_scalar('i',i+random.random(),i)

from math import exp
print(exp(0.08*2))
        
        
