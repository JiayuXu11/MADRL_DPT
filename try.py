import os
import math
from envs.generator_copy import merton
import numpy as np
from matplotlib.pyplot import plot
# EVAL_PTH = ["./eval_data/merton/0/", "./eval_data/merton/1/", "./eval_data/merton/2/"]
EVAL_PTH = ["C:/Users/Jerry/Desktop/data/eval_data/merton/0/","C:/Users/Jerry/Desktop/data/eval_data/merton/1/","C:/Users/Jerry/Desktop/data/eval_data/merton/2/"]

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
            with open(EVAL_PTH[j] + files[i], "rb") as f:
                lines = f.readlines()
                for line in lines:
                    data.append(float(line))
            eval_data_i.append(data)
        eval_data.append(eval_data_i)
    # print(np.array(eval_data).shape)
    return n_eval, eval_data

# n_eval,eval_data=get_eval_data()
# all_demand=0
# for n in range(n_eval):
#     for agent in range(3):
#         for day in range(200):
#             all_demand+=eval_data[n][agent][day]
# mean_demand=all_demand/n_eval/3/200
# print(mean_demand)
# print(mean_demand*3)
# print(mean_demand*2.4)
# a= merton(200, 20).demand_list
# plot(a)
# print(a)

def calculate_updated_sd(mean, sq_sum, n, old_val, new_val):
    sq_sum = sq_sum -old_val**2+new_val**2
    return sq_sum,np.sqrt(sq_sum/n-mean**2)


def del_and_insert(arr, del_num, insert_num):
    # 二分查找要删除的数字的位置
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == del_num:
            # 找到要删除的数字，将其替换为插入的数字
            arr[mid] = insert_num
            # 从插入数字的位置开始向前遍历，直到找到一个比当前位置小的数或者到达数组的开头
            i = mid
            while i > 0 and arr[i - 1] > insert_num:
                arr[i], arr[i - 1] = arr[i - 1], arr[i]
                i -= 1
            # 从插入数字的位置开始向后遍历，直到找到一个比当前位置大的数或者到达数组的结尾
            i = mid
            while i < len(arr) - 1 and arr[i + 1] < insert_num:
                arr[i], arr[i + 1] = arr[i + 1], arr[i]
                i += 1
            break
        elif arr[mid] < del_num:
            left = mid + 1
        else:
            right = mid - 1
    return arr

import time
episode_length=200
arr = np.sort(np.random.randint(10,20,episode_length))
print(arr)
list_arr=list(arr)
s_t=time.time()
for i in range(10000):
    id=np.random.randint(0,200)
    arr[id]=np.random.randint(10,20)
    arr=np.sort(arr)
    arr[int(0.05*episode_length)]
    arr[int(0.25*episode_length)]
    arr[int(0.5*episode_length)]
    arr[int(0.75*episode_length)]
    arr[int(0.95*episode_length)]
    # np.mean(arr[:10])
print(time.time()-s_t)

# s_t=time.time()
# for i in range(10000):
#     id=np.random.randint(0,200)
#     arr[id]=np.random.randint(10,20)
#     # arr=np.sort(arr)
#     np.quantile(arr,0.05)
#     np.quantile(arr,0.25)
#     np.quantile(arr,0.5)
#     np.quantile(arr,0.75)
#     np.quantile(arr,0.95)
# print(time.time()-s_t)

# s_t=time.time()
# for i in range(10000):
#     id=np.random.randint(0,200)
#     del_num = arr[id]
#     insert_num=np.random.randint(10,20)
#     del_and_insert(arr,del_num,insert_num)
#     # arr=np.sort(arr)
#     arr[int(0.05*episode_length)]
#     arr[int(0.25*episode_length)]
#     arr[int(0.5*episode_length)]
#     arr[int(0.75*episode_length)]
#     arr[int(0.95*episode_length)]
# print(time.time()-s_t)


s_t=time.time()
m=np.mean(arr)
sq_sum = np.sum(arr**2)
s=np.std(arr)
for i in range(1000):
    id=np.random.randint(0,200)
    del_num = arr[id]
    insert_num=np.random.randint(10,20)
    arr[id]=insert_num
    # arr=np.sort(arr)
    m=m+(-del_num+insert_num)/200
    # print((-del_num**2+insert_num**2)/episode_length)
    # print(arr)
    # print(np.std(arr))
    sq_sum,s=calculate_updated_sd(m,sq_sum,200,del_num,insert_num)
    # print(s)
    if math.isnan(s):
        print((-del_num**2+insert_num**2)/episode_length)
        print(np.std(arr))
        print(s)
    # np.std(arr)
    # np.mean(arr[:10])
print(time.time()-s_t)

s_t=time.time()
m=np.mean(arr)
for i in range(10000):
    id=np.random.randint(0,200)
    del_num = arr[id]
    insert_num=np.random.randint(10,20)
    arr[id]=np.random.randint(10,20)
    # arr=np.sort(arr)
    np.mean(arr)
    np.std(arr)
    # np.std(arr)
    # np.mean(arr[:10])
print(time.time()-s_t)

# s_t=time.time()
# mean(list_arr)
# print(time.time()-s_t)

arr = np.sort(np.random.randint(10,20,(5,episode_length)))

s_t=time.time()
m=np.mean(arr)
for i in range(10000):
    id=np.random.randint(0,200)
    del_num = arr[0,id]
    insert_num=np.random.randint(10,20)
    arr[0,id]=np.random.randint(10,20)
    del_num = arr[1,id]
    arr[1,id]=np.random.randint(10,20)
    # arr[2,id]=np.random.randint(10,20)
    # arr[3,id]=np.random.randint(10,20)

print(time.time()-s_t)

arr=[np.sort(np.random.randint(10,20,episode_length)) for i in range(5)]
s_t=time.time()
# m=np.mean(arr)
for i in range(1000000):
    len(arr[0])
    # arr[2,id]=np.random.randint(10,20)
    # arr[3,id]=np.random.randint(10,20)

print(time.time()-s_t)



s_t=time.time()
# m=np.mean(arr)
for i in range(1000000):
    arr[0].size+1
    arr[0].size+2
    arr[0].size+3
    # arr[2,id]=np.random.randint(10,20)
    # arr[3,id]=np.random.randint(10,20)

print(time.time()-s_t)

s_t=time.time()
# m=np.mean(arr)
for i in range(1000000):
    # p=arr[0].size
    len(arr[0])+1
    len(arr[0])+2
    len(arr[0])+3
    # arr[2,id]=np.random.randint(10,20)
    # arr[3,id]=np.random.randint(10,20)

print(time.time()-s_t)