import math
import numpy as np

def replace_and_calculate_sd(arr, mean, sd, n, old_val, new_val):
    # 计算原始数组的总和
    total = mean * n

    # 计算原始数组的平方和
    sq_sum = sum(x**2 for x in arr)

    
    return np.sqrt((sq_sum -old_val**2+new_val**2)/n-((total-old_val+new_val)/n)**2)


arr=[1,3,8,10,5,4]
arr_copy=[1,3,8,10,5,4]
mean=np.mean(arr)
sd = np.std(arr)


# sq_sum = sum(x**2 for x in arr)
# sq_mean = mean**2
# print(sq_sum/)
n=len(arr)
old_val=3
new_val=5

arr.remove(old_val)
arr.append(new_val)
print(np.std(arr))
print(replace_and_calculate_sd(arr_copy, mean, sd, n, old_val, new_val))