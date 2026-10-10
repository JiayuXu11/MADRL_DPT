import os
import random
import numpy as np
import inspect
import shutil

def filter_args(func, args):
    """Filter out the args that are not in the function signature."""
    sig = inspect.signature(func)
    valid_args = {k: v for k, v in args.items() if k in sig.parameters}
    return valid_args

def del_folder(folder_path):
    if os.path.isdir(folder_path):
        shutil.rmtree(folder_path, ignore_errors=True)
        print(f'{folder_path} 文件夹及其内容已删除')

def filter_scalar_dict(data):
    """防止出现dict中的value为bool/非scalar，导致tensorboard的记录出现问题"""
    new_data = {}
    for key, value in data.items():
        # scalar should be 0D, 如果不是就丢掉
        if np.isscalar(value) and not isinstance(value, bool):
            new_data[key] = value
        # 如果是bool就转换成int
        elif isinstance(value, bool):
            new_data[key] = int(value)
    return new_data
