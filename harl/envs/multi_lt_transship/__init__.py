import numpy as np
import os
DISTANCE_DIR = 'harl/envs/multi_lt_transship/distance/'
TEST_DIR = 'harl/envs/multi_lt_transship/test_data/{}/'
EVAL_DIR = 'harl/envs/multi_lt_transship/eval_data/{}/'
TRAIN_DIR = 'harl/envs/multi_lt_transship/train_data/{}/'
SHANSHU_GEN_SOURCE = 'harl/envs/multi_lt_transship/gen_source/{}/{}.csv'

# 按顺序取DISTANCE_DIR下的文件名，读取文件，转换为int类型，存入DISTANCE列表中
DISTANCE_LIST = [
    np.loadtxt(os.path.join(DISTANCE_DIR, file), delimiter=',', encoding='utf-8')
    for file in sorted(os.listdir(DISTANCE_DIR)) if file.endswith(".txt")
]


# 向量化环境配置参数
# 参数格式：标量（所有agent相同）或列表（按索引顺序，每个agent不同值）
ENV_ARGS_DEFAULT = {
    'num_agents': 3,
    # 以下是gen_func: [*gen_args]形式的说明
    # merton: [std, max_num], poisson: [mean, max_num],
    # normal: [mean, std, max_num], uniform: [min_num, max_num],
    # kim: [meanMin, meanMax, stdMin, stdMax, max_num]
    # constant: [value]
    # kim_merton: [mean_std, mean_max, std_max, std_std, max_num]
    # 声明LT生成器列表，若有多个，则表示有多段LT
    # 示例（多段LT）:
    # 'LT_generators': [{'gen_func':'uniform', 'gen_args':{'min_num':1, 'max_num':3}},
    #                   {'gen_func':'poisson', 'gen_args':{'mean':2, 'max_num':4}},
    #                   {'gen_func':'constant', 'gen_args':{'value': 1}}]
    'LT_generators': [{'gen_func':'poisson', 'gen_args':{'mean':2, 'max_num':6}}],

    # 声明demand生成器（所有agent使用相同配置）
    # 示例: {'gen_func':'merton', 'gen_args':{'std':4, 'max_num':20}}
    'demand_generators': {'gen_func':'kim_merton', 'gen_args':{'mean_std': 4, 'mean_max': 30, 'std_max': 5, 'std_std': 1, 'max_num': 50}},
    # 声明shipping loss生成器（所有agent使用相同配置）
    # 若不考虑shipping loss，则 {'gen_func':'constant', 'gen_args':{'value':0}}
    'shipping_loss_generators': {'gen_func':'constant', 'gen_args':{'value':0}},  #{'gen_func':'uniform', 'gen_args':{'min_num':0, 'max_num':0.2}},

    # backlog 还是 lost sales
    'backlog_tf': True,

    # cost parameters
    # 标量: 所有agent使用相同值
    # 列表: 每个agent使用不同值，如 [0.1, 0.15, 0.2] 表示3个agent的不同值

    'H': 0.1,  # 0.05,0.1,0.15,0.2

    'B': 1.0,  # 0.5,1.0,1.5,2.0

    'P': 0,

    'C': 0,

    'K': 0,

    'shipping_cost_per_distance': 2e-4,

    # distance matrix index, 会自动取出对应的distance matrix
    'distance_matrix_index': 1,

    # transshipment mechanism
    'reactive_tf': False, # 是否允许reactive transship
    'transship_type': 'pure_autonomy', # 转运类型 'pure_autonomy': 自己决策transship的量, 'half_autonomy': 只决策是否参与transship, 'no': 不参与transship, 'force_to_transship': 强制参与转运
    'mechanism_agent_type': 'none',  # 'pricing', 'cen_allocation', 'dec_allocation', 'none'



    'product_allocation_method': {'allocate_func':'least_distance'}, # 'homo_distance', 'least_distance', 'anup', 'no', 'mechanism_agent', 'baseline'
    'revenue_allocation_method': 'constant', # 'constant' or 'dual'
    'constant_transshipment_price': 0.15, # only used when revenue_allocation_method is 'constant'

    # reward calculation

    'self_interest_rate': 1.0, # self-interest parameter

    'payment_type': 'selling_first', # 先款后货: 'pay_first'，先货后款: 'product_first'，先销后款: 'selling_first'
    # reward的计算方式
    'reward_type': 'profit', # 'profit', 'norm_profit'
    'reward_scale_tf': False, # 是否对reward进行scale

    # observation
    'state_normalize': True,            # 是否把state normalize 到均值为0，但无法保证服从N(0, 1)
    'state_clip_list': [-5, 5],         # 强制控制norm后的state在[-5, 5]范围，防止出现过于破坏分布的数据点，从而阻碍训练

    'actor_obs_type': 'full', # 'full', 'partial', 'self'

    'critic_obs_type': 'full', # 'full', 'partial', 'self'
    'demand_info_for_critic': ['long_quantile', 'long_mean', 'short_all', 'short_mean'], # choices: 'long_quantile': 远期需求的分位数,
    # 'long_mean': 远期需求的均值, 'short_all': 近期需求的所有值, 'short_mean': 近期需求的均值
    'inventory_est_for_critic_tf': True, # True/False 是否将inventory估计值作为critic的输入
    'inventory_est_others_for_critic_tf': True, # True/False, 是否把其他agent的库存多寡作为critic的输入
    'inventory_est_horizon': 4, # inventory估计值的时间跨度, 同时也是区分长短期的时间跨度

    # ordering frequency
    # MOQ: minimum order quantity - 标量或列表，如 [20, 25, 30]
    'MOQ': 20,
    # average_frequency: average order frequency - 标量或列表，如 [1, 2, 1]
    'average_frequency': 1,

    # action space
    'action_type': 'multi_discrete', # 'discrete', 'continuous' or 'multi_discrete'
    'order_step_size': 0.1, # 非continuous的order的action space的步长
    'transship_step_size': 0.05, # 非continuous的transship的action space的步长
    # order_ub: 最高几倍的MOQ(或0.8*(average_frequency-1)*demand_mean) - 标量或列表，如 [3, 4, 3]
    'order_ub': 3,
    # transship_lb: 最高转运出去是几倍的库存 - 标量或列表，如 [-1, -1.5, -1]
    'transship_lb': -1,
    # transship_ub: 最高转运进来是几倍的demand_mean - 标量或列表，如 [1.5, 2, 1.5]
    'transship_ub': 1.5,
    'estimate_demand_method': 'ema',  # 'mean', 'rolling_mean', 'ema'
    'rolling_window': 10, # rolling window size, 用于估计demand_mean
    'alpha': 0.2, # ema的alpha值

    'gamma': 0.95,   # 和algo的参数里设置一致
    'episode_length': 50,   # 和algo的参数里设置一致
    'num_envs': 1,   # 向量化环境数量

    # Dataset configuration
    'dataset_name': 'default',  # 验证集/测试集的数据集名称

    # Advantage-splitting / logging helpers
    'advantage_split_mode': 'lt_today',    # 可选: 'lt_today', 'lt_avg', 'fixed'
    'advantage_fixed_lt': 0,               # 当mode为'fixed'时分配的固定horizon长度
    'advantage_split_max_horizon': 100,    # 最长horizon, 通常设置为episode_length
    'price_agent_feature_mode': None,      # agent的价格特征计算方式（可选项，设置为None）
    'include_price_signal_in_obs': False,  # 是否将价格信号纳入观测空间

    # Mechanism observation configuration
    'actor_obs_type_mechanism': 'full',  # mechanism agent的actor obs类型: 'intentions_only' | 'full' | 'compressed'
    'critic_obs_type_mechanism': 'full',  # mechanism agent的critic obs类型: 'compressed' | 'full'
    'mechanism_reward_mode': 'avg_reward', # 'avg_reward' or 'pooled_cost_saving'
}
