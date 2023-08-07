import argparse


def t_or_f(arg):
    ua = str(arg).upper()
    if 'TRUE'.startswith(ua):
        return True
    elif 'FALSE'.startswith(ua):
        return False
    else:
        pass  # error condition maybe?


def get_config():
    """
    The configuration parser for common hyperparameters of all environment. 
    Please reach each `scripts/train/<env>_runner.py` file to find private hyperparameters
    only used in <env>.
    """
    parser = argparse.ArgumentParser(
        description='onpolicy_algorithm', formatter_class=argparse.RawDescriptionHelpFormatter)

    # prepare parameters
    parser.add_argument("--algorithm_name", type=str,
                        default='happo', choices=["happo"])
    # parser.add_argument("--algorithm_name", type=str,
    #                     default='heuristic2', choices=["happo"])

    parser.add_argument('--scenario_name', type=str,
                        default='InventoryManagement', help="Which scenario to run on")
    parser.add_argument("--num_landmarks", type=int,
                        default=3)
    parser.add_argument('--num_agents', type=int,
                        default=3, help="number of players")
    parser.add_argument("--use_single_network", type=t_or_f,
                        default=False)
    parser.add_argument("--experiment_name", type=str,
                        default="check", help="an identifier to distinguish different experiment.")
    parser.add_argument("--seed", type=int,
                        default=[0], help="Random seed for numpy/torch")
    parser.add_argument("--seed_specify", action="store_true",
                        default=True, help="Random or specify seed for numpy/torch")
    parser.add_argument("--running_id", type=int,
                        default=1, help="the running index of experiment")
    parser.add_argument("--cuda", type=t_or_f,
                        default=True, help="by default True, will use GPU to train; or else will use CPU;")
    parser.add_argument("--cuda_deterministic", type=t_or_f,
                        default=True, help="by default, make sure random seed effective. if set, bypass such function.")
    parser.add_argument("--n_training_threads", type=int,
                        default=1, help="Number of torch threads for training")
    parser.add_argument("--n_rollout_threads", type=int,
                        default=50, help="Number of parallel envs for training rollouts")
    parser.add_argument("--n_eval_rollout_threads", type=int,
                        default=1, help="Number of parallel envs for evaluating rollouts(deprecated,暂时是和验证集大小一致)")
    parser.add_argument("--n_render_rollout_threads", type=int,
                        default=1, help="Number of parallel envs for rendering rollouts")
    parser.add_argument("--num_env_steps", type=int,
                        default=300e4, help='Number of environment steps to train (default: 10e6) (deprecated)')

    parser.add_argument("--num_episodes", type=int,
                        default=2000, help='Number of episodes, which will set steps automatically')

    parser.add_argument("--n_warmup_evaluations", type=int,
                        default=10, help="Number of evaluations for warmup")  # 在n_warmup_evaluations中不会触发no_improvement导致的中断
    parser.add_argument("--n_no_improvement_thres", type=int,
                        default=30, help="Threshold number of evaluations with no improvement")
    parser.add_argument("--user_name", type=str,
                        default='marl', help="[for wandb usage], to specify user's name for simply collecting training data.")
    # env parameters
    parser.add_argument("--env_name", type=str,
                        default='TransshipNewEnv', help="specify the name of environment")
    parser.add_argument("--use_obs_instead_of_state", type=t_or_f,
                        default=False, help="Whether to use global state or concatenated obs")

    # replay buffer parameters
    parser.add_argument("--episode_length", type=int,
                        default=200, help="Max length for any episode")

    # network parameters
    parser.add_argument("--share_policy", type=t_or_f,
                        default=False, help='Whether agent share the same policy')
    parser.add_argument("--use_centralized_V", type=t_or_f,
                        default=True, help="Whether to use centralized V function")
    parser.add_argument("--stacked_frames", type=int,
                        default=1, help="Dimension of hidden layers for actor/critic networks")
    parser.add_argument("--use_stacked_frames", type=t_or_f,
                        default=False, help="Whether to use stacked_frames")
    parser.add_argument("--hidden_size", type=int,
                        default=32, help="Dimension of hidden layers for actor networks")
    parser.add_argument("--hidden_size_critic", type=int,
                        default=32, help="Dimension of hidden layers for critic networks")
    parser.add_argument("--layer_N", type=int,
                        default=2, help="Number of layers for actor/critic networks")
    parser.add_argument("--use_ReLU", type=t_or_f,
                        default=True, help="Whether to use ReLU")
    parser.add_argument("--use_popart", type=t_or_f,
                        default=False, help="by default True, use running mean and std to normalize rewards.")
    parser.add_argument("--use_valuenorm", type=t_or_f,
                        default=True, help="by default True, use running mean and std to normalize rewards.")
    parser.add_argument("--use_feature_normalization", type=t_or_f,
                        default=False, help="Whether to apply layernorm to the inputs")
    parser.add_argument("--use_orthogonal", type=t_or_f,
                        default=True, help="Whether to use Orthogonal initialization for weights and 0 initialization for biases")
    parser.add_argument("--gain", type=float,
                        default=0.01, help="The gain # of last action layer")

    # recurrent parameters
    parser.add_argument("--use_naive_recurrent_policy", type=t_or_f,
                        default=True, help='Whether to use a naive recurrent policy')
    parser.add_argument("--use_recurrent_policy", type=t_or_f,
                        default=False, help='use a recurrent policy')
    parser.add_argument("--recurrent_N", type=int,
                        default=2, help="The number of recurrent layers.")
    parser.add_argument("--data_chunk_length", type=int,
                        default=10, help="Time length of chunks used to train a recurrent_policy")

    # optimizer parameters
    parser.add_argument("--lr", type=float,
                        default=1e-4, help='learning rate (default: 5e-4)')
    parser.add_argument("--critic_lr", type=float,
                        default=1e-4, help='critic learning rate (default: 5e-4)')
    parser.add_argument("--opti_eps", type=float,
                        default=1e-5, help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument("--weight_decay", type=float, default=0)

    # trpo parameters
    parser.add_argument("--kl_threshold", type=float,
                        default=0.01, help='the threshold of kl-divergence (default: 0.01)')
    parser.add_argument("--ls_step", type=int,
                        default=10, help='number of line search (default: 10)')
    parser.add_argument("--accept_ratio", type=float,
                        default=0.5, help='accept ratio of loss improve (default: 0.5)')

    # ppo parameters
    parser.add_argument("--ppo_epoch", type=int,
                        default=15, help='number of ppo epochs (default: 15)')
    parser.add_argument("--use_clipped_value_loss", type=t_or_f,
                        default=True, help="by default, clip loss value. If set, do not clip loss value.")
    parser.add_argument("--clip_param", type=float,
                        default=0.2, help='ppo clip parameter (default: 0.2)')
    parser.add_argument("--num_mini_batch", type=int,
                        default=1, help='number of batches for ppo (default: 1)')
    parser.add_argument("--entropy_coef", type=float,
                        default=0.01, help='entropy term coefficient (default: 0.01)')
    parser.add_argument("--value_loss_coef", type=float,
                        default=0.5, help='value loss coefficient (default: 0.5)')
    parser.add_argument("--use_max_grad_norm", type=t_or_f,
                        default=False, help="by default, use max norm of gradients. If set, do not use.")
    parser.add_argument("--max_grad_norm", type=float,
                        default=0.5, help='max norm of gradients (default: 0.5)')
    parser.add_argument("--use_gae", type=t_or_f,
                        default=True, help='use generalized advantage estimation')
    parser.add_argument("--gamma", type=float, default=0.99,
                        help='discount factor for rewards (default: 0.99)')
    parser.add_argument("--gae_lambda", type=float, default=0.95,
                        help='gae lambda parameter (default: 0.95)')
    parser.add_argument("--use_proper_time_limits", type=t_or_f,
                        default=True, help='compute returns taking into account time limits')
    parser.add_argument("--use_huber_loss", type=t_or_f,
                        default=True, help="by default, use huber loss. If set, do not use huber loss.")
    parser.add_argument("--use_value_active_masks", type=t_or_f,
                        default=False, help="by default True, whether to mask useless data in value loss.")
    parser.add_argument("--use_policy_active_masks", type=t_or_f,
                        default=False, help="by default True, whether to mask useless data in policy loss.")
    parser.add_argument("--huber_delta", type=float,
                        default=10.0, help=" coefficience of huber loss.")

    # run parameters
    parser.add_argument("--use_linear_lr_decay", type=t_or_f,
                        default=False, help='use a linear schedule on the learning rate')
    parser.add_argument("--use_step_lr_decay", type=t_or_f, default=False,
                        help='use a step decay schedule on the learning rate')
    parser.add_argument("--lr_decay_stepsize", type=int, default=200,
                        help='learning rate decays after constant stepsize (default: 200)')
    parser.add_argument("--lr_decay_gamma", type=float, default=0.9,
                        help='lr decay rate under step scheduler (default: 0.9)')
    parser.add_argument("--save_interval", type=int,
                        default=1, help="time duration between contiunous twice models saving.")
    parser.add_argument("--log_interval", type=int,
                        default=20, help="time duration between contiunous twice log printing.")
    # parser.add_argument("--model_dir", type=str,
    #                     default=r"C:\Users\Jerry\Desktop\thesis\code\Multi-Agent-Deep-Reinforcement-Learning-on-Multi-Echelon-Inventory-Management-main\results\TransshipNewEnv\Inventory Management\happo\check\run_seed_1\models", help="by default None. set the path to pretrained model.")
    parser.add_argument("--model_dir", type=str,
                        default=None, help="by default None. set the path to pretrained model.")
    # eval parameters
    parser.add_argument("--use_eval", type=t_or_f,
                        default=True, help="by default, do not start evaluation. If set`, start evaluation alongside with training.")
    parser.add_argument("--eval_interval", type=int,
                        default=5, help="time duration between contiunous twice evaluation progress.")
    # parser.add_argument("--eval_episodes", type=int,
    #                    default=32, help="number of episodes of a single evaluation.")

    # render parameters 这一部分好像没啥用，因为render是gym里显示图像的，这里面不是用gym的环境
    parser.add_argument("--save_gifs", type=t_or_f,
                        default=False, help="by default, do not save render video. If set, save video.")
    parser.add_argument("--use_render", type=t_or_f,
                        default=False, help="by default, do not render the env during training. If set, start render. Note: something, the environment has internal render process which is not controlled by this hyperparam.")
    parser.add_argument("--render_episodes", type=int,
                        default=5, help="the number of episodes to render a given env")
    parser.add_argument("--ifi", type=float,
                        default=0.1, help="the play interval of each rendered image in saved video.")

    parser.add_argument("--training_bf", type=t_or_f,
                        default=True, help="whether to train the model ")

    parser.add_argument("--std_x_coef", type=float,
                        default=1.0, help="continue random setting")
    parser.add_argument("--std_y_coef", type=float,
                        default=5.0, help="continue random setting")

    parser.add_argument("--clip_bound", type=float,
                        default=None, help="continue clip setting")

    parser.add_argument("--instant_info_sharing", type=t_or_f,
                        default=False, help="是否采用即时信息共享机制")

    parser.add_argument("--delayed_info_sharing", type=t_or_f,
                        default=False, help="是否采用滞后信息共享机制")

    parser.add_argument("--central_controller", type=t_or_f,
                        default=False, help="是否全部采用中央控制")

   #  parser.add_argument("--yaml_path", type=str,
   #                      default='setting_yaml/discrete/0425_error.yaml', help="yaml的路径")
    parser.add_argument("--yaml_path", type=str,
                        default=None, help="yaml的路径")

    parser.add_argument("--sample_mean_advantage", type=t_or_f,
                        default=True, help="是否对advantage采用sample_mean_advantage")

    parser.add_argument("--num_involver", type=int,
                        default=3, help="表示有多少个零售商")

    parser.add_argument("--critic_learning_pure_returns", type=t_or_f,
                        default=False, help="若为True,则critic network学习没有被处理过的returns,而非经gae或其他方式处理过的return")
    parser.add_argument("--advantage_pure_returns", type=t_or_f,
                        default=False, help="若为True,则advantage使用没有被处理过的returns,而非经gae或其他方式处理过的return")

    parser.add_argument("--alpha", type=float,
                        default=0.7, help="自私程度")

    parser.add_argument("--ratio_transship", type=t_or_f,
                        default=False, help="transship是否采用ratio的分配机制")

    parser.add_argument("--discrete", type=t_or_f,
                        default=True, help="是否discrete(deprecated)")
    parser.add_argument("--multi_discrete", type=t_or_f,
                        default=False, help="是否multi_discrete(deprecated)")

    parser.add_argument("--beta", type=t_or_f,
                        default=False, help="连续情况是否采用beta分布(deprecated)")

    parser.add_argument("--norm_input", type=t_or_f,
                        default=True, help="是否将输入先映射到-1,1")

    parser.add_argument("--entropy_decrease", type=t_or_f,
                        default=True, help="是否递进减少entropy需求")
    parser.add_argument("--entropy_decrease_time", type=int,
                        default=5, help="递进减少entropy多少次(deprecated)")
    parser.add_argument("--entropy_decrease_list", type=float,
                        default=[0.5, 0.1, 0.01, 0], help="递进选择的entropy_coef")

    parser.add_argument("--model_new", type=t_or_f,
                        default=False, help="是否用新的model")

    parser.add_argument("--action_type", type=str,
                        default='discrete', choices=['discrete', 'multi_discrete', 'continue', 'central_multi_discrete', 'central_discrete'], help="actor网络输出格式")
    parser.add_argument("--obs_transship", type=str,
                        default='all_transship', choices=['no_transship', 'self_transship', 'all_transship'], help="transship信息是否作为actor网络的输入")
    parser.add_argument("--actor_obs_step", type=t_or_f,
                        default=False, help="step信息是否作为actor网络的输入")

    parser.add_argument("--train_episode_length", type=int,
                        default=195, help="用于训练的episode长度(针对actor_obs_step为False而设计),不用管，会自动调节的")

    parser.add_argument("--if_transship", type=bool,
                        default=True, help="Whether allow transshipments within the system.")
    parser.add_argument("--transship_revenue_method", type=str,
                        default='constant', choices=['constant', 'ratio', 'market_ratio'], help="transship机制创造收益的分配模式")
    parser.add_argument("--constant_transship_revenue", type=float,
                        default=0.1, help="每transship一单位,可收获的收益")
    parser.add_argument("--ratio_transship_revenue", type=float,
                        default=0.7, help="transship接收方获得transship创造价值的比例")

    parser.add_argument("--lead_time", type=int,
                        default=4, help="订货到达时间")
    parser.add_argument("--reward_type", type=str,
                        default='norm_cost', choices=['cost', 'reward', 'norm_cost'], help="reward 是什么")
    parser.add_argument("--reward_norm_multiplier", type=float,
                        default=2.4, help="使reward均值为0而添加在当期demand上的系数")
   #  parser.add_argument("--demand_mean_val", type=float,
   #                      default=9.478111111111112, help="验证集需求的平均数")

    # parser.add_argument("--generator_method", type=str,
    #                     default="merton", choices=['merton', 'uniform', 'poisson', 'normal', 'shanshu'], help="数据生成的方法")
    parser.add_argument("--eval_dir", type=str,
                        default="./eval_data/merton", help="验证集目录(./xx/xx的格式)")
    parser.add_argument("--test_dir", type=str,
                        default="./test_data/merton", help="测试集目录(./xx/xx的格式)")
    parser.add_argument("--generator_method", type=str, 
                        default="shanshu_arima",choices=['merton','uniform','poisson','normal','shanshu','shanshu_arima','shanshu_sampling','random_fragment','align_random_fragment','random_resample'], help="数据生成的方法")
    # parser.add_argument("--eval_dir", type=str, 
    #                     default="./eval_data/SKU029", help="验证集目录(./xx/xx的格式)")
    # parser.add_argument("--test_dir", type=str, 
    #                     default="./test_data/SKU029", help="测试集目录(./xx/xx的格式)")
    parser.add_argument("--train_dir", type=str, 
                        default="./train_data/SKU029", help="训练集目录(./xx/xx的格式)")
    parser.add_argument("--SKU_id", type=str, 
                        default=None, help="index of tested SKU")

    parser.add_argument("--demand_info_for_critic", type=str,
                        default=['quantile', 'LT_all'], choices=['quantile', 'LT_all'], help="给critic network披露的未来需求信息")

    parser.add_argument("--setting_time_end", type=t_or_f,
                        default=True, help="当时间超过episode_length时,之后的收益是否不考虑")

    parser.add_argument("--homo_distance", type=t_or_f,
                        default=False, help="是否认为零售商间距离一致")

    parser.add_argument("--mini_pooling", type=t_or_f,
                        default={"flag": False, "threshold": 200, "how": "even"}, help="设置mini_pooling机制, how: even or ratio")
 
    parser.add_argument("--pay_first", type=t_or_f,
                        default=False, help="若为True, 则订货时付钱, False则到货到了再付钱")

    parser.add_argument("--cat_self", type=t_or_f,
                        default=False, help="若为True, 则把feature_normalization前后的向量concat在一起")
    parser.add_argument("--cat_self_critic", type=t_or_f,
                        default=False, help="若为True, 则把feature_normalization前后的向量concat在一起")

    parser.add_argument("--H", type=float,
                        default=0.2, help="holding cost per unit")
    parser.add_argument("--R", type=float,
                        default=3.0, help="selling price per unit")
    parser.add_argument("--P", type=float,
                        default=3.5, help="penalty cost per unit")
    parser.add_argument("--C", type=float,
                        default=2, help="ordering cost per unit")
    parser.add_argument("--FIXED_COST", type=float,
                        default=5, help="fixed cost for each order")

    parser.add_argument("--shipping_cost_per_distance", type=float,
                        default=0.0005, help="shipping_cost_per_distance")

    parser.add_argument("--distance_index", type=int,
                        default=0, choices=[0, 1, 2, 3], help="用哪一个distance矩阵,0表示homo_distance")

    parser.add_argument("--reset_episode", type=int,
                        default=99999, help="训练指定次数后，重置所有网络的weight")

    parser.add_argument("--ignore_after", type=t_or_f,
                        default=False, help="T之外的收益用return均值表示")

    parser.add_argument("--demand_for_action_dim", type=list, 
                    default=None,help="跟据该list设定action dim。eg. [10,15,20] ")
    # 只在shanshu_arima里用
    parser.add_argument("--demand_max_for_clip", type=list, 
                    default=None,help="跟据该list设定action dim。eg. [10,15,20] ")
    
    parser.add_argument("--use_factor", type=t_or_f, 
                        default=True,help="用不用sequential factor")
    
    parser.add_argument("--critic_obs_step", type=t_or_f,
                        default=False, help="step信息是否作为critic网络的输入")
    
    parser.add_argument("--rnn_name", type=str,
                        default="GRU", choices=["GRU","LSTM","RNN"], help="rnn用的哪个layer")

    return parser
