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
                        default=1, help="Number of parallel envs for evaluating rollouts(deprecated)")
    parser.add_argument("--n_render_rollout_threads", type=int,
                        default=1, help="Number of parallel envs for rendering rollouts")
    parser.add_argument("--num_env_steps", type=int,
                        default=300e4, help='Number of environment steps to train (default: 10e6) (deprecated)')

    parser.add_argument("--num_episodes", type=int,
                        default=2000, help='Number of episodes, which will set steps automatically')

    parser.add_argument("--n_warmup_evaluations", type=int,
                        default=10, help="Number of evaluations for warmup")  
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
    
    parser.add_argument("--model_dir", type=str,
                        default=None, help="by default None. set the path to pretrained model.")
    # eval parameters
    parser.add_argument("--use_eval", type=t_or_f,
                        default=True, help="by default, do not start evaluation. If set`, start evaluation alongside with training.")
    parser.add_argument("--eval_interval", type=int,
                        default=5, help="time duration between contiunous twice evaluation progress.")


    # render parameters
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
                        default=False, help="info-sharing without dimension reduction")

    parser.add_argument("--adjusted_info_sharing", type=t_or_f,
                        default=False, help="info-sharing with dimension reduction")

    parser.add_argument("--central_controller", type=t_or_f,
                        default=False, help="whether use a central network to control all agents")

    parser.add_argument("--yaml_path", type=str,
                        default=None, help="the path of .yaml")

    parser.add_argument("--sample_mean_advantage", type=t_or_f,
                        default=True, help="whether to use sample_mean_advantage")

    parser.add_argument("--num_involver", type=int,
                        default=3, help="num of retailers/involvers")

    parser.add_argument("--critic_learning_pure_returns", type=t_or_f,
                        default=False, help="About critic loss, true for pure returns, and false for GAE version's returns")
    parser.add_argument("--advantage_pure_returns", type=t_or_f,
                        default=False, help="About calculation of advantage, true for simple advantage, False for GAE version's advantage")

    parser.add_argument("--alpha", type=float,
                        default=0.7, help="self-interest rate")

    parser.add_argument("--ratio_transship", type=t_or_f,
                        default=False, help="reallocate the revenue based on the ratio of transshipments")

    parser.add_argument("--discrete", type=t_or_f,
                        default=True, help="discrete(deprecated)")
    parser.add_argument("--multi_discrete", type=t_or_f,
                        default=False, help="multi_discrete(deprecated)")

    parser.add_argument("--beta", type=t_or_f,
                        default=False, help="beta distribution in continuous setting(deprecated)")

    parser.add_argument("--norm_input", type=t_or_f,
                        default=True, help="whether to map the input to the range -1 to 1 before entering the network")


    parser.add_argument("--action_type", type=str,
                        default='discrete', choices=['discrete', 'multi_discrete', 'continue', 'central_multi_discrete', 'central_discrete'], help="the output of actor network")
    parser.add_argument("--obs_transship", type=str,
                        default='no_transship', choices=['no_transship', 'self_transship', 'all_transship'], help="Whether to include the 'transship' information as input to the actor network.")
    parser.add_argument("--actor_obs_step", type=t_or_f,
                        default=False, help="Whether to include the 'step' information as input to the actor network.")
    parser.add_argument("--critic_obs_step", type=t_or_f,
                        default=False, help="Whether to include the 'step' information as input to the actor network.")

    parser.add_argument("--train_episode_length", type=int,
                        default=195, help="deprecated")

    parser.add_argument("--if_transship", type=t_or_f,
                        default=True, help="Whether allow transshipments within the system.")
    parser.add_argument("--transship_revenue_method", type=str,
                        default='constant', choices=['constant', 'ratio', 'market_ratio'], help="revenue allocation rule for transshipment")
    parser.add_argument("--constant_transship_revenue", type=float,
                        default=0.1, help="The profit earned per unit for each transshipment.(constant)")
    parser.add_argument("--ratio_transship_revenue", type=float,
                        default=0.7, help="The proportion of value created by the transshipment that is received by the receiving party.(ratio)")

    parser.add_argument("--lead_time", type=int,
                        default=4, help="lead time")
    parser.add_argument("--reward_type", type=str,
                        default='norm_cost', choices=['cost', 'reward', 'norm_cost'], help="the objective function of the system")
    parser.add_argument("--reward_norm_multiplier", type=float,
                        default=2.4, help="the norm coefficient of the reward function.(norm_cost)")
    parser.add_argument("--eval_dir", type=str,
                        default="./eval_data/merton", help="evaluation dataset")
    parser.add_argument("--test_dir", type=str,
                        default="./test_data/merton", help="test dataset")
    parser.add_argument("--generator_method", type=str, 
                        default="merton",choices=['merton','uniform','poisson','normal','shanshu_arima','random_fragment','random_resample'], help="training data generator method")
    parser.add_argument("--train_dir", type=str, 
                        default="./train_data/SKU006", help="The source of data for the generator.")
    parser.add_argument("--SKU_id", type=str, 
                        default=None, help="index of tested SKU")

    parser.add_argument("--demand_info_for_critic", type=str,
                        default=['quantile', 'LT_all'], choices=['quantile', 'LT_all'], help="incorporate with exogeneous demand information")

    parser.add_argument("--setting_time_end", type=t_or_f,
                        default=False, help="whether the return considered is finite sum of rewards in T")

    parser.add_argument("--homo_distance", type=t_or_f,
                        default=False, help="whether the distances between retailers are consistent")

    parser.add_argument("--mini_pooling", type=t_or_f,
                        default={"flag": False, "threshold": 200, "how": "even"}, help="hierarchical pooling, how: even or ratio")
 
    parser.add_argument("--pay_first", type=t_or_f,
                        default=False, help="If True, payment is made when placing the order; if False, payment is made upon delivery.")

    parser.add_argument("--cat_self", type=t_or_f,
                        default=False, help="If True, concatenate the vectors before and after feature normalization together.")

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
    parser.add_argument("--use_scale", type=t_or_f,
                        default=False, help="whether to use scale on fixed cost for each order(designed for shanshu)")
    parser.add_argument("--FIXED_COST_scale", type=float,
                        default=1, help="scale on fixed cost for each order(designed for shanshu)")

    parser.add_argument("--shipping_cost_per_distance", type=float,
                        default=0.0005, help="shipping_cost_per_distance")

    parser.add_argument("--distance_index", type=int,
                        default=0, choices=[0, 1, 2, 3], help="the distance matrix used in the environment")

    parser.add_argument("--reset_episode", type=int,
                        default=99999, help="After training for a specified number of iterations, reset the weights of all networks.")

    parser.add_argument("--ignore_after", type=t_or_f,
                        default=False, help="The return outside of T is represented by the mean return.")

    parser.add_argument("--demand_for_action_dim", type=list, 
                    default=None,help="Set the action dimension based on the given list. eg. [10,15,20] ")

    parser.add_argument("--demand_max_for_clip", type=list, 
                    default=None,help="Set the upper limit for generating demand based on the given list. eg. [100,150,200] ")
    
    parser.add_argument("--use_factor", type=t_or_f, 
                        default=True,help="Happo/Mappo")
    
    parser.add_argument("--rnn_name", type=str,
                        default="GRU", choices=["GRU","LSTM","RNN"], help="RNN layer")

    return parser
