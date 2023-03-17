import numpy as np
import gym
from gym import spaces
# from envs.newsvendor_continue import Env
# from envs.newsvendor import Env
from envs.transship_new_all import Env
# from envs.serial import Env
#from envs.net_2x3 import Env


class MultiDiscrete(gym.Space):
    """
    - The multi-discrete action space consists of a series of discrete action spaces with different parameters
    - It can be adapted to both a Discrete action space or a continuous (Box) action space
    - It is useful to represent game controllers or keyboards where each key can be represented as a discrete action space
    - It is parametrized by passing an array of arrays containing [min, max] for each discrete action space
       where the discrete action space can take any integers from `min` to `max` (both inclusive)
    Note: A value of 0 always need to represent the NOOP action.
    e.g. Nintendo Game Controller
    - Can be conceptualized as 3 discrete action spaces:
        1) Arrow Keys: Discrete 5  - NOOP[0], UP[1], RIGHT[2], DOWN[3], LEFT[4]  - params: min: 0, max: 4
        2) Button A:   Discrete 2  - NOOP[0], Pressed[1] - params: min: 0, max: 1
        3) Button B:   Discrete 2  - NOOP[0], Pressed[1] - params: min: 0, max: 1
    - Can be initialized as
        MultiDiscrete([ [0,4], [0,1], [0,1] ])
    """

    def __init__(self, array_of_param_array):
        super().__init__()
        self.low = np.array([x[0] for x in array_of_param_array])
        self.high = np.array([x[1] for x in array_of_param_array])
        self.num_discrete_space = self.low.shape[0]
        self.n = np.sum(self.high) + 2

    def sample(self):
        """ Returns a array with one sample from each discrete action space """
        # For each row: round(random .* (max - min) + min, 0)
        random_array = np.random.rand(self.num_discrete_space)
        return [int(x) for x in np.floor(np.multiply((self.high - self.low + 1.), random_array) + self.low)]

    def contains(self, x):
        return len(x) == self.num_discrete_space and (np.array(x) >= self.low).all() and (
                    np.array(x) <= self.high).all()

    @property
    def shape(self):
        return self.num_discrete_space

    def __repr__(self):
        return "MultiDiscrete" + str(self.num_discrete_space)

    def __eq__(self, other):
        return np.array_equal(self.low, other.low) and np.array_equal(self.high, other.high)


class SubprocVecEnv(object):
    def __init__(self, all_args):
        """
        envs: list of gym environments to run in subprocesses
        """

        self.env_list = [Env(all_args) for i in range(all_args.n_rollout_threads)]
        self.num_envs = all_args.n_rollout_threads

        self.num_agent = self.env_list[0].agent_num 
        self.signal_obs_dim = self.env_list[0].obs_dim
        self.signal_obs_critic_dim = self.env_list[0].obs_critic_dim
        self.signal_action_dim = self.env_list[0].action_dim
        self.num_agent = 1 if all_args.central_controller else self.num_agent

        self.u_range = 100.0  # control range for continuous control
        self.movable = True

        # environment parameters
        # if true, action is a number 0...N, otherwise action is a one-hot N-dimensional vector
        self.discrete_action_input = False
        # if true, even the action is continuous, action will be performed discretely
        self.force_discrete_action = False

        # configure spaces
        self.action_space = []
        self.observation_space = []
        self.observation_space_critic = []

        for agent in range(self.num_agent):
            total_action_space = []
            if all_args.action_type == 'multi_discrete':
                for d in self.signal_action_dim:
                    total_action_space.append(spaces.Discrete(d))  #订货可定0-60,transship 可-20 - 20
            # physical action space
            elif all_args.action_type == 'discrete':
                u_action_space = spaces.Discrete(self.signal_action_dim)  # 5个离散的动作
                total_action_space.append(u_action_space)
            else:
                u_action_space = spaces.Box(low=-self.u_range, high=+self.u_range, shape=(self.signal_action_dim,), dtype=np.float32)  # [-1,1]
                total_action_space.append(u_action_space)
                

            # total action space
            if len(total_action_space) > 1:
                # all action spaces are discrete, so simplify to MultiDiscrete action space
                if all([isinstance(act_space, spaces.Discrete) for act_space in total_action_space]):
                    act_space = MultiDiscrete([[0, act_space.n - 1] for act_space in total_action_space])
                else:
                    act_space = spaces.Tuple(total_action_space)
                self.action_space.append(act_space)
            else:
                self.action_space.append(total_action_space[0])

            # observation space

            self.observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(self.signal_obs_dim,),
                                                     dtype=np.float32))  # [-inf,inf]

            self.observation_space_critic.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(self.signal_obs_critic_dim,),
                                                     dtype=np.float32))

    def step(self, actions):
        results = [env.step(action) for env, action in zip(self.env_list, actions)]
        obs, critic_obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(critic_obs), np.stack(rews), np.stack(dones), infos

    def get_property(self):
        inv = [env.get_inventory() for env in self.env_list]
        demand = [env.get_demand() for env in self.env_list]
        orders = [env.get_orders() for env in self.env_list]
        return inv, demand, orders
    
    def get_hist_demand(self):
        demand = [env.get_hist_demand() for env in self.env_list]
        return demand    
    
    def reset(self,train = True, normalize = True, test_tf = False):
        results = [env.reset(train, normalize, test_tf) for env in self.env_list]
        obs, critic_obs = zip(*results)
        return np.stack(obs), np.stack(critic_obs)

    def close(self):
        pass

    def render(self, mode="rgb_array"):
        pass


# single env
class DummyVecEnv(object):
    def __init__(self, all_args):
        """
        envs: list of gym environments to run in subprocesses
        """

        self.env_list = [Env(all_args) for i in range(1)]


        self.num_envs = all_args.n_rollout_threads

        # self.env_list = [Env() for i in range(all_args.n_rollout_threads)]
        # self.num_envs = all_args.n_rollout_threads

        self.num_agent = self.env_list[0].agent_num 
        self.signal_obs_dim = self.env_list[0].obs_dim
        self.signal_obs_critic_dim = self.env_list[0].obs_critic_dim
        self.signal_action_dim = self.env_list[0].action_dim
        self.num_agent = 1 if all_args.central_controller else self.num_agent


        self.u_range = 1.0  # control range for continuous control
        self.movable = True

        # environment parameters

        # if true, action is a number 0...N, otherwise action is a one-hot N-dimensional vector
        self.discrete_action_input = False
        # if true, even the action is continuous, action will be performed discretely
        self.force_discrete_action = False
        # in this env, force_discrete_action == False��because world do not have discrete_action

        # configure spaces
        self.action_space = []
        self.observation_space = []
        self.observation_space_critic = []

        for agent in range(self.num_agent):
            total_action_space = []
            if all_args.action_type == 'multi_discrete':
                for d in self.signal_action_dim:
                    total_action_space.append(spaces.Discrete(d))  #订货可定0-60,transship 可-20 - 20
            # physical action space
            elif all_args.action_type == 'discrete':
                u_action_space = spaces.Discrete(self.signal_action_dim)  # 5个离散的动作
                total_action_space.append(u_action_space)
            else:
                u_action_space = spaces.Box(low=-self.u_range, high=+self.u_range, shape=(self.signal_action_dim,), dtype=np.float32)  # [-1,1]
                total_action_space.append(u_action_space)
                
            # total action space
            if len(total_action_space) > 1:
                # all action spaces are discrete, so simplify to MultiDiscrete action space
                if all([isinstance(act_space, spaces.Discrete) for act_space in total_action_space]):
                    act_space = MultiDiscrete([[0, act_space.n - 1] for act_space in total_action_space])
                else:
                    act_space = spaces.Tuple(total_action_space)
                self.action_space.append(act_space)
            else:
                self.action_space.append(total_action_space[0])


            self.observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(self.signal_obs_dim,), dtype=np.float32))  # [-inf,inf]
            self.observation_space_critic.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(self.signal_obs_critic_dim,),
                                                     dtype=np.float32))


    def step(self, actions):

        results = [env.step(action) for env, action in zip(self.env_list, actions)]
        obs, critic_obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(critic_obs),np.stack(rews), np.stack(dones), infos
    def get_hist_demand(self):
        demand = [env.get_hist_demand() for env in self.env_list]
        return demand    
    def reset(self, test_tf=False, normalize = True):
        results = [env.reset(train = False,test_tf = test_tf, normalize=normalize) for env in self.env_list]
        obs, critic_obs = zip(*results)
        return np.stack(obs), np.stack(critic_obs)
    
    def get_eval_bw_res(self):
        res = self.env_list[0].get_eval_bw_res()
        return res
    
    def get_eval_num(self):
        eval_num = self.env_list[0].get_eval_num()
        return eval_num
        
    def close(self):
        pass

    def render(self, mode="rgb_array"):
        pass
