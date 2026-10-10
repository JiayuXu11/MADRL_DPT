"""Tools for HARL."""

import copy

from harl.envs.env_wrappers import ShareSubprocVecEnv, ShareDummyVecEnv
from harl.envs.multi_lt_transship.transship_vec_env import VecTransshipEnv
def get_shape_from_obs_space(obs_space):
    """Get shape from observation space.
    Args:
        obs_space: (gym.spaces or list) observation space
    Returns:
        obs_shape: (tuple) observation shape
    """
    if obs_space.__class__.__name__ == "Box":
        obs_shape = obs_space.shape
    elif obs_space.__class__.__name__ == "list":
        obs_shape = obs_space
    else:
        raise NotImplementedError
    return obs_shape


def get_shape_from_act_space(act_space):
    """Get shape from action space.
    Args:
        act_space: (gym.spaces) action space
    Returns:
        act_shape: (tuple) action shape
    """
    if act_space.__class__.__name__ == "Discrete":
        act_shape = 1
    elif act_space.__class__.__name__ == "MultiDiscrete":
        act_shape = act_space.shape[0]
    elif act_space.__class__.__name__ == "Box":
        act_shape = act_space.shape[0]
    elif act_space.__class__.__name__ == "MultiBinary":
        act_shape = act_space.shape[0]
    return act_shape


def _init_vec_transship_env(n_threads, env_args):
    """Instantiate the vectorized multi-LT transshipment environment."""
    env_cfg = copy.deepcopy(env_args)
    env_cfg["num_envs"] = int(n_threads)
    return VecTransshipEnv(env_cfg)


def make_train_env(env_name, seed, n_threads, env_args, async_tf=True):
    """Make env for training."""

    if env_name == "multi_lt_transship":
        return _init_vec_transship_env(n_threads, env_args)
    else:
        print("Can not support the " + env_name + "environment.")
        raise NotImplementedError


def make_eval_env(env_name, seed, n_threads, env_args, async_tf=True):
    """Make env for evaluation."""
    if env_name == "multi_lt_transship":
        return _init_vec_transship_env(n_threads, env_args)
    else:
        print("Can not support the " + env_name + "environment.")
        raise NotImplementedError

# 应该用不太到
def make_render_env(env_name, seed, env_args):
    """Make env for rendering."""
    print("Can not support the " + env_name + "environment.")
    raise NotImplementedError


def get_num_agents(env, env_args, envs):
    """Get the number of agents in the environment."""
    if hasattr(envs, "n_agents"):
        return envs.n_agents
    if hasattr(envs, "num_agents"):
        return envs.num_agents
    raise AttributeError("Environment does not expose agent count.")
