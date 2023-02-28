import torch
from algorithms.actor_critic import Actor, Critic
from utils.util import update_linear_schedule
import numpy as np

DEMAND_MAX=20
# 只输出订货量
class Heuristic_Policy:
    """
    HAPPO Policy  class. Wraps actor and critic networks to compute actions and value function predictions.

    :param args: (argparse.Namespace) arguments containing relevant model and policy information.
    :param obs_space: (gym.Space) observation space.
    :param cent_obs_space: (gym.Space) value function input space (centralized input for HAPPO, decentralized for IPPO).
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """

    def __init__(self, args, obs_space, cent_obs_space, act_space, device=torch.device("cpu")):
        self.args=args
        self.device = device
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay

        self.obs_space = obs_space
        self.share_obs_space = cent_obs_space
        self.act_space = act_space

        self.hist_demand=None

        self.actor = Actor(args, self.obs_space, self.act_space, self.device)

        ######################################Please Note#########################################
        #####   We create one critic for each agent, but they are trained with same data     #####
        #####   and using same update setting. Therefore they have the same parameter,       #####
        #####   you can regard them as the same critic.                                      #####
        ##########################################################################################
        self.critic = Critic(args, self.share_obs_space, self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=self.lr, eps=self.opti_eps,
                                                weight_decay=self.weight_decay)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=self.critic_lr,
                                                 eps=self.opti_eps,
                                                 weight_decay=self.weight_decay)

    def lr_decay(self, episode, episodes):
        pass

    def get_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, masks, available_actions=None,
                    deterministic=False):
        actions,temp_rnn_state=self.act(obs,rnn_states_actor,None)
        return torch.tensor([[0.]]), actions,None,temp_rnn_state,torch.tensor(rnn_states_critic)

    def get_values(self, cent_obs, rnn_states_critic, masks):
        pass

    def evaluate_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, action, masks,
                         available_actions=None, active_masks=None):
        pass


    def act(self, obs, rnn_states_actor, masks, available_actions=None, deterministic=False,safety_stock=1):
        """
        Compute actions using the given inputs.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.
        """
        def get_order_revenue_list(revenue_per, holding_cost, backorder_cost, shipping_cost, demand_pred):
            revenue_list=[]
            revenue_list.append(-demand_pred*backorder_cost)
            for k in range(1,30):
                revenue_k=(k*demand_pred*revenue_per-sum(range(k))*demand_pred*holding_cost-shipping_cost)/k
                revenue_list.append(revenue_k)
            return revenue_list

        # obs=(obs*DEMAND_MAX)[0]
        inv = obs[0]
        demand = obs[1]
        orders = obs[3:7]

        inv_pred=max(inv+orders[0]-demand,0)
        inv_pred=max(inv_pred+orders[1]-demand,0)
        inv_pred=max(inv_pred+orders[2]-demand,0)
        inv_pred=max(inv_pred+orders[3]-demand,0)
        order_best=np.argmax(get_order_revenue_list(1., 0.2, 0.5, 5, demand))*demand
        order_tf=1.0*min(demand,inv_pred)-0.5*max(demand-inv_pred,0)-0.2*max(inv_pred-demand,0)<1.0*min(demand,order_best)-0.5*max(demand-order_best,0)-0.2*max(order_best-demand,0)
        # order_heu=max(order_best-inv_pred,0) if order_tf else 0
        order_heu=max(order_best-inv_pred+safety_stock,0) if order_tf else 0

        # order_heu=order_heu if order_heu>5 else 0



        return torch.tensor([[order_heu,0.]]), torch.tensor(rnn_states_actor)
    