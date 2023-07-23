import torch
from algorithms.actor_critic import Actor, Critic
from utils.util import update_linear_schedule
import numpy as np
import math

DEMAND_MAX = 20

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
        self.args = args
        self.device = device
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay

        self.obs_space = obs_space
        self.share_obs_space = cent_obs_space
        self.act_space = act_space

        self.hist_demand = None

        self.actor = Actor(args, self.obs_space, self.act_space, self.device)

        ###################################### Please Note#########################################
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
        
        self.lead_time = args.lead_time

        self.k1 = -2.4
        self.k2 = -0.8
        self.pre_ema = [0]*30
        self.pre_ema_d_sqr = [0]*30
        self.window = self.lead_time // 2
        # self.window = 200

    def lr_decay(self, episode, episodes):
        pass

    def get_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic,cell_states_actor, cell_states_critic, masks, available_actions=None,
                    deterministic=False):
        actions, temp_rnn_state,temp_cell_state = self.act(obs, rnn_states_actor,cell_states_actor, None)
        return torch.tensor([[0.]*30]).T, actions, None, temp_rnn_state, torch.tensor(rnn_states_critic), temp_cell_state, torch.tensor(cell_states_critic)

    def get_values(self, cent_obs, rnn_states_critic,cell_states_critic, masks):
        pass

    def evaluate_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic,cell_states_actor, cell_states_critic, action, masks,
                         available_actions=None, active_masks=None):
        pass

    @staticmethod
    def get_order_revenue_list(revenue_per, holding_cost, backorder_cost, shipping_cost, demand_pred):
        revenue_list = []
        revenue_list.append(-demand_pred * backorder_cost)
        for k in range(1, 30):
            revenue_k = (k * demand_pred * revenue_per - sum(
                range(k)) * demand_pred * holding_cost - shipping_cost) / k
            revenue_list.append(revenue_k)
        return revenue_list

    @staticmethod
    def calculate_mu_and_d_sqr(alpha, t, ema, ema_d_sqr, pre_ema, pre_ema_d_sqr):
        tmp_ema = ema
        tmp_pre_ema = pre_ema
        tmp_ema_d_sqr = ema_d_sqr
        tmp_pre_ema_d_sqr = pre_ema_d_sqr
        mu = ema
        d_sqr = ema_d_sqr
        for i in range(t - 1):
            tmp1 = tmp_ema
            tmp2 = tmp_ema_d_sqr
            tmp_ema = alpha * tmp_ema + (1 - alpha) * tmp_pre_ema
            tmp_ema_d_sqr = alpha * \
                (tmp_ema - tmp1) ** 2 + (1 - alpha) * (tmp_pre_ema_d_sqr)
            tmp_pre_ema = tmp1
            tmp_pre_ema_d_sqr = tmp2
            mu += tmp_ema
            d_sqr += tmp_ema_d_sqr
        return mu, d_sqr

    def act(self, obs, rnn_states_actor,cell_states_actor, masks, available_actions=None, deterministic=False, safety_stock=1):

        self.window = max(8, self.lead_time)
        # self.window = self.lead_time
        actions = []
        # print(len(obs))
        for i in range(len(obs)):
            obs_curr = obs[i]
            inv = obs_curr[0]
            demand = obs_curr[1]
            orders = obs_curr[2:2+self.lead_time]
            total_inv = inv + sum(orders)

            window = self.window
            alpha = 2 / (window + 1)
            t_plus_l = self.lead_time + 1

            if len(self.hist_demand) == 0:
                ema = demand
                ema_d_sqr = 0
            # if True:
            #     ema = demand
            #     ema_d_sqr = 0
            else:
                ema = alpha * demand + (1 - alpha) * self.pre_ema[i]
                ema_d_sqr = alpha * (demand - ema)**2 + \
                (1 - alpha) * (self.pre_ema_d_sqr[i])

            mu_t_plus_l, d_sqr_t_plus_l = self.calculate_mu_and_d_sqr(
                alpha, t_plus_l, ema, ema_d_sqr, self.pre_ema[i], self.pre_ema_d_sqr[i])
            sig_t_plus_l = math.sqrt(d_sqr_t_plus_l)

            s = mu_t_plus_l + self.k1 * sig_t_plus_l
            n = np.argmax(self.get_order_revenue_list(1., 0.2, 0.5, 5, demand))
            n_t_plus_l = n + self.lead_time

            mu_n_t_plus_l, d_sqr_n_t_plus_l = self.calculate_mu_and_d_sqr(
                alpha, n_t_plus_l, ema, ema_d_sqr, self.pre_ema[i], self.pre_ema_d_sqr[i])
            sig_n_t_plus_l = math.sqrt(d_sqr_n_t_plus_l)
            S = mu_n_t_plus_l + self.k2 * sig_n_t_plus_l

            self.pre_ema[i] = ema
            self.pre_ema_d_sqr[i] = ema_d_sqr

            if total_inv < s:
                order_heu = max(int(round(S - total_inv)), 0)
            else:
                order_heu = 0
            actions.append([order_heu, 0.])
            # print(actions)
        # print(order_heu, order_heu1, order_heu-order_heu1)
        return torch.tensor(actions), torch.tensor(rnn_states_actor), torch.tensor(cell_states_actor)
