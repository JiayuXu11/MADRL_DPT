import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from harl.utils.envs_tools import get_shape_from_obs_space
from harl.models.base.plain_cnn import PlainCNN
from harl.models.base.plain_mlp import PlainMLP

LOG_STD_MAX = 2
LOG_STD_MIN = -20


class DeterministicMechanismPolicy(nn.Module):
    """Deterministic policy network for continuous action space.(Tailored for Mechanism)"""

    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        """Initialize DeterministicPolicy model.
        Args:
            args: (dict) arguments containing relevant model information.
            obs_space: (gym.Space) observation space.
            action_space: (gym.Space) action space.
            device: (torch.device) specifies the device to run on (cpu/gpu).
        """
        super().__init__()
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.num_agents = args["num_agents"]
        hidden_sizes = args["hidden_sizes"]
        activation_func = args["activation_func"]
        final_activation_func = args["final_activation_func"]
        obs_shape = get_shape_from_obs_space(obs_space)

        self.feature_extractor = None
        feature_dim = obs_shape[0] - self.num_agents
        act_dim = action_space.shape[0]
        # 分离隐藏层和输出层，以便添加 log_std_layer
        self.net = PlainMLP(
            [feature_dim] + list(hidden_sizes), activation_func, final_activation_func
        )
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        low = torch.tensor(action_space.low).to(**self.tpdv)
        high = torch.tensor(action_space.high).to(**self.tpdv)
        self.scale = (high - low) / 2
        self.mean = (high + low) / 2
        self.to(device)

    def forward(self, obs, stochastic=True, with_logprob=True):
        # Return output from network scaled to action space limits.
        # action_inputs 只用于判断正负 mask，不参与网络计算
        action_inputs = obs[:, -self.num_agents:]
        net_input = obs[:, :-self.num_agents]

        if self.feature_extractor is not None:
            x = self.feature_extractor(net_input)
        else:
            x = net_input

        net_out = self.net(x)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # 在 outputs (mu) 上加高斯扰动
        pi_distribution = Normal(mu, std)
        if stochastic:
            outputs = pi_distribution.rsample()
        else:
            outputs = mu
        pos_indices = action_inputs > 0
        neg_indices = action_inputs < 0
        pos_greater = action_inputs.sum(dim=1) > 0

        # 初始化 fulfill_ratio
        fulfill_ratio = torch.zeros_like(outputs)

        # 创建一个掩码，标记正负值同时存在的样本
        valid_mask = (pos_indices.sum(dim=1) > 0) & (neg_indices.sum(dim=1) > 0)

        # 对于正的情况
        pos_mask = valid_mask & pos_greater
        pos_outputs = outputs.clone()
        pos_outputs[~pos_indices] = float('-inf')  # 非正的位置标记为-inf，从而在softmax中被忽略
        pos_softmax = torch.softmax(pos_outputs, dim=1)
        fulfill_ratio[pos_mask] = pos_softmax[pos_mask]

        # 对于负的情况
        neg_mask = valid_mask & ~pos_greater
        neg_outputs = outputs.clone()
        neg_outputs[~neg_indices] = float('inf')  # 非负的位置标记为inf，从而在softmax中被忽略
        neg_softmax = torch.softmax(-neg_outputs, dim=1)
        fulfill_ratio[neg_mask] = neg_softmax[neg_mask]

        # 计算 logp_action，只考虑有效位置（不包括被标记为 -inf 的部分）
        if with_logprob:
            logp = pi_distribution.log_prob(outputs)  # shape: [batch, act_dim]

            # 创建有效位置掩码：pos_mask 样本用 pos_indices，neg_mask 样本用 neg_indices
            valid_positions = torch.zeros_like(outputs, dtype=torch.bool)
            valid_positions[pos_mask] = pos_indices[pos_mask]
            valid_positions[neg_mask] = neg_indices[neg_mask]

            # 只对有效位置的 log_prob 求和
            logp_masked = logp * valid_positions.float()
            logp_action = logp_masked.sum(dim=-1, keepdim=True)

            # 如果 valid_positions 只有一个 True，logp_action 置为 0
            valid_count = valid_positions.sum(dim=-1, keepdim=True)
            logp_action = logp_action * (valid_count > 1).float()
        else:
            logp_action = None

        return fulfill_ratio, logp_action
