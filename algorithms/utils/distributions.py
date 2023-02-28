import torch
import torch.nn as nn
from .util import init

"""
Modify standard PyTorch distributions so they to make compatible with this codebase. 
"""

#
# Standardize distribution interfaces
#

# Categorical
class FixedCategorical(torch.distributions.Categorical):
    def sample(self):
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions):
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)

# Beta
class FixedBeta(torch.distributions.beta.Beta):
    def log_probs(self, actions):
        return super().log_prob(self.to_beta(actions))
        # return super().log_prob(actions).sum(-1, keepdim=True)

    def entrop(self):
        return super().entropy().sum(-1)
    def sample(self, round_tf=True, bound=None):
        s_data= super().sample()
        p=self.log_prob(s_data)
        print(torch.isinf(p).any())
        action = self.to_action(s_data)
        k=self.log_probs(torch.round(action))
        print(torch.isinf(k).any())
        return torch.round(action) if round_tf else action
    def mode(self, round_tf=True, bound=None):
        # bound=(torch.tensor([0,-1000],device='cuda'),torch.tensor([1000,1000],device='cuda'))
        # bound=(torch.tensor([0],device='cuda'),torch.tensor([1000],device='cuda'))
        action = self.to_action(self.mean)
        return torch.round(action) if round_tf else action
    
    # 让0-1映射到订货0 -- 40，transship-20 -- 20
    def to_action(self, data):
        agent_num=data.shape[-1]//2
        k=data*40.0+torch.tensor([[0.]*agent_num+[-20.]*agent_num],device='cuda')
        return data*40.0+torch.tensor([[0.]*agent_num+[-20.]*agent_num],device='cuda')
    
    # 让订货0 -- 40，transship-20 -- 20 映射到0-1
    def to_beta(self, data):
        agent_num=data.shape[-1]//2
        k=(data-torch.tensor([[0.]*agent_num+[-20.]*agent_num],device='cuda'))/40.

        return (data-torch.tensor([[0.]*agent_num+[-20.]*agent_num],device='cuda'))/40.

    
# Normal
class FixedNormal(torch.distributions.Normal):
    def log_probs(self, actions):
        return super().log_prob(actions)
        # return super().log_prob(actions).sum(-1, keepdim=True)

    def entrop(self):
        return super().entropy().sum(-1)

    def mode(self, round_tf=True, bound=None):
        # bound=(torch.tensor([0,-1000],device='cuda'),torch.tensor([1000,1000],device='cuda'))
        # bound=(torch.tensor([0],device='cuda'),torch.tensor([1000],device='cuda'))
        mode_data = torch.round(self.mean) if round_tf else self.mean
        return torch.clamp(torch.round(mode_data),bound[0],bound[1]) if bound else mode_data


# Bernoulli
class FixedBernoulli(torch.distributions.Bernoulli):
    def log_probs(self, actions):
        return super.log_prob(actions).view(actions.size(0), -1).sum(-1).unsqueeze(-1)

    def entropy(self):
        return super().entropy().sum(-1)

    def mode(self):
        return torch.gt(self.probs, 0.5).float()


class Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs, use_orthogonal=True, gain=0.01):
        super(Categorical, self).__init__()
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        def init_(m): 
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x, available_actions=None):
        x = self.linear(x)
        if available_actions is not None:
            x[available_actions == 0] = -1e10
        return FixedCategorical(logits=x)


# newsvendor
# class DiagGaussian(nn.Module):
#     def __init__(self, num_inputs, num_outputs, use_orthogonal=True, gain=0.01, args=None):
#         super(DiagGaussian, self).__init__()

#         init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]

#         def init_(m):
#             return init(m, init_method, lambda x: nn.init.constant_(x, 0.), gain)

#         if args is not None:
#             self.std_x_coef = args.std_x_coef
#             self.std_y_coef = args.std_y_coef
#         else:
#             self.std_x_coef = 1.
#             self.std_y_coef = 0.5
#         self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
#         self.active_func=nn.ReLU()
#         # self.fc_mean = nn.Sequential(
#             # init_(nn.Linear(num_inputs, num_outputs)), nn.ReLU())
#         log_std = torch.ones(num_outputs) * self.std_x_coef
#         # self.log_std = torch.nn.Parameter(log_std,requires_grad=True)
#         self.log_std = torch.nn.Parameter(log_std,requires_grad=False)

#     def forward(self, x, available_actions=None):
#         action_mean = self.fc_mean(x)
#         action_mean = self.active_func(action_mean)
#         action_std = torch.sigmoid(self.log_std / self.std_x_coef) * self.std_y_coef
#         return FixedNormal(action_mean, action_std)
# transship
class DiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs, use_orthogonal=True, gain=0.01, args=None):
        super(DiagGaussian, self).__init__()

        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0.), gain)

        if args is not None:
            self.std_x_coef = args.std_x_coef
            self.std_y_coef = torch.tensor(args.std_y_coef,device='cuda')
        else:
            self.std_x_coef = 1.
            self.std_y_coef = 0.5
        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
        self.active_func=nn.ReLU()
        # self.fc_mean = nn.Sequential(
            # init_(nn.Linear(num_inputs, num_outputs)), nn.ReLU())
        log_std = torch.ones(num_outputs) * self.std_x_coef
        # self.log_std = torch.nn.Parameter(log_std,requires_grad=True)
        self.log_std = torch.nn.Parameter(log_std,requires_grad=False)

    def forward(self, x, available_actions=None):
        action_mean_a = self.fc_mean(x)
        action_mean_0 = self.active_func(action_mean_a[:,[0]])*30.0
        action_mean_1 = torch.clamp(action_mean_a[:,[1]],-20.0,20.0)
        action_mean = torch.cat((action_mean_0,action_mean_1),1)
        action_std = torch.sigmoid(self.log_std / self.std_x_coef) * self.std_y_coef
        return FixedNormal(action_mean, action_std)
    
# central controller专属
class DiagGaussianCentral(nn.Module):
    def __init__(self, num_inputs, num_outputs, use_orthogonal=True, gain=0.01, args=None):
        super(DiagGaussianCentral, self).__init__()

        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0.), gain)

        if args is not None:
            self.std_x_coef = args.std_x_coef
            std_y_coef=[s for s in args.std_y_coef for _ in range(args.num_involver) ]
            self.std_y_coef = torch.tensor(std_y_coef,device='cuda')
        else:
            self.std_x_coef = 1.
            self.std_y_coef = 0.5
        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
        self.active_func=nn.ReLU()
        # self.fc_mean = nn.Sequential(
            # init_(nn.Linear(num_inputs, num_outputs)), nn.ReLU())
        log_std = torch.ones(num_outputs) * self.std_x_coef
        # self.log_std = torch.nn.Parameter(log_std,requires_grad=True)
        self.log_std = torch.nn.Parameter(log_std,requires_grad=False)

    def forward(self, x, available_actions=None):
        action_mean_a = self.fc_mean(x)
        action_mean_0 = self.active_func(action_mean_a[:,[0,1,2]])*30.0
        action_mean_1 = torch.clamp(action_mean_a[:,[3,4,5]],-20.0,20.0)
        action_mean = torch.cat((action_mean_0,action_mean_1),1)
        action_std = torch.sigmoid(self.log_std / self.std_x_coef) * self.std_y_coef
        return FixedNormal(action_mean, action_std)
    
# beta分布
class Actor_Beta(nn.Module):
    def __init__(self,  num_inputs, num_outputs, use_orthogonal=True,  gain=0.01, args=None):
        super(Actor_Beta, self).__init__()
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0.), gain)
        
        self.alpha_layer = init_(nn.Linear(num_inputs, num_outputs))
        self.beta_layer = init_(nn.Linear(num_inputs, num_outputs))
        self.active_func=nn.Softplus()

    def forward(self, x, available_actions= None):
        # alpha and beta need to be larger than 1,so we use 'softplus' as the activation function and then plus 1
        alpha = torch.clamp(self.active_func(self.alpha_layer(x)) + 1.0,2.0,5.0)
        beta = torch.clamp(self.active_func(self.beta_layer(x)) + 1.0,2.0,5.0)
        return FixedBeta(alpha, beta)


    
class Bernoulli(nn.Module):
    def __init__(self, num_inputs, num_outputs, use_orthogonal=True, gain=0.01):
        super(Bernoulli, self).__init__()
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        def init_(m): 
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)
        
        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        x = self.linear(x)
        return FixedBernoulli(logits=x)

class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias
