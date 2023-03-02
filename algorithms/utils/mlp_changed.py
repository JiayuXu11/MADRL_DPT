import torch
import torch.nn as nn
from .util import init, get_clones

"""MLP modules."""

class MLPLayer(nn.Module):
    def __init__(self, input_dim, output_dim, layer_N, use_orthogonal, use_ReLU):
        super(MLPLayer, self).__init__()
        self._layer_N = layer_N

        active_func = [nn.Tanh(), nn.ReLU()][use_ReLU]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(['tanh', 'relu'][use_ReLU])

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        self.fc1 = nn.Sequential(
            init_(nn.Linear(input_dim, output_dim)), active_func)
        # self.fc_h = nn.Sequential(init_(
        #     nn.Linear(output_dim, output_dim)), active_func, nn.LayerNorm(output_dim))
        # self.fc2 = get_clones(self.fc_h, self._layer_N)
        self.fc2 = nn.ModuleList([nn.Sequential(init_(
            nn.Linear(output_dim, output_dim)), active_func) for i in range(self._layer_N)])
        self.norm = nn.LayerNorm(output_dim)
    def forward(self, x):
        x = self.fc1(x)
        # for module in self.fc1:
        #     x = module(x)
        #     assert not torch.isnan(x).any()
        for i in range(self._layer_N):
            x = self.fc2[i](x)
        x= self.norm(x)
        return x


class MLPBase(nn.Module):
    def __init__(self, args, obs_shape, output_dim, layer_N=1, cat_self=True, attn_internal=False):
        super(MLPBase, self).__init__()

        self._use_feature_normalization = args.use_feature_normalization
        self._use_orthogonal = args.use_orthogonal
        self._use_ReLU = args.use_ReLU
        self._stacked_frames = args.stacked_frames
        self._layer_N = layer_N
        self.output_dim = output_dim
        self.cat_self = cat_self
        input_dim = obs_shape

        if self.cat_self:
            self.feature_norm = nn.LayerNorm(input_dim)
            input_dim = 2*input_dim

        self.mlp = MLPLayer(input_dim, self.output_dim,
                              self._layer_N, self._use_orthogonal, self._use_ReLU)

    def forward(self, x):
        if self.cat_self:
            norm_x = self.feature_norm(x)
            x= torch.cat((x,norm_x),-1)
        x = self.mlp(x)

        return x