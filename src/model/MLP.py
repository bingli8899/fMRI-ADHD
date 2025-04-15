import torch
import torch.nn.functional as F
from torch.nn import Linear, Module, ReLU, LayerNorm 
from torch_geometric.nn import NNConv, global_mean_pool, global_add_pool, global_sort_pool, global_max_pool 
from torch.nn import Sequential as Seq, Linear as Lin

import torch
from torch.nn import Module, Sequential as Seq, Linear as Lin, ReLU
from src.utility.ut_general import name_to_pooling, name_to_predictor, name_to_activation, name_to_norm

class MLP_model(Module):

    def __init__(self, config):
        super(MLP_model, self).__init__()
        self.config = config

        in_dim = config.predictor_paras.mlp_input_channels
        hidden1 = config.predictor_paras.mlp_first_hidden_channels
        out_dim = config.predictor_paras.out_channels

        self.nn1 = Seq(
            Lin(in_dim, hidden1),
            ReLU(),
        )
        self.nn2 = Seq(
            Lin(hidden1, out_dim)
        )

    def forward(self, x):
        out = self.nn1(x)
        out = self.nn2(out)
        return out





