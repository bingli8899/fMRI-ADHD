import torch
import torch.nn.functional as F
from torch.nn import Linear, Module, ReLU, LayerNorm 
from torch_geometric.nn import NNConv, global_mean_pool, global_add_pool, global_sort_pool, global_max_pool 
from torch.nn import Sequential as Seq, Linear as Lin

class NNConv_model(Module): 

    def __init__(self, config):

        super().__init__()
        self.config = config

        nn1 = Seq(
            Lin(1, config.model_params.nn1_channels), 
            ReLU(),
            Lin(config.model_params.nn1_channels, config.model_params.in_channels * config.model_params.first_hidden_channels))
        self.conv1 = NNConv(config.model_params.in_channels, 
                            config.model_params.first_hidden_channels,
                            nn1,
                            aggr='mean')

        nn2 = Seq(Lin(1, config.model_params.nn2_channels), 
                ReLU(),
                Lin(config.model_params.nn2_channels, config.model_params.second_hidden_channels * config.model_params.out_channels))
        self.conv2 = NNConv(config.model_params.second_hidden_channels,
                            config.model_params.out_channels,
                            nn2,
                            aggr='mean')

        name_to_pooling = {
            "global_mean_pool": global_mean_pool,
            "global_add_pool": global_add_pool,
            "global_sort_pool": global_sort_pool,
            "global_max_pool": global_max_pool,
        }

        self.pooling_function = name_to_pooling.get(config.model_params.pooling_function)

        name_to_predictor = {
            "Linear": Linear
        }
        name_to_activation = {
            "ReLU": ReLU
        }

        predictor_function = name_to_predictor.get(config.predictor_paras.predictor_type)
        activation_function = name_to_activation.get(config.predictor_paras.activation_btw_predictors)

        predict_input_shape = config.model_params.out_channels + 81 if config.add_metadata else config.model_params.out_channels

        if config.predictor_paras.norm_enabled:
            self.norm = LayerNorm(predict_input_shape)

        self.projector1 = predictor_function(predict_input_shape, 128)
        self.projector2 = predictor_function(128, 4)
        self.activation = activation_function()

    def forward(self, data):

        x = data.x.to(self.config.device)
        edge_index = data.edge_index.to(self.config.device)
        edge_attr = data.edge_attr.to(self.config.device).view(-1, 1)
        batch = data.batch.to(self.config.device)

        x = self.conv1(x, edge_index, edge_attr)
        x = self.activation(x) 
        x = self.conv2(x, edge_index, edge_attr)
        x = self.activation(x) 

        out = self.pooling_function(x, batch)

        if self.config.add_metadata:
            length = data.metadata.shape[0] // 81
            out = torch.cat((out, data.metadata.view((length, 81))), axis=1)

        if self.config.predictor_paras.norm_enabled:
            out = self.norm(out)

        out = self.projector1(out)
        out = self.activation(out)
        out = self.projector2(out)

        return out
