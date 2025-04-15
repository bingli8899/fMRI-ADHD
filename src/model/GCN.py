from torch_geometric.nn import GCN, global_mean_pool, global_add_pool, global_sort_pool, global_max_pool
from torch.nn import Linear, Module, ReLU, LayerNorm 
import torch

class GCN_Model(Module):
    def __init__(self, config):

        super(GCN_Model, self).__init__()

        self.config = config

        self.model = GCN(in_channels = config.model_params.in_channels, 
            hidden_channels = config.model_params.hidden_channels, 
            num_layers = config.num_layers, 
            out_channels = config.model_params.out_channels, 
            norm=config.model_params.norm, 
            act_first=False, # activation after normalization
            dropout = config.dropout
        ).to(config.device)

        name_to_pooling = {
        "global_mean_pool": global_mean_pool,
        "global_add_pool": global_add_pool,
        "global_sort_pool": global_sort_pool,
        "global_max_pool": global_max_pool,
        }
        name_to_predictor = {
            "Linear": Linear
        }
        name_to_activation = {
            "ReLU": ReLU
        }

        self.pooling_function = name_to_pooling.get(config.model_params.pooling_function) 
        predictor_function = name_to_predictor.get(config.predictor_paras.predictor_type)

        predict_input_shape = 281 if self.config.add_metadata else 200

        if self.config.predictor_paras.norm_enabled:
            self.norm = LayerNorm(predict_input_shape)
        
        self.projector1 = predictor_function(predict_input_shape, 128)
        self.projector2 = predictor_function(128, 4) 
        activation_function = name_to_activation.get(config.predictor_paras.activation_btw_predictors)
        self.activation = activation_function() 

    def forward(self, data):
        out = self.model(data.x.to(self.config.device),
                         data.edge_index.to(self.config.device),
                         data.edge_attr.to(self.config.device),
                         batch_size = 8)
        out = self.pooling_function(out, data.batch.to(self.config.device))

        if self.config.add_metadata: # if adding metadata enabled 
            length = data.metadata.shape[0] // 81 
            out = torch.cat((out, data.metadata.view((length, 81))), axis = 1)

        if self.config.predictor_paras.norm_enabled: # if adding normalization layer before predictor enabled 
            out = self.norm(out)

        out = self.projector1(out)
        out = self.activation(out) 
        out = self.projector2(out)

        return out