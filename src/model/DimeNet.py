import torch
import torch.nn.functional as F
from torch_geometric.nn.models import DimeNetPlusPlus, DimeNet
from torch_geometric.nn import global_mean_pool, global_add_pool, global_sort_pool, global_max_pool
from torch.nn import Linear, Module, ReLU, LayerNorm 

class DimeNet_Model(Module):
    def __init__(self, config):
        super(DimeNet_Model, self).__init__()

        self.config = config

        self.model = DimeNetPlusPlus(
            hidden_channels=config.model_params.hidden_channels,
            out_channels=config.model_params.out_channels,
            num_blocks=config.model_params.num_blocks,
            num_spherical = config.model_params.num_spherical,
            num_radial = config.model_params.num_radial, 
            int_emb_size = config.model_params.int_emb_size,
            basis_emb_size = config.model_params.basis_emb_size,
            out_emb_channels = config.model_params.out_emb_channels,
            cutoff=5.0,
            max_num_neighbors=32,
            envelope_exponent=5,
            num_output_layers=3
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

        predict_input_shape = config.model_params.out_channels + 81 if config.add_metadata else config.model_params.out_channels

        if self.config.predictor_paras.norm_enabled:
            self.norm = LayerNorm(predict_input_shape)

        self.projector1 = predictor_function(predict_input_shape, 128)
        self.projector2 = predictor_function(128, 4)

        activation_function = name_to_activation.get(config.predictor_paras.activation_btw_predictors)
        self.activation = activation_function()

    def forward(self, data):
        out = self.model(
            z=data.x.to(self.config.device),
            edge_index=data.edge_index.to(self.config.device),
            edge_attr=data.edge_attr.to(self.config.device),
            batch=data.batch.to(self.config.device)
        )

        out = self.pooling_function(out, data.batch.to(self.config.device))

        if self.config.add_metadata:
            length = data.metadata.shape[0] // 81 # 81 is the number of metadata features
            out = torch.cat((out, data.metadata.view((length, 81))), axis=1)

        if self.config.predictor_paras.norm_enabled:
            out = self.norm(out)

        out = self.projector1(out)
        out = self.activation(out)
        out = self.projector2(out)

        return out

