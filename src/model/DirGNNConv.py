import torch
import torch.nn.functional as F
from torch.nn import Linear, Module, ReLU, LayerNorm 
from torch_geometric.nn import DirGNNConv, GCNConv, global_mean_pool, global_add_pool, global_sort_pool, global_max_pool 
from torch.nn import Sequential as Seq, Linear as Lin

class DirGNN_model(Module): 

    def __init__(self, config):

        super().__init__()
        self.config = config
        self.dropout = config.model_params.dropout

        self.conv1 = GCNConv(config.model_params.in_channels, 
                            config.model_params.first_hidden_channels)
        self.conv1 = DirGNNConv(self.conv1, config.model_params.alpha, root_weight=False)

        self.conv2 = GCNConv(config.model_params.first_hidden_channels, 
                            config.model_params.second_hidden_channels)
        self.conv2 = DirGNNConv(self.conv2, config.model_params.alpha, root_weight=False)

        self.conv3 = GCNConv(config.model_params.second_hidden_channels, 
                            config.model_params.out_channels)
        self.conv3 = DirGNNConv(self.conv3, config.model_params.alpha, root_weight=False)

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
        self.projector2 = predictor_function(128, len(config.model_params.loss_weights))
        self.activation = activation_function()

    def forward(self, data):

        x = data.x.to(self.config.device)
        edge_index = data.edge_index.to(self.config.device)

        x = self.conv1(x, edge_index)
        x = self.activation(x) 
        x = self.conv2(x, edge_index)

        x = F.dropout(x, p=self.dropout, training=self.training) 

        x = self.activation(x)
        x = self.conv3(x, edge_index)

        x = F.dropout(x, p=self.dropout, training=self.training) 

        out = self.pooling_function(x, data.batch.to(self.config.device))

        # print("model output from DirGNN", out)

        if self.config.add_metadata:
            length = data.metadata.shape[0] // 81
            out = torch.cat((out, data.metadata.view((length, 81))), axis=1)

        if self.config.predictor_paras.norm_enabled:
            out = self.norm(out)

        out = self.projector1(out)
        out = self.activation(out)
        out = self.projector2(out)

        # print("model output after two prediction layers", out)

        return out

   