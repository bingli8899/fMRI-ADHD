import torch 
from torch_geometric.nn import GCN, global_mean_pool, global_add_pool, global_sort_pool, global_max_pool
from torch_geometric.nn.pool import SAGPooling, EdgePooling, ASAPooling, PANPooling, MemPooling 
from torch.nn import Linear, Module, ReLU, LayerNorm, Dropout
from torch_geometric.nn import Sequential, GATv2Conv
from src.utility.ut_model import name_to_pooling, name_to_predictor, name_to_activation 


class GATv2Conv_Model(Module):

    def __init__(self, config): 

        super(GATv2Conv_Model, self).__init__()
        self.config = config  

        # Depending on the performance, think of kwargs later 
        self.GATconv1 = GATv2Conv(in_channels = config.model_params.in_channels,
                             out_channels = config.model_params.out_channels,
                             heads = config.model_params.heads, 
                             # concate = config.model_params.concate, 
                             # negative_slope = config.model_parmas.activation_slope,
                             dropout = config.dropout,
                             edge_dim = 1, 
                             fill_value = 1.0, 
                             # residual = True, 
                             # kwargs = config.model_params.message_passing
                             ).to(config.device) 

        self.pooling_function = name_to_pooling.get(config.model_params.pooling_function) 
        predictor_function = name_to_predictor.get(config.predictor_paras.predictor_type)

        out_dim = config.model_params.out_channels * config.model_params.heads

        if self.config.predictor_paras.norm_enabled:
            self.norm = BatchNorm(out_dim) # This should not be hard-coded 

        predict_input_shape = out_dim + 81 if self.config.add_metadata else out_dim
        
        self.projector1 = predictor_function(predict_input_shape, 128)
        self.projector2 = predictor_function(128, len(config.model_params.loss_weights)) 
        activation_function = name_to_activation.get(config.predictor_paras.activation_btw_predictors)
        self.activation = activation_function()


    def forward(self, data):
        
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = self.GATconv1(x, edge_index, edge_attr)  
        x = self.pooling_function(x, batch)

        if self.config.add_metadata: 
            length = data.metadata.shape[0] // 81 
            x = torch.cat((x, data.metadata.view((length, 81))), axis = 1)

        x = self.projector1(x)
        x = self.activation(x) 
        x = self.projector2(x) 

        return x 
        

    