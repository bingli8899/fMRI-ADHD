import torch 
from torch_geometric.nn import GCN, global_mean_pool, global_add_pool, global_sort_pool, global_max_pool, BatchNorm
from torch_geometric.nn.pool import SAGPooling, EdgePooling, ASAPooling, PANPooling, MemPooling 
from torch.nn import Linear, Module, ReLU, LayerNorm, Dropout
from torch_geometric.nn import Sequential, TransformerConv
from src.utility.ut_model import name_to_pooling, name_to_predictor, name_to_activation 

class TransformerConv_Model(Module):

    def __init__(self, config): 

        super(TransformerConv_Model, self).__init__()
        self.config = config  

        self.TransformerConv1 = TransformerConv(in_channels = config.model_params.in_channels,
                             out_channels = config.model_params.out_channels,
                             heads = config.model_params.heads, 
                             # concate = config.model_params.concate, 
                             dropout = config.dropout,
                             edge_dim = 1, 
                             # kwargs = config.model_params.message_passing
                             aggr = "max"
                             ).to(config.device) 
        
        self.pooling_function = name_to_pooling.get(config.model_params.pooling_function) 
        predictor_function = name_to_predictor.get(config.predictor_paras.predictor_type)

        output_dim = config.model_params.out_channels * config.model_params.heads 

        if self.config.predictor_paras.norm_enabled:
            self.norm = BatchNorm(output_dim) # This should not be hard-coded 

        predict_input_shape = output_dim + 81 if self.config.add_metadata else output_dim
        
        self.projector1 = predictor_function(predict_input_shape, 128)
        self.projector2 = predictor_function(128, len(config.model_params.loss_weights)) 
        activation_function = name_to_activation.get(config.predictor_paras.activation_btw_predictors)
        self.activation = activation_function()


    def forward(self, data):
        
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        edge_attr = edge_attr.view(-1, 1)  # num of edges * num of edge feature for each edge 

        x = self.TransformerConv1(x, edge_index, edge_attr)  
        x = self.pooling_function(x, batch)

        if self.config.add_metadata: 
            length = data.metadata.shape[0] // 81 
            x = torch.cat((x, data.metadata.view((length, 81))), axis = 1)

        x = self.projector1(x)
        x = self.activation(x) 
        x = self.projector2(x) 

        return x 