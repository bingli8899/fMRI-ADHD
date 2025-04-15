import torch 
from torch_geometric.nn import GCN, global_mean_pool, global_add_pool, global_sort_pool, global_max_pool
from troch_geometric.pool import SAGPooling, EdgePooling, ASAPooling, PANPooling, MemPooling 
from torch.nn import GENConv, Linear, Module, ReLU, LayerNorm, Dropout, BatchNorm
from torch.geometric_nn import Sequential, GCNConv 


from src.utility.ut_model import name_to_pooling, name_to_predictor, name_to_activation 


class GENConv_Model(Module):

    def __init__(self, config): 
        super(GENConv_Model, self).__init__()
        self.config = config  



# class GNN_Model(Module): 

#     def __init__(self, config):

#         super(GNN_Model, self).__init__()

#         self.config = config  

#         self.pooling_function = name_to_pooling.get(config.model_params.pooling_function) 
#         predictor_function = name_to_predictor.get(config.predictor_paras.predictor_type)

#         if self.config.predictor_paras.norm_enabled:
#             self.norm = BatchNorm(config.model_params.out_channels * 2) # This should not be hard-coded 

#         if not self.config.model_params.undirectional_graph: 
#             predict_input_shape = config.model_params.out_channels + 81 if self.config.add_metadata else config.model_params.out_channels
#         else: 
#             predict_input_shape = config.model_params.out_channels * 2 + 81 if self.config.add_metadata else config.model_params.out_channels * 2
        
#         self.projector1 = predictor_function(predict_input_shape, 128)
#         self.projector2 = predictor_function(128, 4) 
#         activation_function = name_to_activation.get(config.predictor_paras.activation_btw_predictors)
#         self.activation = activation_function() 

#         final_output_channels = 4 if config.type.lower() == "four" else 2 

#         self.model = Sequential( 
#             (Dropout(p=config.model_params.dropout), 'x -> x')
#             GCNConv(config.model_params.in_channels, 
#                     config.model_params.first_hidden_channles, 
#                     improved = True), # normalization = true by default  
#             ReLU(inplace = True), 
#             GCNConv(config.model_params.first_hidden_channels, 
#                     config.model_params.out_channels, 
#                     improved = True) 
#             ReLU(inplace=True),
#             pooling_function(config.model_params.out_channels)
#             Linear(config.predictors_paras.first_linear_channels,
#                     final_output_channels),
#         )

#     def forward(self, data_pos, data_neg): 

#         out_pos = self.model(data_pos) 
#         out_neg = self.model(data_neg) 

#         out = torch.cat((out_pos, out_neg), axis = 1) 

        

#         return out 


    