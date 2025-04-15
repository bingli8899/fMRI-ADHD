from torch_geometric.nn import GCN, global_mean_pool, global_add_pool, global_sort_pool, global_max_pool
from torch_geometric.nn import SAGPooling, EdgePooling, ASAPooling, PANPooling, MemPooling 
from torch.nn import Linear, Module, ReLU, LayerNorm 
import torch
from src.utility.ut_model import name_to_pooling, name_to_predictor, name_to_activation, name_to_norm 

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

        self.pooling_function = name_to_pooling.get(config.model_params.pooling_function) 
        normalization_function = name_to_norm.get(config.predictor_paras.norm_type)
        predictor_function = name_to_predictor.get(config.predictor_paras.predictor_type)


        # If we want to apply the normalization later before concating metadata 
        if self.config.predictor_paras.norm_enabled and self.config.predictor_paras.norm_timing == "before":
            self.norm = normalization_function(config.model_params.out_channels * 2)

        # adding normalization layer after concating the metadata 
        if self.config.predictor_paras.norm_enabled and self.config.predictor_paras.norm_timing == "after":
            self.norm = normalization_function(config.model_params.out_channels * 2 + 81)

        if not self.config.model_params.undirectional_graph: 
            predict_input_shape = config.model_params.out_channels + 81 if self.config.add_metadata else config.model_params.out_channels
        else: 
            predict_input_shape = config.model_params.out_channels * 2 + 81 if self.config.add_metadata else config.model_params.out_channels * 2
        
        self.projector1 = predictor_function(predict_input_shape, 128)
        self.projector2 = predictor_function(128, len(config.model_params.loss_weights)) 
        activation_function = name_to_activation.get(config.predictor_paras.activation_btw_predictors)
        self.activation = activation_function() 

    def forward(self, data_pos, data_neg):

        out_pos = self.model(data_pos.x.to(self.config.device),
                         data_pos.edge_index.to(self.config.device),
                         data_pos.edge_attr.to(self.config.device),
                         batch_size = self.config.batch_size)
        out_pos = self.pooling_function(out_pos, data_pos.batch.to(self.config.device))

        out_neg = self.model(data_neg.x.to(self.config.device),
                         data_neg.edge_index.to(self.config.device),
                         data_neg.edge_attr.to(self.config.device),
                         batch_size = self.config.batch_size)
        out_neg = self.pooling_function(out_neg, data_neg.batch.to(self.config.device))

        out = torch.cat((out_pos, out_neg), axis = 1)

        # Try to apply normalization layer before concatenation 
        if self.config.predictor_paras.norm_enabled and self.config.predictor_paras.norm_timing == "before": # if adding normalization layer before predictor enabled 
            out = self.norm(out)
            
        if self.config.add_metadata: # if adding metadata enabled 
            length = data_pos.metadata.shape[0] // 81 # Using either data_pos or data_neg should be the sa,e 
            out = torch.cat((out, data_pos.metadata.view((length, 81))), axis = 1)

        # try to apply normalization after concating metadata 
        if self.config.predictor_paras.norm_enabled and self.config.predictor_paras.norm_timing == "after": 
            out = self.norm(out)
        
        out = self.projector1(out)
        out = self.activation(out) 
        out = self.projector2(out)

        # print("model output after two prediction layers", out)

        return out