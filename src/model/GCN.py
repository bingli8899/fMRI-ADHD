from torch_geometric.nn import GCN, global_mean_pool, global_add_pool, global_sort_pool, global_max_pool
from torch_geometric.nn import SAGPooling, EdgePooling, ASAPooling, PANPooling, MemPooling 
from torch.nn import Linear, Module, ReLU, LayerNorm, Sequential
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
        # normalization_function = name_to_norm.get(config.predictor_paras.norm_type)
        predictor_function = name_to_predictor.get(config.predictor_paras.predictor_type)


        # # If we want to apply the normalization later before concating metadata 
        # if self.config.predictor_paras.norm_enabled and self.config.predictor_paras.norm_timing == "before":
        #     self.norm = normalization_function(config.model_params.out_channels * 2)

        # # adding normalization layer after concating the metadata 
        # if self.config.predictor_paras.norm_enabled and self.config.predictor_paras.norm_timing == "after":
        #     self.norm = normalization_function(config.model_params.out_channels * 2 + 81)

        activation_function = name_to_activation.get(config.predictor_paras.activation_btw_predictors)
        self.activation = activation_function()

        out_dim = config.model_params.out_channels * 2 
        self.metadata_predictor = None
        if self.config.add_metadata and not self.config.metadata_MLP: 
            predict_input_shape = out_dim + 81
        elif self.config.add_metadata and self.config.metadata_MLP:
            predict_input_shape = out_dim + self.config.metada_MLP_out_channels
            self.metadata_predictor = Sequential(
                    Linear(81, self.config.metada_MLP_out_channels),
                    ReLU())
        else: 
            predict_input_shape = out_dim

        self.projector1 = None
        self.projector2 = None

        if self.config.predictor_paras.num_pred_layers == 2: 
            self.projector1 = predictor_function(predict_input_shape, 
                                                config.predictor_paras.hidden_channels)
            self.projector2 = predictor_function(config.predictor_paras.hidden_channels, 
                                                len(config.model_params.loss_weights)) 
        elif self.config.predictor_paras.num_pred_layers == 1: 
            self.projector1 = predictor_function(predict_input_shape, 
                                                len(config.model_params.loss_weights))

        if self.config.predictor_paras.norm_enabled and self.config.predictor_paras.norm_timing == "before": # if adding normalization layer before predictor enabled 
            out = self.norm(out)


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
        
        if self.config.add_metadata:
            length = data_pos.metadata.shape[0] // 81 
            if not self.metadata_predictor: 
                out_pos = torch.cat((out_pos, data_pos.metadata.view((length, 81))), axis = 1)  
            else:
                metadata = data_pos.metadata.view(-1, 81) 
                metadata_out = self.metadata_predictor(metadata)
                out_pos = torch.cat((out_pos, metadata_out), axis = 1)
            
        out = torch.cat((out_pos, out_neg), axis = 1)

        if self.projector2: 
            out = self.projector1(out)
            out = self.activation(out) 
            out = self.projector2(out) 
        else: 
            out = self.projector1(out)

        # print("model output after two prediction layers", out)

        return out