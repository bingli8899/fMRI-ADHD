import torch 
from torch_geometric.nn import GCN, global_mean_pool, global_add_pool, global_sort_pool, global_max_pool
from torch_geometric.nn.pool import SAGPooling, EdgePooling, ASAPooling, PANPooling, MemPooling 
from torch.nn import Module, ReLU, LayerNorm, Dropout, BatchNorm1d, Linear
from torch_geometric.nn import Sequential, SAGEConv 
from src.utility.ut_model import name_to_pooling, name_to_predictor, name_to_activation 
from torch_geometric.nn import SAGPooling, EdgePooling, GlobalAttention
from torch.nn import Sequential as Seq
import torch.nn.functional as F


class SageGNN_model(Module):

    def __init__(self, config): 

        super(SageGNN_model, self).__init__()
        self.config = config 
        self.dropout = config.dropout 

        out_dim = config.model_params.out_channels 

        # Depending on the performance, think of kwargs later 
        self.SAGEconv1 = SAGEConv(in_channels = config.model_params.in_channels,
                             out_channels = out_dim,
                             aggr = config.model_params.aggr,
                             normalize = config.model_params.normalize, 
                             project = config.model_params.project
                             ).to(config.device) 
        
        # self.SAGEconv2 = SAGEConv(in_channels = config.model_params.hidden_channels,
        #                      out_channels = out_dim,
        #                      aggr = config.model_params.aggr,
        #                      normalize = config.model_params.normalize, 
        #                      project = config.model_params.project 
        #                      ).to(config.device) 

        if config.model_params.use_attention_pool:
            self.pooling_function = GlobalAttention(
                gate_nn=Seq(
                    Linear(out_dim, out_dim),
                    ReLU(),
                    Linear(out_dim, 1)
                )) 
        else: 
            self.pooling_function = name_to_pooling.get(config.model_params.pooling_function)
 
        predictor_function = name_to_predictor.get(config.predictor_paras.predictor_type)

        # Two layer seems to cause a huge overfitting -- Below are codes for two layers: 
        # if self.config.model_params.batch_norm:
        #     self.norm_1layer = BatchNorm1d(config.model_params.hidden_channels) 
        #     self.norm_2layer = BatchNorm1d(out_dim)
        # else: 
        #     self.norm_1layer = None
        #     self.norm_2layer = None

        # if self.config.model_params.residual: 
        #     self.residual_1layer = Linear(config.model_params.in_channels, 
        #                                   config.model_params.hidden_channels * config.model_params.heads)
        #     self.residual_2layer = Linear(config.model_params.in_channels, 
        #                                   out_dim) 
        # else: 
        #     self.residual_1layer = None
        #     self.residual_2layer = None

        # for one layer: 
        if self.config.model_params.batch_norm:
            self.norm_1layer = BatchNorm1d(out_dim) 
        else: 
            self.norm_1layer = None

        if self.config.model_params.residual: 
            self.residual_1layer = Linear(config.model_params.in_channels, 
                                          out_dim)
        else: 
            self.residual_1layer = None

        self.metadata_predictor = None
        if self.config.add_metadata and not self.config.metadata_MLP: 
            predict_input_shape = out_dim + 81
        elif self.config.add_metadata and self.config.metadata_MLP:
            predict_input_shape = out_dim + self.config.metada_MLP_out_channels
            self.metadata_predictor = Seq(
                    Linear(81, self.config.metada_MLP_out_channels),
                    ReLU())
        else: 
            predict_input_shape = out_dim

        # For two prokectors: 
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
                                            
        activation_function = name_to_activation.get(config.predictor_paras.activation_btw_predictors)
        self.activation = activation_function()


    def forward(self, data):
        
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x_original = x 
        x = self.SAGEconv1(x, edge_index)  

        # x = self.SAGEconv2(x, edge_index)
        x = F.dropout(x, self.dropout, training=self.training)
        
        if self.norm_1layer: 
            x = self.norm_1layer(x) 
        if self.residual_1layer: 
            x = x + self.residual_1layer(x_original)
        
        # x = self.SAGEconv2(x, edge_index, edge_attr) 
        # if self.norm_2layer: 
        #     x = self.norm_2layer(x)
        # x = x + self.residual_2layer(x_original)

        x = self.pooling_function(x, batch)

        if self.config.add_metadata and not self.metadata_predictor: 
            length = data.metadata.shape[0] // 81 
            x = torch.cat((x, data.metadata.view((length, 81))), axis = 1)
            
        elif self.config.add_metadata and self.metadata_predictor: 
            metadata = data.metadata.view(-1, 81) 
            metadata_out = self.metadata_predictor(metadata)
            x = torch.cat((x, metadata_out), 
                           axis = 1)

        if self.projector2: 
            x = self.projector1(x)
            x = self.activation(x) 
            x = self.projector2(x) 
        else: 
            x = self.projector1(x)

        return x 
        

    