import torch
import torch.nn.functional as F
from torch.nn import Linear, Module, ReLU, LayerNorm, ModuleList, BatchNorm1d
from torch_geometric.nn import DirGNNConv, GCNConv, GATv2Conv, GraphConv
from torch_geometric.nn import global_mean_pool, global_add_pool, global_sort_pool, global_max_pool
from torch_geometric.nn import SAGPooling, EdgePooling, GlobalAttention
from torch.nn import Sequential as Seq, Linear as Lin
from src.utility.ut_model import name_to_pooling, name_to_predictor, name_to_activation, name_to_norm

class DirGNN_GatConv_model(Module): 

    def __init__(self, config):

        super().__init__() 
        self.config = config

        self.dirgnn_layers = ModuleList()
        self.gatv2_layers = ModuleList()

        self.dropout_DirGNN = config.model_params.dropout_DirGNN
        self.dropout_GATv2 = config.model_params.dropout_GATv2

        output_channels_DirGNN = config.model_params.out_channels_DirGNN
        output_channels_GATv2 = config.model_params.output_channels_GATv2

        input_dim = config.model_params.in_channels 
        for i in range(config.model_params.num_layers_DirGNN):
            hidden_dim = output_channels_DirGNN # let's use out_channels = hidden_channels now 
            baseconv = GCNConv(input_dim, 
                               hidden_dim)
            dir_layer = DirGNNConv(baseconv, config.model_params.alpha, root_weight=False)
            self.dirgnn_layers.append(dir_layer)
            input_dim = hidden_dim

        heads = config.model_params.heads
        input_dim = config.model_params.in_channels 
        out_dim = output_channels_GATv2
        for i in range(config.model_params.num_layers_GATv2):
            gatv2_layer = GATv2Conv(in_channels=input_dim,
                                    out_channels=out_dim,
                                    heads=heads,
                                    concat=True, # concate outputs across heads 
                                    dropout=config.model_params.dropout_GATv2, # hard-coded for now 
                                    edge_dim=1,
                                    fill_value=1.0).to(config.device)
            self.gatv2_layers.append(gatv2_layer)
            input_dim = out_dim * heads
        
        # Output dimension for GatV2
        output_dim_GATv2 = output_channels_GATv2 * heads 
        output_dim_DirGNN = output_channels_DirGNN

        self.pooling_function_1 = name_to_pooling.get(config.model_params.pooling_function_DirGNN) 

        if config.model_params.use_attention_pool:
            self.pooling_function_2 = GlobalAttention(
                gate_nn=Seq(
                    Lin(output_dim_GATv2, output_dim_GATv2),
                    ReLU(),
                    Lin(output_dim_GATv2, 1)
                )) 
        else: 
            self.pooling_function_2 = name_to_pooling.get(config.model_params.pooling_function_GATv2)

        predictor_function = name_to_predictor.get(config.predictor_paras.predictor_type)

        activation_function1 = name_to_activation.get(config.model_params.activation_btw_conv)
        self.activation_btw_conv_layers = activation_function1()

        predict_input_shape = output_dim_GATv2 + output_dim_DirGNN + 81 if self.config.add_metadata else output_dim_GATv2 + output_dim_DirGNN 
        
        normalization_btw_predictors = name_to_norm.get(config.predictor_paras.norm_type)
        if self.config.predictor_paras.norm_enabled:
            self.norm = normalization_btw_predictors(predict_input_shape) 

        self.projector1 = predictor_function(predict_input_shape, 
                                             config.predictor_paras.predictor_hidden_channels)
        self.projector2 = predictor_function(config.predictor_paras.predictor_hidden_channels, 
                                             len(config.model_params.loss_weights)) 
        activation_function2 = name_to_activation.get(config.predictor_paras.activation_btw_predictors)
        self.activation_for_linear_projector = activation_function2()


    def forward(self, data):
        
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        x_1 = x.clone() 
        for layer in self.dirgnn_layers:
            x_1 = layer(x_1, edge_index)
            x_1 = self.activation_btw_conv_layers(x_1) 
            x_1 = F.dropout(x_1, p=self.dropout_DirGNN, training=self.training) 
        x_1 = self.pooling_function_1(x_1, batch)
        
        for layer in self.gatv2_layers:
            x_2 = layer(x, edge_index, edge_attr)
        x_2 = self.pooling_function_2(x_2, batch)

        x = torch.cat((x_1, x_2), axis = 1)

        if self.config.add_metadata: 
            length = data.metadata.shape[0] // 81 
            x = torch.cat((x, data.metadata.view((length, 81))), axis = 1)

        x = self.projector1(x)
        x = self.activation_for_linear_projector(x) 
        x = self.projector2(x) 

        return x 
        

    