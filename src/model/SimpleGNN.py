from torch_geometric.nn import GCN, global_mean_pool, global_add_pool, global_sort_pool, global_max_pool
from torch.nn import Linear, Module

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
        self.projector = Linear(200, 4)

    def forward(self, data):
        out = self.model(data.x.to(self.config.device),
                         data.edge_index.to(self.config.device),
                         data.edge_attr.to(self.config.device),
                         batch_size = 8)
        out = global_mean_pool(out, data.batch.to(self.config.device))
        out = self.projector(out)
        return out