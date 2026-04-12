
import torch
from torch import nn
import torch.nn.functional as F

from torchdrug import layers, core,models
from torchdrug.core import Registry as R

from torch_cluster import knn_graph

from evoif import gvp_layer as layer



def rbf(d, d_min=0.0, d_max=20.0, dim=16):
    d_mu = torch.linspace(d_min, d_max, dim, device=d.device)
    d_mu = d_mu.view([1, -1])
    d_sigma = (d_max - d_min) / dim
    d_expand = torch.unsqueeze(d, -1)

    rbf = torch.exp(-((d_expand - d_mu) / d_sigma) ** 2)
    return rbf



@R.register("models.GVPGNN")
class GVPGNN(nn.Module, core.Configurable):
    '''
    Modified based on https://github.com/drorlab/gvp-pytorch/blob/main/gvp/models.py
    GVP-GNN for Model Quality Assessment as described in manuscript.
    
    Takes in protein structure graphs of type `torchdrug.data.Graph` 
    or `torchdrug.data.PackedGraph` and returns a scalar representation for
    each graph and node in the batch in a `torch.Tensor` of shapes [n_nodes, d]
    and [batch_size, d]
    
    :param node_in_dim: node dimensions in input graph, should be
                        (6, 3) if using original features
    :param node_h_dim: node dimensions to use in GVP-GNN layers
    :param node_in_dim: edge dimensions in input graph, should be
                        (32, 1) if using original features
    :param edge_h_dim: edge dimensions to embed to before use
                       in GVP-GNN layers
    :seq_in: if `True`, sequences will also be passed in with
             the forward pass; otherwise, sequence information
             is assumed to be part of input node embeddings
    :param num_layers: number of GVP-GNN layers
    :param drop_rate: rate to use in all dropout layers
    '''
    def __init__(self, node_in_dim, node_h_dim, 
                 edge_in_dim, edge_h_dim, readout="sum",
                 num_layers=3, drop_rate=0.1,
                 activations=(F.relu, None), vector_gate=True):

        super().__init__()
        self.output_dim = node_h_dim[0]
        self.rbf_dim = edge_in_dim[0]

        self.residue_embdding = nn.Linear(node_in_dim[0], node_in_dim[0], bias=False)
        self.W_v = nn.Sequential(
            layer.GVPLayerNorm(node_in_dim),
            layer.GVP(node_in_dim, node_h_dim, activations=(None, None), vector_gate=vector_gate)
        )
        self.W_e = nn.Sequential(
            layer.GVPLayerNorm(edge_in_dim),
            layer.GVP(edge_in_dim, edge_h_dim, activations=(None, None), vector_gate=vector_gate)
        )

        self.layers = nn.ModuleList(
                layer.GVPConvLayer(node_h_dim, edge_h_dim, drop_rate=drop_rate,
                                 activations=activations, vector_gate=vector_gate)
            for _ in range(num_layers))
        
        ns, _ = node_h_dim
        self.W_out = nn.Sequential(
            layer.GVPLayerNorm(node_h_dim),
            layer.GVP(node_h_dim, (ns, 0), activations=activations, vector_gate=vector_gate)
        )

        if readout == "sum":
            self.readout = layers.SumReadout()
        elif readout == "mean":
            self.readout = layers.MeanReadout()
        else:
            raise ValueError("Unknown readout `%s`" % readout)

    def forward(self, graph, input, all_loss=None, metric=None):      
        h_node = self.residue_embdding(input)

        edge_index = graph.edge_list.t()[:2]
        node_in, node_out = edge_index
        pos_in, pos_out = graph.node_position[node_in], graph.node_position[node_out]
        vec_edge = (pos_out - pos_in).unsqueeze(-2)  # [n_edge, 1, 3]
        h_edge = rbf((pos_out - pos_in).norm(dim=-1), dim=self.rbf_dim), vec_edge
        
        h_node = self.W_v(h_node)
        h_edge = self.W_e(h_edge)
        for layer in self.layers:
            h_node = layer(h_node, edge_index, h_edge)
        node_feature = self.W_out(h_node)

        graph_feature = self.readout(graph, node_feature)

        return {
            "graph_feature": graph_feature,
            "node_feature": node_feature
        }
