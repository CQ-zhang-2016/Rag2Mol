import torch
from torch.nn import Module, Sequential
from torch.nn import functional as F

from torch_geometric.utils import softmax
from torch_geometric.nn.norm import GraphNorm
from torch import Tensor

from .common import GaussianSmearing, EdgeExpansion
from .utils import GVLinear, GVPerceptronVN, MessageModule, global_mean_pool
from torch_scatter import scatter_add
from torch_geometric.utils import unbatch


class Support_model_CA(Module):
    def __init__(self, in_sca, in_vec, neighbor):
        super().__init__()
        self.in_sca, self.in_vec, self.n = in_sca, in_vec, neighbor

        self.distance_expansion = GaussianSmearing(stop=10.0, num_gaussians=64)
        self.vector_expansion = EdgeExpansion(64)
        self.message_module = MessageModule(in_sca, in_vec, 64, 64, in_sca, in_vec, 10.0)

 
        self.attn_sca = torch.nn.MultiheadAttention(in_sca, 16, dropout=0.2, batch_first=True)
        self.attn_vec = torch.nn.MultiheadAttention(in_vec, 8, dropout=0.2, batch_first=True)
        
    def forward(self, h=None, h_pos=None, h_idx=None, ret=None, ret_pos=None, ret_idx=None):

        h_sca_lst, h_pos_lst = unbatch(h[0], h_idx), unbatch(h_pos, h_idx) 
        ret_sca_lst, ret_vec_lst, ret_pos_lst = unbatch(ret[0], ret_idx), unbatch(ret[1], ret_idx), unbatch(ret_pos, ret_idx)

        assert len(h_sca_lst) == len(ret_sca_lst)

        for i, (h0, minihpos, ret0, ret1, miniretpos) in enumerate(zip(h_sca_lst, h_pos_lst, ret_sca_lst, ret_vec_lst, ret_pos_lst)):

            mm, nn = len(minihpos), len(miniretpos)
            dist_idx = torch.zeros((mm, self.n)).long()
            cdist_idx = torch.cdist(minihpos, miniretpos, p=2).topk(min(self.n, nn), dim=-1, largest=False)[1]
            dist_idx[:,:min(self.n, nn)] = cdist_idx
            dist_idx = dist_idx.flatten()

            if i==0:
                new_sca = ret0[dist_idx]
                new_vec = ret1[dist_idx]
                new_pos = miniretpos[dist_idx]
            else:
                new_sca = torch.vstack([new_sca, ret0[dist_idx]])
                new_vec = torch.vstack([new_vec, ret1[dist_idx]])
                new_pos = torch.vstack([new_pos, miniretpos[dist_idx]])


        
        h_edge_idx = torch.arange(len(h_pos)).repeat(self.n, 1).T.flatten().to(h_pos).long()
        ret_edge_idx = torch.arange(len(new_pos)).to(new_pos).long()
        
        vec_ij = h_pos[h_edge_idx] - new_pos[ret_edge_idx]
        dist_ij = torch.norm(vec_ij, p=2, dim=-1).view(-1, 1)
        edge_ij = self.distance_expansion(dist_ij), self.vector_expansion(vec_ij)
        h_ret = self.message_module([new_sca, new_vec], edge_ij, ret_edge_idx, dist_ij, annealing=True)
        # Aggregate messages
        h_add = [scatter_add(h_ret[0], index=h_edge_idx, dim=0, dim_size=h_pos.size(0)), # (N_query, F)
                scatter_add(h_ret[1], index=h_edge_idx, dim=0, dim_size=h_pos.size(0))]
                

        att_sca = self.attn_sca(h[0], h_add[0], h_add[0])[0]
        att_vec = self.attn_vec(h[1].transpose(-1,-2).reshape(-1, self.in_vec), h_add[1].transpose(-1,-2).reshape(-1, self.in_vec), h_add[1].transpose(-1,-2).reshape(-1, self.in_vec))[0]
        att_vec = att_vec.reshape(-1, 3, self.in_vec).transpose(-1,-2)


        return [att_sca, att_vec]