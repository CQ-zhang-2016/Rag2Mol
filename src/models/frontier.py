import torch
from torch.nn import Module, Sequential
from torch.nn import functional as F

from .utils import GVPerceptronVN, GVLinear


class FrontierLayerVN(Module):
    def __init__(self, in_sca, in_vec, hidden_dim_sca, hidden_dim_vec):
        super().__init__()
        self.net = Sequential(
            GVPerceptronVN(in_sca, in_vec, hidden_dim_sca, hidden_dim_vec),
            GVLinear(hidden_dim_sca, hidden_dim_vec, 1, 1)
        )

    def forward(self, h_att, idx_ligands):
        h_att_ligand = [h_att[0][idx_ligands], h_att[1][idx_ligands]]
        pred = self.net(h_att_ligand)[0]
        
        return pred

