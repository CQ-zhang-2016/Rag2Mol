import torch
import torch.nn as nn
from torch.nn import Module, Sequential
from torch.nn import functional as F
from torch import Tensor



class RankModel(Module):
    def __init__(self, ):
        super().__init__()


        self.enc_protein = nn.Sequential(
                    nn.Linear(1024, 512),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    )
        self.enc_mol1 = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            )
        self.enc_mol2 = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            )
        self.enc_mol3 = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            )

        self.joint1 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            )
        self.joint2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            )

        self.out_layer = nn.Sequential(
                    nn.Linear(1024, 128),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(128, 1)
                    )
        
    def forward(self, protein, mol_part, mol_ret):
        
        protein, mol_part = protein.repeat(len(mol_ret), 1), mol_part.repeat(len(mol_ret), 1)

        protein, mol_part, mol_ret1, mol_ret2 = self.enc_protein(protein), self.enc_mol1(mol_part), self.enc_mol2(mol_ret), self.enc_mol3(mol_ret)


        out1 = self.joint1(torch.hstack((protein, mol_ret1)))
        out2 = self.joint2(torch.hstack((mol_part, mol_ret2)))
        out = self.out_layer(torch.hstack((out1, out2)))

        
        
        return out.squeeze(-1)