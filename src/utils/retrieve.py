from .retrieval_utils import extract_protein_structure, extract_esm_feature, convert_1, convert, convert_ligand, generate_morgan
import esm
import torch
import os
import os.path as osp

from rdkit import Chem
import torch.nn as nn
from rdkit.Chem import rdFMCS
from .conplex.protein import ProtBertFeaturizer
from .conplex.model import SimpleCoembedding
from .fabind.fabind_inference import inference
from .protein_ligand import parse_sdf_file, PDBProtein
from .reconstruct import reconstruct_from_generated_with_edges
from .rank import RankModel
from tqdm import tqdm
import random
from vina import Vina
import numpy as np
import traceback
from tqdm.contrib import tzip

class Retrieval(object):
    def __init__(self, device, config, task_dir):
        super().__init__()

        db_total = torch.load(osp.join(config.dock.db_path, 'database.pth'))
        self.db_name, self.db_emb, self.db_smiles = [], [], []
        for i in db_total:
            if not osp.exists(osp.join(config.dock.db_path, 'mol', str(i)+'.pt')): continue
            self.db_name.append(i)
            self.db_emb.append(db_total[i][0])
            self.db_smiles.append(db_total[i][1])

        self.db_emb = torch.stack(self.db_emb, 0).to(device)
        self.device = device

        self.pf = ProtBertFeaturizer(config.conplex.protbert_path).to(torch.device(device))
        self.conplex = SimpleCoembedding()
        self.conplex.load_state_dict(torch.load(config.conplex.conplex_path, map_location=device))
        self.conplex = self.conplex.eval().to(torch.device(device))
        
        self.esm_model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.esm_model = self.esm_model.eval().to(device)
        
        self.dockdir = osp.join(task_dir, 'retrieve_sdf')
        os.makedirs(self.dockdir, exist_ok=True)
        self.convertdir = osp.join(task_dir, 'retrieve_pdbqt')
        os.makedirs(self.convertdir, exist_ok=True)
        self.config = config
        self.vina = Vina(sf_name='vina')

        self.use_rank = config.rank.use_rank
        if self.use_rank:
            self.RankModel = RankModel()
            self.RankModel.load_state_dict(torch.load(config.rank.model, map_location=device)['model'])
            self.RankModel = self.RankModel.eval().to(device)

        
    def __call__(self, pdb_path, idx, chian_id, pkt_xyz):
        
        protein_structure = extract_protein_structure(pdb_path, chian_id, idx)
        protein_esm_feature = extract_esm_feature(protein_structure, self.esm_model, self.alphabet, self.device)

        pkt_xyz = torch.Tensor(pkt_xyz)  

        self.protein_conplex_feature = self.conplex.get_protein_feature(self.pf(protein_structure['seq']))

        _, aff_idx = self.conplex(self.protein_conplex_feature, self.db_emb).topk(self.config.dock.dock_nums)
        supp_names, supp_smiles = [], []
        for i in aff_idx:
            supp_names.append(str(self.db_name[i]))
            supp_smiles.append(self.db_smiles[i])

        inference(supp_smiles, supp_names, [protein_esm_feature, protein_structure], self.config.dock.db_path, 
                        self.config.dock.dock_model, self.device, self.dockdir)
        
        try:
            receptor_lines = convert_1(pdb_path, self.convertdir)
            pdbqt_path = osp.join(self.convertdir, pdb_path.split('/')[-1].replace('.pdb', '.pdbqt'))
            with open(pdbqt_path,'w') as f:f.write(receptor_lines)
        except:
            convert(pdb_path, self.convertdir)


        self.retrieve_dict = {}
        skip = 0
        for name in tqdm(supp_names):
            try:
                ligand_path = osp.join(self.dockdir, name+'.sdf')
                
                ligand_dict = parse_sdf_file(ligand_path)
                rdmol = next(iter(Chem.SDMolSupplier(ligand_path, removeHs=False)))
                lig_xyz = torch.Tensor(ligand_dict['pos'])
                if torch.cdist(pkt_xyz, lig_xyz, p=2).min() > 4.5: 
                    skip+=1
                    print('skipping', skip, 'molecules for not contact')
                    continue
                
                    
                ligqt_path = osp.join(self.convertdir, name+'.pdbqt')
                ligand_text = convert_ligand(ligand_path)
                with open(ligqt_path, 'w') as f:f.write(ligand_text)
                
                self.vina.set_receptor(pdbqt_path)
                self.vina.set_ligand_from_file(ligqt_path)

                self.vina.compute_vina_maps(center=np.array(ligand_dict['pos'], dtype=np.float64).mean(0), box_size=[30., 30., 30.])

                energy_minimized = self.vina.optimize()[0]

                if energy_minimized > self.config.dock.vina_th:
                    skip+=1
                    print('skipping', skip, 'molecules for not contact')
                    continue
                if self.use_rank:
                    self.retrieve_dict[name] = [ligand_dict, rdmol, torch.from_numpy(generate_morgan(rdmol))]
                else:
                    self.retrieve_dict[name] = [ligand_dict, rdmol]
            except:
                skip+=1
                print('skipping', skip, 'molecules for ERROR')
        
        print('retrieve', len(self.retrieve_dict),'molecules')

    def find_mcs(self, data):

        try:
            target_mol = reconstruct_from_generated_with_edges(data)
            wait_keys = [i for i in self.retrieve_dict.keys()]

            if self.use_rank:
                protein = self.protein_conplex_feature.unsqueeze(0)
                target_morgan = torch.from_numpy(generate_morgan(target_mol)).unsqueeze(0).float()

                supp_morgan = torch.cat([self.retrieve_dict[key][2].unsqueeze(0) for key in wait_keys], dim=0).float()
                scores = self.RankModel(protein.to(self.device), target_morgan.to(self.device), supp_morgan.to(self.device)).detach().cpu()

                wait_keys = [wait_keys[i] for i in scores.topk(self.config.rank.choose_num*2, 0)[1]]

            mcs_score, wait_data = [], []
            
            for key in wait_keys:
                mcs_score.append(rdFMCS.FindMCS([target_mol, self.retrieve_dict[key][1]], timeout=1).numAtoms)
                wait_data.append(self.retrieve_dict[key][0])

            return wait_data[random.choice(torch.Tensor(mcs_score).topk(self.config.rank.choose_num, 0)[1])]

        except:
            print('choose failed')
            #traceback.print_exc()
            return self.retrieve_dict[random.sample(self.retrieve_dict.keys(), 1)[0]][0]
        


