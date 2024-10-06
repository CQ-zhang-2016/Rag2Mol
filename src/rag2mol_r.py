import os
import os.path as osp
import json
import argparse
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
import lmdb
import pickle
from rdkit import Chem
from rdkit.Chem import rdFMCS

from rdkit.Chem import AllChem
from rdkit import DataStructs
import datetime
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from Bio.PDB.PDBExceptions import PDBConstructionWarning
warnings.simplefilter('ignore', PDBConstructionWarning)
import torch
from tqdm import tqdm
import numpy as np

from rdkit import Geometry as Geom
from multiprocessing import Pool
import pandas

torch.set_num_threads(8)


# You could change parameters here
srcdir = './results/BGL07_ORYSJ_25_504_0_4qlk_A/SDF'
outdir = 'mol_find_in_db'
use_db = ['BindingDB', 'GEOM', 'ZinC']
search_per_num = 4



os.makedirs(outdir, exist_ok=True)
wait = []
for mol_name in os.listdir(srcdir):
    rdmol = next(iter(Chem.SDMolSupplier(osp.join(srcdir, mol_name), removeHs=False)))
    if not rdmol: continue
    smile = Chem.MolToSmiles(rdmol)
    wait.append([rdmol, mol_name.replace('.sdf', '')])
print(len(wait), 'molecules waiting')


smiles_all = []
for db_name in use_db:
    with open(osp.join('../data/retrieval_database', db_name, 'total.smi')) as f: data=f.readlines()
    smiles_all += [smi.split()[0] for smi in tqdm(data)]

smiles_all = list(set(smiles_all))
morgan_all = [AllChem.GetMorganFingerprint(Chem.MolFromSmiles(smi), 2) for smi in tqdm(smiles_all)]
print(len(smiles_all), 'in database')

atom_set = set(['H', 'C', 'N', 'O', 'F', 'Mg', 'P', 'S', 'Cl', 'Ca', 'Mn', 'Fe', 'Zn', 'Br', 'I'])

def run(mole_set):
    
    ccnt = 0
    
    rdmol_coord_mean = np.array(mole_set[0].GetConformer(0).GetPositions()).mean(0)
    target_morgan = AllChem.GetMorganFingerprint(mole_set[0], 2)
    similarities = DataStructs.BulkTanimotoSimilarity(target_morgan, morgan_all)

    for j in np.argsort(similarities)[::-1][:search_per_num]:
        try:
            wait_smile = smiles_all[j]

            tmp_mol = Chem.MolFromSmiles(wait_smile)
            if not tmp_mol: continue
            atoms = set([atom.GetSymbol() for atom in tmp_mol.GetAtoms()])
            assert len(atoms) == len(atoms & atom_set)

            AllChem.EmbedMolecule(tmp_mol)
            AllChem.MMFFOptimizeMolecule(tmp_mol)
            tmpmol_coord = np.array(tmp_mol.GetConformer(0).GetPositions())
            tmpmol_coord = tmpmol_coord + rdmol_coord_mean - tmpmol_coord.mean(0)

            for i in range(tmp_mol.GetNumAtoms()):
                tmp_mol.GetConformer(0).SetAtomPosition(i, Geom.Point3D(tmpmol_coord[i, 0], tmpmol_coord[i, 1], tmpmol_coord[i, 2]))
            
            Chem.MolToMolFile(tmp_mol, osp.join(outdir, str(mole_set[1])+'_'+str(ccnt)+'.sdf'))     
            ccnt+=1
        except:
            continue

for i in tqdm(wait):run(i)

'''
# Maybe you want to use multi-process
pool = Pool(8)  
pool.map(run, wait)
pool.close()  
pool.join()  
'''

print('complete')