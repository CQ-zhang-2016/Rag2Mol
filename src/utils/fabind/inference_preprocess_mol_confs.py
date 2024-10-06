import torch
import argparse
import os
from fabind_utils.inference_mol_utils import read_smiles, extract_torchdrug_feature_from_mol, generate_conformation
from multiprocessing import Pool
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from Bio.PDB.PDBExceptions import PDBConstructionWarning
warnings.simplefilter('ignore', PDBConstructionWarning)

parser = argparse.ArgumentParser(description='Preprocess molecules.')
parser.add_argument("--save_mols_dir", type=str, default="//home/zhangpd/mocular_design/P2M_fabind_RAG/utils/retrieval_database/BindingDB/new_mol",
                    help="Specify where to save the processed pt.")
parser.add_argument("--num_threads", type=int, default=32,
                    help="Multiprocessing threads number")
args = parser.parse_args()

db_total = torch.load('/home/zhangpd/mocular_design/P2M_fabind_RAG/utils/retrieval_database/BindingDB/database.pth')


def get_mol_info(db_name):
    try:
        smiles = db_total[db_name][1]
        mol = read_smiles(smiles)
        mol = generate_conformation(mol)
        molecule_info = extract_torchdrug_feature_from_mol(mol, has_LAS_mask=True)
        
        torch.save([mol, molecule_info], os.path.join(args.save_mols_dir, str(db_name)+'.pt'))
    except Exception as e:
        print('Failed to read molecule id ', db_name, ' We are skipping it. The reason is the exception: ', e)
        
for i in db_total:
    get_mol_info(i)
exit()

with Pool(processes=args.num_threads) as p:
    _ = p.map(get_mol_info, [i for i in db_total])
print('over')
