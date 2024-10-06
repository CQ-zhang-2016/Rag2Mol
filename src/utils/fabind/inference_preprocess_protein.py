import torch
from tqdm import tqdm
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import argparse
from utils.inference_pdb_utils import extract_protein_structure, extract_esm_feature

import esm

parser = argparse.ArgumentParser(description='Preprocess protein.')
parser.add_argument("--pdb_file_dir", type=str, default="../inference_data/pdb_files",
                    help="Specify the pdb data path.")
parser.add_argument("--save_pt_dir", type=str, default="../inference_data",
                    help="Specify where to save the processed pt.")
args = parser.parse_args()

esm2_dict = {}
protein_dict = {}

# Load ESM-2 model with different sizes
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
# model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
model.to('cuda')
model.eval()  # disables dropout for deterministic results

for pdb_file in tqdm(os.listdir(args.pdb_file_dir)):
    pdb = pdb_file.split(".")[0]

    pdb_filepath = os.path.join(args.pdb_file_dir, pdb_file)
    protein_structure = extract_protein_structure(pdb_filepath, pdb.split('_')[-2])

    protein_structure['name'] = pdb
    esm2_dict[pdb] = extract_esm_feature(protein_structure, model, alphabet)
    protein_dict[pdb] = protein_structure

torch.save([esm2_dict, protein_dict], os.path.join(args.save_pt_dir, 'processed_protein.pt'))
print('over')