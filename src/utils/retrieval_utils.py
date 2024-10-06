#!/usr/bin/env python3

from __future__ import absolute_import, print_function
from math import isnan, isinf
from itertools import combinations

import rdkit
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.rdchem import BondType
import sys
from rdkit.Chem import rdFMCS
from rdkit.Chem.rdMolAlign import AlignMol
from rdkit.Chem import ChemicalForceFields
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torch
import os
import os.path as osp
try:
    from .data import ProteinLigandData, torchify_dict
except:
    from utils.data import ProteinLigandData, torchify_dict
from .reconstruct import reconstruct_from_generated_with_edges
from utils.conplex.molecule import MorganFeaturizer
from utils.conplex.protein import ProtBertFeaturizer
from utils.conplex.model import SimpleCoembedding
import torch.nn as nn
import datetime
from rdkit.Chem import rdFMCS
from torch_geometric.data import Batch
from Bio.PDB import PDBParser
from Bio.PDB.PDBIO import PDBIO
from Bio.PDB.PDBIO import Select

import numpy as np
# import esm
from tqdm import tqdm
import subprocess



three_to_one = {'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 
                'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 
                'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'}

def get_protein_structure(res_list):
    # protein feature extraction code from https://github.com/drorlab/gvp-pytorch
    # ensure all res contains N, CA, C and O
    res_list = [res for res in res_list if (('N' in res) and ('CA' in res) and ('C' in res) and ('O' in res))]
    # construct the input for ProteinGraphDataset
    # which requires name, seq, and a list of shape N * 4 * 3
    structure = {}
    structure['seq'] = "".join([three_to_one.get(res.resname) for res in res_list])
    coords = []
    for res in res_list:
        res_coords = []
        for atom in [res['N'], res['CA'], res['C'], res['O']]:
            res_coords.append(list(atom.coord))
        coords.append(res_coords)
    structure['coords'] = coords
    return structure

def get_clean_res_list(res_list, use_chain, idx, verbose=False, ensure_ca_exist=False, bfactor_cutoff=None):
    clean_res_list = []
    for res in res_list:
        hetero, resid, insertion = res.full_id[-1]
        if res.full_id[2] != use_chain: continue
        if resid < idx[0] or resid > idx[1]: continue
        if hetero == ' ':
            if res.resname not in three_to_one:
                if verbose:
                    print(res, "has non-standard resname")
                continue
            if (not ensure_ca_exist) or ('CA' in res):
                if bfactor_cutoff is not None:
                    ca_bfactor = float(res['CA'].bfactor)
                    if ca_bfactor < bfactor_cutoff:
                        continue
                clean_res_list.append(res)
        else:
            if verbose:
                print(res, res.full_id, "is hetero")
    return clean_res_list

def extract_protein_structure(path, use_chain, idx):
    parser = PDBParser(QUIET=True)
    s = parser.get_structure("x", path)
    res_list = get_clean_res_list(s.get_residues(), use_chain, idx, verbose=False, ensure_ca_exist=True)
    sturcture = get_protein_structure(res_list)
    return sturcture

def extract_esm_feature(protein, model, alphabet, device):
    
    letter_to_num = {'C': 4, 'D': 3, 'S': 15, 'Q': 5, 'K': 11, 'I': 9,
                    'P': 14, 'T': 16, 'F': 13, 'A': 0, 'G': 7, 'H': 8,
                    'E': 6, 'L': 10, 'R': 1, 'W': 17, 'V': 19, 
                    'N': 2, 'Y': 18, 'M': 12}

    num_to_letter = {v:k for k, v in letter_to_num.items()}

 
    batch_converter = alphabet.get_batch_converter()

    # Prepare data (first 2 sequences from ESMStructuralSplitDataset superfamily / 4)
    data = [
        ("protein1", protein['seq']),
    ]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    # batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
    batch_tokens = batch_tokens.to(device)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33])
    token_representations = results["representations"][33][0][1: -1]
    assert token_representations.shape[0] == len(protein['seq'])
    return token_representations

def _batch(data_list):
    retrieval_batch, retrieval_batch_idx, cnt = [], [], 0
    
    for data in data_list:
        #for i in data:
            #retrieval_batch.append(i)
            #retrieval_batch_idx += [cnt] * len(i.compose_pos)
        retrieval_batch.append(data)
        retrieval_batch_idx += [cnt] * len(data.compose_pos)
        cnt += 1
        
    return Batch.from_data_list(retrieval_batch, follow_batch=['compose_pos'], exclude_keys=['ligand_nbh_list']), retrieval_batch_idx

def convert_1(pdb_path, savedir):
    receptor_mol=Chem.MolFromPDBFile(pdb_path, removeHs=False)
    receptor_lines=MolToPDBQTBlock(receptor_mol, False, False, True)
    return receptor_lines

def convert(pdb_path, savedir):
    
    commands = """
    eval "$(conda shell.bash hook)"
    conda activate adt
    """


    proc = subprocess.Popen(
            '/bin/bash', 
            shell=False, 
            stdin=subprocess.PIPE, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE
        )

    proc.stdin.write(commands.encode('utf-8'))


    pdbqt_path = osp.join(savedir, pdb_path.split('/')[-1].replace('.pdb', '.pdbqt'))

    commands = """
    prepare_receptor4.py -r {pdb_path} -o {pdbqt_path}
    """.format(
            pdb_path = pdb_path,
            pdbqt_path = pdbqt_path
        )
    proc.stdin.write(commands.encode('utf-8'))


    proc.stdin.close()

    while not osp.exists(pdbqt_path): pass


def PDBQTAtomLines(mol, donors, acceptors):
    atom_lines = [line.replace('HETATM', 'ATOM  ')
                  for line in Chem.MolToPDBBlock(mol).split('\n')
                  if line.startswith('HETATM') or line.startswith('ATOM')]

    pdbqt_lines = []
    for idx, atom in enumerate(mol.GetAtoms()):
        pdbqt_line = atom_lines[idx][:56]

        pdbqt_line += '0.00  0.00    '  # append empty vdW and ele
        # Get charge
        charge = 0.
        fields = ['_MMFF94Charge', '_GasteigerCharge', '_TriposPartialCharge']
        for f in fields:
            if atom.HasProp(f):
                charge = atom.GetDoubleProp(f)
                break
        # FIXME: this should not happen, blame RDKit
        if isnan(charge) or isinf(charge):
            charge = 0.
        pdbqt_line += ('%.3f' % charge).rjust(6)

        # Get atom type
        pdbqt_line += ' '
        atomicnum = atom.GetAtomicNum()
        atomhybridization = atom.GetHybridization()
        atombondsnum = atom.GetDegree()
        if atomicnum == 6 and atom.GetIsAromatic():
            pdbqt_line += 'A '
        elif atomicnum == 7 and idx in acceptors:
            pdbqt_line += 'NA'
        elif atomicnum == 8 and idx in acceptors:
            pdbqt_line += 'OA'
        elif atomicnum == 1 and atom.GetNeighbors()[0].GetIdx() in donors:
            pdbqt_line += 'HD'
        elif atomicnum == 1 and atom.GetNeighbors()[0].GetIdx() not in donors:
            pdbqt_line += 'H '
        elif atomicnum == 16 and ( (atomhybridization == Chem.HybridizationType.SP3 and atombondsnum != 4) or atomhybridization == Chem.HybridizationType.SP2 ):
            pdbqt_line += 'SA'
        else:
            if len(atom.GetSymbol()) >1:
                pdbqt_line += atom.GetSymbol()    
            else:
                pdbqt_line += (atom.GetSymbol() + ' ') 			
        pdbqt_lines.append(pdbqt_line)
    return pdbqt_lines

def MolCoreToPDBQTBlock(mol, core_atom_idx_list, flexible=True, addHs=False, computeCharges=False):
    # make a copy of molecule
    mol = Chem.Mol(mol)

    # if flexible molecule contains multiple fragments write them separately
    if flexible and len(Chem.GetMolFrags(mol)) > 1:
        return ''.join(MolToPDBQTBlock(frag, flexible=flexible, addHs=addHs, computeCharges=computeCharges)
                       for frag in Chem.GetMolFrags(mol, asMols=True))


    patt = Chem.MolFromSmarts('[$([O;H1;v2]),'
                              '$([O;H0;v2;!$(O=N-*),'
                              '$([O;-;!$(*-N=O)]),'
                              '$([o;+0])]),'
                              '$([n;+0;!X3;!$([n;H1](cc)cc),'
                              '$([$([N;H0]#[C&v4])]),'
                              '$([N&v3;H0;$(Nc)])]),'
                              '$([F;$(F-[#6]);!$(FC[F,Cl,Br,I])])]')
    acceptors = list(map(lambda x: x[0], mol.GetSubstructMatches(patt, maxMatches=mol.GetNumAtoms())))
    # Donors
    patt = Chem.MolFromSmarts('[$([N&!H0&v3,N&!H0&+1&v4,n&H1&+0,$([$([Nv3](-C)(-C)-C)]),'
                              '$([$(n[n;H1]),'
                              '$(nc[n;H1])])]),'
                              # Guanidine can be tautormeic - e.g. Arginine
                              '$([NX3,NX2]([!O,!S])!@C(!@[NX3,NX2]([!O,!S]))!@[NX3,NX2]([!O,!S])),'
                              '$([O,S;H1;+0])]')
    donors = list(map(lambda x: x[0], mol.GetSubstructMatches(patt, maxMatches=mol.GetNumAtoms())))
    if addHs:
        mol = Chem.AddHs(mol, addCoords=True, onlyOnAtoms=donors, )
    if addHs or computeCharges:
        AllChem.ComputeGasteigerCharges(mol)

    atom_lines = PDBQTAtomLines(mol, donors, acceptors)
    assert len(atom_lines) == mol.GetNumAtoms()

    pdbqt_lines = []

    pdbqt_lines.append('REMARK  Name = ' + (mol.GetProp('_Name') if mol.HasProp('_Name') else ''))
    if flexible:
        # Find rotatable bonds
        #rot_bond = Chem.MolFromSmarts('[!$([NH]!@C(=O))&!D1&!$(*#*)]-&!@[!$([NH]!@C(=O))&!D1&!$(*#*)]') # From Chemaxon
        rot_bond  = Chem.MolFromSmarts('[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]') #single and not ring, really not in ring?
        bond_atoms = list(mol.GetSubstructMatches(rot_bond))
        
        exclude_list = ['[NX3]-[CX3]=[O,N]', 'C(=[O;!R])N', '[!#1]-[C;H3,Cl3,F3,Br3]']
        excluded_atoms = []
        for excluded_smarts in exclude_list:
            excluded_bond = Chem.MolFromSmarts(excluded_smarts)
            excluded_atoms += [(x[0],x[1]) for x in list(mol.GetSubstructMatches(excluded_bond))]

        for excld_atom in excluded_atoms:
            excld_atom_reverse=(excld_atom[1],excld_atom[0])
            if excld_atom in bond_atoms:
                bond_atoms.remove(excld_atom)
            elif excld_atom_reverse in bond_atoms:
                bond_atoms.remove(excld_atom_reverse)

        tertiary_amide_bonds = Chem.MolFromSmarts('[NX3]([!#1])([!#1])-[CX3]=[O,N]')
        tertiary_amide_bond_atoms=[(x[0],x[3]) for x in list(mol.GetSubstructMatches(tertiary_amide_bonds))]
        if len(tertiary_amide_bond_atoms) > 0:
            for tertiary_amide_bond_atom in tertiary_amide_bond_atoms:
                bond_atoms.append(tertiary_amide_bond_atom)

        to_delete_core_atoms = []
        for i in range(len(bond_atoms)):
            reverse_bond_atoms_i=(bond_atoms[i][1],bond_atoms[i][0])
            for j in core_atom_idx_list:
                atom_j = mol.GetAtomWithIdx(j)
                for neighbor_atom in atom_j.GetNeighbors():
                    neighbor_atom_index = neighbor_atom.GetIdx()
                    if neighbor_atom_index in core_atom_idx_list:
                        bond_atoms_in_core = (j, neighbor_atom_index)
                        if bond_atoms_in_core == bond_atoms[i] or bond_atoms_in_core == reverse_bond_atoms_i:
                            to_delete_core_atoms.append(bond_atoms[i])

        to_delete_core_atoms = list(dict.fromkeys(to_delete_core_atoms))
        for i in to_delete_core_atoms:
            bond_atoms.remove(i)


        atom_lines = PDBQTAtomLines(mol, donors, acceptors) # update coordinate

        bond_ids = [mol.GetBondBetweenAtoms(a1, a2).GetIdx()
                    for a1, a2 in bond_atoms]
        
        if bond_ids:
            for i, b_index in enumerate(bond_ids):
                tmp_frags= Chem.FragmentOnBonds(mol, [b_index], addDummies=False)
                tmp_frags_list=list(Chem.GetMolFrags(tmp_frags))
                #tmp_bigger=0
                if len(tmp_frags_list) == 1:
                    del bond_ids[i]
                    del bond_atoms[i]

            mol_rigid_frags = Chem.FragmentOnBonds(mol, bond_ids, addDummies=False)
        else:
            mol_rigid_frags = mol

        num_torsions = len(bond_atoms)
        # Active torsions header
        pdbqt_lines.append('REMARK  %i active torsions:' % num_torsions)
        pdbqt_lines.append('REMARK  status: (\'A\' for Active; \'I\' for Inactive)')
        for i, (a1, a2) in enumerate(bond_atoms):
            pdbqt_lines.append('REMARK%5.0i  A    between atoms: _%i  and  _%i'
                               % (i + 1, a1 + 1, a2 + 1))

        frags = list(Chem.GetMolFrags(mol_rigid_frags))

        #list frag  from which bonds ?
        fg_bonds=[]
        fg_num_rotbonds={}
        for fg in frags:
            tmp_bonds=[]
            for a1,a2 in bond_atoms:
                if a1 in fg or a2 in fg:
                    tmp_bonds.append(mol.GetBondBetweenAtoms(a1, a2).GetIdx())
            if tmp_bonds:
                fg_bonds.append(tmp_bonds)
            else:
                fg_bonds.append(None)
            fg_num_rotbonds[fg] = len(tmp_bonds)

        # frag with long branch ?
        fg_bigbranch={}
        for i, fg_bond in enumerate(fg_bonds):
            tmp_bigger=0
            frag_i_mol=frags[i]
            if fg_bond != None: # for rigid mol
                tmp_frags= Chem.FragmentOnBonds(mol, fg_bond, addDummies=False)
                tmp_frags_list=list(Chem.GetMolFrags(tmp_frags))
                for tmp_frag_j in tmp_frags_list:
                    len_tmp_fg_j=len(tmp_frag_j)
                    if frag_i_mol == tmp_frag_j:
                        pass
                    else:
                        if len_tmp_fg_j > tmp_bigger:
                            tmp_bigger=len_tmp_fg_j
                #print(f'REMARK FRAG: {i} : {len(frags[i])} : {tmp_bigger} ')
            fg_bigbranch[frags[i]] = tmp_bigger

        def weigh_frags(frag):
            return fg_bigbranch[frag], -fg_num_rotbonds[frag],   # bond_weight
        frags = sorted(frags, key=weigh_frags)

        def count_match(a, b):
            n = 0
            for i in a:
                for j in b:
                    if int(i) == int(j):
                        n = n + 1
            return n
        match_num = []
        for i in range(len(frags)):
            x = frags[i]
            match_num.append(count_match(x, core_atom_idx_list))
        pop_num = match_num.index(max(match_num))


        pdbqt_lines.append('ROOT')
        frag = frags.pop(pop_num)

        for idx in frag:
            pdbqt_lines.append(atom_lines[idx])
        pdbqt_lines.append('ENDROOT')

        branch_queue = []
        current_root = frag
        old_roots = [frag]

        visited_frags = []
        visited_bonds = []
        while len(frags) > len(visited_frags):
            end_branch = True
            for frag_num, frag in enumerate(frags):
                for bond_num, (a1, a2) in enumerate(bond_atoms):
                    if (frag_num not in visited_frags and
                        bond_num not in visited_bonds and
                        (a1 in current_root and a2 in frag or
                         a2 in current_root and a1 in frag)):
                        # direction of bonds is important
                        if a1 in current_root:
                            bond_dir = '%i %i' % (a1 + 1, a2 + 1)
                        else:
                            bond_dir = '%i %i' % (a2 + 1, a1 + 1)
                        pdbqt_lines.append('BRANCH %s' % bond_dir)
                        for idx in frag:
                            pdbqt_lines.append(atom_lines[idx])
                        branch_queue.append('ENDBRANCH %s' % bond_dir)

                        # Overwrite current root and stash previous one in queue
                        old_roots.append(current_root)
                        current_root = frag

                        # remove used elements from stack
                        visited_frags.append(frag_num)
                        visited_bonds.append(bond_num)

                        # mark that we dont want to end branch yet
                        end_branch = False
                        break
                    else:
                        continue
                    break  # break the outer loop as well

            if end_branch:
                pdbqt_lines.append(branch_queue.pop())
                if old_roots:
                    current_root = old_roots.pop()
        # close opened branches if any is open
        while len(branch_queue):
            pdbqt_lines.append(branch_queue.pop())
        pdbqt_lines.append('TORSDOF %i' % num_torsions)
    else:
        pdbqt_lines.extend(atom_lines)

    return '\n'.join(pdbqt_lines)

def MolToPDBQTBlock(mol, flexible=True, addHs=False, computeCharges=False):
    core_atom_idx_list = []
    return MolCoreToPDBQTBlock(mol, core_atom_idx_list, flexible, addHs, computeCharges)

def convert_ligand(ligand_path):

    mol=Chem.MolFromMolFile(ligand_path, removeHs=False, sanitize = False)
    mol_problems = Chem.DetectChemistryProblems(mol)
    if len(mol_problems) > 0:
        for problem in mol_problems:
            if "N, 4" in problem.Message():
                at_idx = problem.GetAtomIdx()
                atom = mol.GetAtomWithIdx(at_idx)
                chg = atom.GetFormalCharge()
                print(f'REMARK    N {at_idx} with formal charge {chg}')
                atom.SetFormalCharge(1)
                atom.UpdatePropertyCache()
            else:
                print(problem.Message())
                exit()
    Chem.SanitizeMol(mol)
    core_atom_idx_list = []

    pdbqtlines=MolCoreToPDBQTBlock(mol, core_atom_idx_list, True, True, True)
    return pdbqtlines        
    
def generate_morgan(rdmol):
    
    features_vec = AllChem.GetMorganFingerprintAsBitVect(
        rdmol, 2, nBits=2048
    )
    features = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(features_vec, features)
    return features