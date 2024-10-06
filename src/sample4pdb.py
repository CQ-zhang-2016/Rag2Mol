import os
import argparse
import warnings
from easydict import EasyDict
from Bio import BiopythonWarning
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Selection import unfold_entities
from rdkit import Chem

from utils.protein_ligand import PDBProtein
from copy import deepcopy
import os
import torch
import numpy as np
from torch_geometric.data import Batch
from tqdm.auto import tqdm

from models.Rag2Mol import Rag2Mol
from utils.transforms import *
from utils.sample import *
from utils.misc import *
from utils.reconstruct import *
from utils.retrieve import Retrieval
from utils.data import torchify_dict
import shutil
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


def read_sdf(file):
    supp = Chem.SDMolSupplier(file)
    return [i for i in supp]


def pdb_to_pocket_data(pdb_path, pdb_chain, center, bbox_size):
    center = torch.from_numpy(center).float()
    warnings.simplefilter('ignore', BiopythonWarning)
    ptable = Chem.GetPeriodicTable()
    parser = PDBParser()
    model = parser.get_structure(None, pdb_path)[0][pdb_chain]


    protein_dict = EasyDict({
        'element': [],
        'pos': [],
        'is_backbone': [],
        'atom_to_aa_type': [],
    })
    min_idx, max_idx = 10000, 0
    for atom in unfold_entities(model, 'A'):

        res = atom.get_parent()
        
        resname = res.get_resname()
        
        if resname not in PDBProtein.AA_NAME_NUMBER: continue   # Ignore water, heteros, and non-standard residues.

        element_symb = atom.element.capitalize()
        if element_symb == 'H': continue
        x, y, z = atom.get_coord()
        pos = torch.FloatTensor([x, y, z])

        if torch.cdist(pos.unsqueeze(0), center, p=2).min() > bbox_size: 
            continue

        res_idx = res.get_id()[1]
        min_idx = min(min_idx, res_idx)
        max_idx = max(max_idx, res_idx)
        
        protein_dict['element'].append(ptable.GetAtomicNumber(element_symb))
        protein_dict['pos'].append(pos)
        protein_dict['is_backbone'].append(atom.get_name() in ['N', 'CA', 'C', 'O'])
        protein_dict['atom_to_aa_type'].append(PDBProtein.AA_NAME_NUMBER[resname])
        
    if len(protein_dict['element']) == 0:
        raise ValueError('No atoms found in the bounding box (center=%r, size=%f).' % (center, bbox_size))

    protein_dict['element'] = torch.LongTensor(protein_dict['element'])
    protein_dict['pos'] = torch.stack(protein_dict['pos'], dim=0)
    protein_dict['is_backbone'] = torch.BoolTensor(protein_dict['is_backbone'])
    protein_dict['atom_to_aa_type'] = torch.LongTensor(protein_dict['atom_to_aa_type'])

    data = ProteinLigandData.from_protein_ligand_dicts(
        protein_dict = protein_dict,
        ligand_dict = {
            'element': torch.empty([0,], dtype=torch.long),
            'pos': torch.empty([0, 3], dtype=torch.float),
            'atom_feature': torch.empty([0, 8], dtype=torch.float),
            'bond_index': torch.empty([2, 0], dtype=torch.long),
            'bond_type': torch.empty([0,], dtype=torch.long),
        }
    )
    return data, [min_idx, max_idx], protein_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--bbox_size', type=float, default=10.0, 
                        help='Pocket bounding box size')
    parser.add_argument('--config', type=str, default='./configs/sample.yml')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--outdir', type=str, default='./results')
    parser.add_argument(
            '--check_point',type=str,default='./params/rag2mol_ckpt/val_165.pt',
            help='load the parameter')
    parser.add_argument(
            '--pdb_file', action='store',required=False,type=str,default='../data/test_set/PHP_SULSO_1_314_0_4keu_A.pdb',
            help='protein file specified for generation')
    parser.add_argument(
            '--pdb_chain', action='store',required=False,type=str,default='A',
            help='protein chian specified for generation')
    parser.add_argument(
            '--sdf_file', action='store',required=False,type=str,default='../data/test_set/PHP_SULSO_1_314_0_4keu_A.sdf',
            help='original ligand sdf_file, only for providing center')
    parser.add_argument(
            '--center', action='store',required=False,type=str,default=None,
            help='provide center explcitly, e.g., 32.33,25.56,45.67')

    args = parser.parse_args()

    # Load configs
    config = load_config(args.config)
    seed_all(config.sample.seed)
    
    print(args)
    print(config)


    # # Transform
    print('Loading data...')
    protein_featurizer = FeaturizeProteinAtom()
    ligand_featurizer = FeaturizeLigandAtom()
    masking = LigandMaskAll()
    unmasking = LigandMaskNone()
    transform = Compose([
        RefineData(),
        LigandCountNeighbors(),
        protein_featurizer,
        ligand_featurizer,
        masking,
    ])

    # # Data# define the pocket data for generation
    if args.sdf_file is not None:
        mol= read_sdf(args.sdf_file)[0]
        atomCoords = np.array(mol.GetConformers()[0].GetPositions())
        data, idx, protein_dict = pdb_to_pocket_data(args.pdb_file, args.pdb_chain, center=atomCoords, bbox_size=args.bbox_size)

    if args.center is not None: 
        center = np.array([[float(i) for i in args.center.split(',')]])
        data, idx, protein_dict = pdb_to_pocket_data(args.pdb_file, args.pdb_chain, center=center, bbox_size=args.bbox_size)

    if data is None:
        sys.exit('pocket residues is None, please check the box you choose or the PDB file you upload')
    
    

    
    # save the generation results
    task_name = args.pdb_file.split('/')[-1][:-4]
    os.makedirs(args.outdir, exist_ok=True)
    task_dir = os.path.join(args.outdir,task_name)
    os.makedirs(task_dir, exist_ok=True)

    # docking top 1000 retrieval molecules
    retrieve_model = Retrieval(args.device, config.retrieve, task_dir)
    retrieve_model(args.pdb_file, idx, args.pdb_chain, protein_dict['pos'])


    print('retrieval model & data completed')

    # # Model (Main)
    print('Loading main model...')
    ckpt = torch.load(args.check_point, map_location=args.device)
    model = Rag2Mol(
        ckpt['config'].model, 
        num_classes = 7,
        protein_atom_feature_dim = protein_featurizer.feature_dim,
        ligand_atom_feature_dim = ligand_featurizer.feature_dim,
        num_bond_types = 3,
    ).to(args.device)
    model.load_state_dict(ckpt['model'])

    print('start sampling')
    # Sampling
    # The algorithm is the same as the one `sample.py`.

    pool = EasyDict({
        'queue': [],
        'failed': [],
        'finished': [],
        'duplicate': [],
        'smiles': set(),
    })
    # # Sample the first atoms
    print('Initialization')
    pbar = tqdm(total=config.sample.beam_size, desc='InitSample')
    atom_composer = AtomComposer(protein_featurizer.feature_dim, ligand_featurizer.feature_dim, 48)
    data = transform(data)
    data = transform_data(data, atom_composer)

    transform_ret = Compose([
        RefineData(),
        LigandCountNeighbors(),
        protein_featurizer,
        ligand_featurizer,
        unmasking,

        atom_composer
    ])


    clone_data = data.clone()
    try:
        ret_ligand = torchify_dict(retrieve_model.find_mcs(data))
        ret_data = ProteinLigandData.from_protein_ligand_dicts(protein_dict = protein_dict, ligand_dict = ret_ligand)
        data.retrieval_data = transform_ret(ret_data)
    except: 
        data.retrieval_data = transform_ret(clone_data)
    
    init_data_list = get_init(data.to(args.device),   # sample the initial atoms
            model = model,
            transform=atom_composer,
            threshold=config.sample.threshold
    )
    pool.queue = init_data_list
    if len(pool.queue) > config.sample.beam_size:
        pool.queue = init_data_list[:config.sample.beam_size]
        pbar.update(config.sample.beam_size)
    else:
        pbar.update(len(pool.queue))
    pbar.close()


    # # Sampling loop
    print('Start sampling')
    global_step = 0

    try:
        while len(pool.finished) < config.sample.num_samples:
            global_step += 1
            if global_step > config.sample.max_steps:
                break
            queue_size = len(pool.queue)
            # # sample candidate new mols from each parent mol
            queue_tmp = []
            queue_weight = []
            for data in tqdm(pool.queue):
                nexts = []
                
                if data.ligand_context_pos.size(0) % 5 == 0:
                    try:
                        ret_ligand = torchify_dict(retrieve_model.find_mcs(data))
                        ret_data = ProteinLigandData.from_protein_ligand_dicts(protein_dict = protein_dict, ligand_dict = ret_ligand)
                        data.retrieval_data = transform_ret(ret_data)
                    except: 
                        print('retrieve failed')
                        pass
                #else: 
                #    try:
                #        data.retrieval_data = transform_ret(clone_data)
                #    except: pass

                data_next_list = get_next(
                    data.to(args.device), 
                    model = model,
                    transform = atom_composer,
                    threshold = config.sample.threshold
                )

                for data_next in data_next_list:
                    if data_next.status == STATUS_FINISHED:
                        try:
                            rdmol = reconstruct_from_generated_with_edges(data_next)
                            data_next.rdmol = rdmol
                            mol = Chem.MolFromSmiles(Chem.MolToSmiles(rdmol))
                            smiles = Chem.MolToSmiles(mol)
                            data_next.smiles = smiles
                            if smiles in pool.smiles:
                                pool.duplicate.append(data_next)
                            elif '.' in smiles:
                                pool.failed.append(data_next)
                            else:   # Pass checks
                                print('Success: %s' % smiles)
                                pool.finished.append(data_next)
                                pool.smiles.add(smiles)
                        except MolReconsError:
                            pool.failed.append(data_next)
                    elif data_next.status == STATUS_RUNNING:
                        nexts.append(data_next)

                queue_tmp += nexts
            # # random choose mols from candidates
            prob = logp_to_rank_prob(np.array([p.average_logp[2:] for p in queue_tmp], dtype="object"))  # (logp_focal, logpdf_pos), logp_element, logp_hasatom, logp_bond
            n_tmp = len(queue_tmp)
            if not sum(prob):
                print('can not generate more')
                exit()
            next_idx = np.random.choice(np.arange(n_tmp), p=prob, size=min(config.sample.beam_size, n_tmp), replace=False)
            pool.queue = [queue_tmp[idx] for idx in next_idx]

    except:
        print('Terminated. Generated molecules will be saved.')




    SDF_dir = os.path.join(task_dir,'SDF')
    os.makedirs(SDF_dir, exist_ok=True)
    for j in range(len(pool['finished'])):
        writer = Chem.SDWriter(SDF_dir+f'/{j}.sdf')
        writer.write(pool['finished'][j].rdmol)
        writer.close()

    shutil.copy(args.pdb_file, task_dir)
