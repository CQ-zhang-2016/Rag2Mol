import numpy as np
import os

import torch

from torch_geometric.loader import DataLoader
from datetime import datetime
import sys
import argparse
import random

import shlex
import glob
import time
import pathlib

from tqdm import tqdm

from .fabind_utils.fabind_inference_dataset import InferenceDataset
from .fabind_utils.inference_mol_utils import write_mol
from .fabind_utils.post_optim_utils import post_optimize_compound_coords
import pandas as pd
from .models.model import *


def inference(supp_smiles, supp_names, protein_features, preprocess_dir, ckpt_dir, device, out_dir):
    parser = argparse.ArgumentParser(description='Train your own TankBind model.')

    parser.add_argument("-d", "--data", type=str, default="0",
                        help="data specify the data to use. \
                        0 for re-docking, 1 for self-docking.")
    parser.add_argument('--seed', type=int, default=128,
                        help="seed to use.")
    parser.add_argument("--gs-tau", type=float, default=1,
                        help="Tau for the temperature-based softmax.")
    parser.add_argument("--gs-hard", action='store_true', default=False,
                        help="Hard mode for gumbel softmax.")

    parser.add_argument("--restart", type=str, default=None,
                        help="continue the training from the model we saved from scratch.")
    parser.add_argument("--reload", type=str, default=None,
                        help="continue the training from the model we saved.")
    parser.add_argument("--addNoise", type=str, default=None,
                        help="shift the location of the pocket center in each training sample \
                        such that the protein pocket encloses a slightly different space.")

    pair_interaction_mask = parser.add_mutually_exclusive_group()
    # use_equivalent_native_y_mask is probably a better choice.
    pair_interaction_mask.add_argument("--use_y_mask", action='store_true', default=False,
                        help="mask the pair interaction during pair interaction loss evaluation based on data.real_y_mask. \
                        real_y_mask=True if it's the native pocket that ligand binds to.")
    pair_interaction_mask.add_argument("--use_equivalent_native_y_mask", action='store_true', default=False,
                        help="mask the pair interaction during pair interaction loss evaluation based on data.equivalent_native_y_mask. \
                        real_y_mask=True if most of the native interaction between ligand and protein happen inside this pocket.")

    parser.add_argument("--use_affinity_mask", type=int, default=0,
                        help="mask affinity in loss evaluation based on data.real_affinity_mask")
    parser.add_argument("--affinity_loss_mode", type=int, default=1,
                        help="define which affinity loss function to use.")

    parser.add_argument("--pred_dis", type=int, default=1,
                        help="pred distance map or predict contact map.")
    parser.add_argument("--posweight", type=int, default=8,
                        help="pos weight in pair contact loss, not useful if args.pred_dis=1")

    parser.add_argument("--relative_k", type=float, default=0.01,
                        help="adjust the strength of the affinity loss head relative to the pair interaction loss.")
    parser.add_argument("-r", "--relative_k_mode", type=int, default=0,
                        help="define how the relative_k changes over epochs")

    parser.add_argument("--label", type=str, default="",
                        help="information you want to keep a record.")

    parser.add_argument("--use-whole-protein", action='store_true', default=False,
                        help="currently not used.")

    parser.add_argument("--data-path", type=str, default="",
                        help="Data path.")
                        
    parser.add_argument("--exp-name", type=str, default="",
                        help="data path.")

    parser.add_argument("--tqdm-interval", type=float, default=0.1,
                        help="tqdm bar update interval")

    parser.add_argument("--lr", type=float, default=0.0001)

    parser.add_argument("--pocket-coord-huber-delta", type=float, default=3.0)

    parser.add_argument("--coord-loss-function", type=str, default='SmoothL1', choices=['MSE', 'SmoothL1'])

    parser.add_argument("--coord-loss-weight", type=float, default=1.0)
    parser.add_argument("--pair-distance-loss-weight", type=float, default=1.0)
    parser.add_argument("--pair-distance-distill-loss-weight", type=float, default=1.0)
    parser.add_argument("--pocket-cls-loss-weight", type=float, default=1.0)
    parser.add_argument("--pocket-distance-loss-weight", type=float, default=0.05)
    parser.add_argument("--pocket-cls-loss-func", type=str, default='bce', choices=['bce', 'dice'])

    # parser.add_argument("--warm-mae-thr", type=float, default=5.0)

    parser.add_argument("--use-compound-com-cls", action='store_true', default=False,
                        help="only use real pocket to run pocket classification task")

    parser.add_argument("--compound-coords-init-mode", type=str, default="pocket_center_rdkit",
                        choices=['pocket_center_rdkit', 'pocket_center', 'compound_center', 'perturb_3A', 'perturb_4A', 'perturb_5A', 'random', 'diffdock'])

    parser.add_argument('--trig-layers', type=int, default=1)

    parser.add_argument('--distmap-pred', type=str, default='mlp',
                        choices=['mlp', 'trig'])
    parser.add_argument('--mean-layers', type=int, default=3)
    parser.add_argument('--n-iter', type=int, default=8)
    parser.add_argument('--inter-cutoff', type=float, default=10.0)
    parser.add_argument('--intra-cutoff', type=float, default=8.0)
    parser.add_argument('--refine', type=str, default='refine_coord',
                        choices=['stack', 'refine_coord'])

    parser.add_argument('--coordinate-scale', type=float, default=5.0)
    parser.add_argument('--geometry-reg-step-size', type=float, default=0.001)
    parser.add_argument('--lr-scheduler', type=str, default="constant", choices=['constant', 'poly_decay', 'cosine_decay', 'cosine_decay_restart', 'exp_decay'])

    parser.add_argument('--add-attn-pair-bias', action='store_true', default=False)
    parser.add_argument('--explicit-pair-embed', action='store_true', default=False)
    parser.add_argument('--opm', action='store_true', default=False)

    parser.add_argument('--add-cross-attn-layer', action='store_true', default=False)
    parser.add_argument('--rm-layernorm', action='store_true', default=False)
    parser.add_argument('--keep-trig-attn', action='store_true', default=False)

    parser.add_argument('--pocket-radius', type=float, default=20.0)

    parser.add_argument('--rm-LAS-constrained-optim', action='store_true', default=False)
    parser.add_argument('--rm-F-norm', action='store_true', default=False)
    parser.add_argument('--norm-type', type=str, default="per_sample", choices=['per_sample', '4_sample', 'all_sample'])

    # parser.add_argument("--only-predicted-pocket-mae-thr", type=float, default=3.0)
    parser.add_argument('--noise-for-predicted-pocket', type=float, default=5.0)
    parser.add_argument('--test-random-rotation', action='store_true', default=False)

    parser.add_argument('--random-n-iter', action='store_true', default=False)
    parser.add_argument('--clip-grad', action='store_true', default=False)

    # one batch actually contains 20000 samples, not the size of training set
    parser.add_argument("--sample-n", type=int, default=0, help="number of samples in one epoch.")

    parser.add_argument('--fix-pocket', action='store_true', default=False)
    parser.add_argument('--pocket-idx-no-noise', action='store_true', default=False)
    parser.add_argument('--ablation-no-attention', action='store_true', default=False)
    parser.add_argument('--ablation-no-attention-with-cross-attn', action='store_true', default=False)

    parser.add_argument('--redocking', action='store_true', default=True)
    parser.add_argument('--redocking-no-rotate', action='store_true', default=False)

    parser.add_argument("--pocket-pred-layers", type=int, default=1, help="number of layers for pocket pred model.")
    parser.add_argument('--pocket-pred-n-iter', type=int, default=1, help="number of iterations for pocket pred model.")

    parser.add_argument('--use-esm2-feat', action='store_true', default=False)
    parser.add_argument("--center-dist-threshold", type=float, default=8.0)

    parser.add_argument("--mixed-precision", type=str, default='no', choices=['no', 'fp16'])
    parser.add_argument('--disable-tqdm', action='store_true', default=False)
    parser.add_argument('--log-interval', type=int, default=100)
    parser.add_argument('--optim', type=str, default='adam', choices=['adam', 'adamw'])
    parser.add_argument("--warmup-epochs", type=int, default=15,
                        help="used in combination with relative_k_mode.")
    parser.add_argument("--total-epochs", type=int, default=400, help="option to switch training data after certain epochs.")
    parser.add_argument('--disable-validate', action='store_true', default=False)
    parser.add_argument('--disable-tensorboard', action='store_true', default=False)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--stage-prob", type=float, default=0.5)
    parser.add_argument("--pocket-pred-hidden-size", type=int, default=128)

    parser.add_argument("--local-eval", action='store_true', default=False)
    # parser.add_argument("--eval-dir", type=str, default=None)

    parser.add_argument("--train-ligand-torsion-noise", action='store_true', default=False)
    parser.add_argument("--train-pred-pocket-noise", type=float, default=0.0)
    parser.add_argument("--esm2-concat-raw", action='store_true', default=False)
    parser.add_argument("--test-sample-n", type=int, default=1)
    parser.add_argument("--return-hidden", action='store_true', default=False)
    parser.add_argument("--confidence-task", type=str, default='classification', choices=['classification', 'regression', 'perfect'])
    parser.add_argument("--confidence-rmsd-thr", type=float, default=2.0)
    parser.add_argument("--confidence-thr", type=float, default=0.5)

    parser.add_argument("--post-optim", action='store_true', default=True)
    parser.add_argument('--post-optim-mode', type=int, default=0)
    parser.add_argument('--post-optim-epoch', type=int, default=1000)

    parser.add_argument("--ensemble", action='store_true', default=False)
    parser.add_argument("--confidence", action='store_true', default=False)
    parser.add_argument("--test-gumbel-soft", action='store_true', default=True)
    parser.add_argument("--test-pocket-noise", type=float, default=5)
    parser.add_argument("--test-unseen", action='store_true', default=False)



    command = "main_fabind.py -d 0 --label baseline --addNoise 5 --tqdm-interval 60 --use-compound-com-cls --distmap-pred mlp --n-iter 8 --mean-layers 4 --refine refine_coord --coordinate-scale 5 --geometry-reg-step-size 0.001 --rm-layernorm --add-attn-pair-bias --explicit-pair-embed --add-cross-attn-layer --noise-for-predicted-pocket 0.0 --clip-grad --random-n-iter --pocket-idx-no-noise --seed 128 --use-esm2-feat --pocket-pred-layers 1 --pocket-pred-n-iter 1 --center-dist-threshold 4 --pocket-cls-loss-func bce --mixed-precision no --disable-tqdm --disable-validate --log-interval 50 --optim adamw --norm-type per_sample --weight-decay 0.01 --hidden-size 512 --pocket-pred-hidden-size 128 --stage-prob 0.25"
    command = shlex.split(command)
    args = parser.parse_args(command[1:])
        
    args.local_eval = False
    args.data_path = ""
    args.seed = 128
    args.exp_name = ""
    args.return_hidden = False
    args.confidence_task = 'classification'
    args.confidence_rmsd_thr = 2.0
    args.confidence_thr = 0.5
    args.test_sample_n = 1
    args.disable_tqdm = False
    args.tqdm_interval = 0.1
    args.train_pred_pocket_noise = 0.0
    args.post_optim = True
    args.post_optim_mode = 0
    args.post_optim_epoch = 5000
    args.ensemble = False
    args.confidence = False
    args.test_gumbel_soft = True
    args.test_pocket_noise = 5
    args.test_unseen = False
    args.gs_tau = 1
    args.compound_coords_init_mode = 'pocket_center_rdkit'
    args.n_iter = 8
    args.redocking = True

    '''
    args.local_eval = args_new.local_eval
    args.data_path = args_new.data_path
    args.seed = args_new.seed
    args.exp_name = args_new.exp_name
    args.return_hidden = args_new.return_hidden
    args.confidence_task = args_new.confidence_task
    args.confidence_rmsd_thr = args_new.confidence_rmsd_thr
    args.confidence_thr = args_new.confidence_thr
    args.test_sample_n = args_new.test_sample_n
    args.disable_tqdm = False
    args.tqdm_interval = 0.1
    args.train_pred_pocket_noise = args_new.train_pred_pocket_noise
    args.post_optim = args_new.post_optim
    args.post_optim_mode = args_new.post_optim_mode
    args.post_optim_epoch = args_new.post_optim_epoch
    args.ensemble = args_new.ensemble
    args.confidence = args_new.confidence
    args.test_gumbel_soft = args_new.test_gumbel_soft
    args.test_pocket_noise = args_new.test_pocket_noise
    args.test_unseen = args_new.test_unseen
    args.gs_tau = args_new.gs_tau
    args.compound_coords_init_mode = args_new.compound_coords_init_mode
    args.n_iter = args_new.n_iter
    args.redocking = args_new.redocking
    '''


    print("start inference")

    # torch.set_num_threads(16)
    # # ----------without this, I could get 'RuntimeError: received 0 items of ancdata'-----------
    torch.multiprocessing.set_sharing_strategy('file_system')
    os.makedirs(out_dir, exist_ok=True)
    # train, valid, test: only native pocket. train_after_warm_up, all_pocket_test include all other pockets(protein center and P2rank result)
    args.compound_coords_init_mode = "redocking"

    def post_optim_mol(args, data, com_coord_pred, compound_batch, LAS_tmp):
        
        for i in range(compound_batch.max().item()+1):

            i_mask = (compound_batch == i)
            predict_coord = com_coord_pred[i_mask]

            save_coords = predict_coord + data[i].coord_offset
            mol = write_mol(reference_mol=data[i].mol, coords=save_coords, output_file=os.path.join(out_dir, data[i].uid+'.sdf'))

        return 



    dataset = InferenceDataset(supp_smiles, supp_names, protein_features, preprocess_dir)
    print(f"data point: {len(dataset)}")
    num_workers = 0
    data_loader = DataLoader(dataset, batch_size=1, follow_batch=['x'], shuffle=False, pin_memory=False, num_workers=num_workers)

    model = get_model(args)

    model.load_state_dict(torch.load(ckpt_dir))
    model = model.to(device).eval()


    print(f"Begin inference")
    start_time = time.time()  # 记录开始时间

    for batch_id, data in enumerate(tqdm(data_loader)):
        #try:
        data = data.to(device)

        LAS_tmp = []
        for i in range(len(data)):
            LAS_tmp.append(data[i]['compound', 'LAS', 'compound'].edge_index.detach().clone())

        with torch.no_grad():
            com_coord_pred, compound_batch = model.inference(data.clone().to(device))  

        post_optim_mol(args, data, com_coord_pred, compound_batch, LAS_tmp=LAS_tmp)
        
        #except:
        #    print('inference error')
        #    continue



    end_time = time.time()  # 记录开始时间
    print(f"End test, time spent: {end_time - start_time}")


