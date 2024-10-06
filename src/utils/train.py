import copy
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch

def repeat_data(data: Data, num_repeat) -> Batch:
    datas = [copy.deepcopy(data) for i in range(num_repeat)]
    return Batch.from_data_list(datas)


def repeat_batch(batch: Batch, num_repeat) -> Batch:
    datas = batch.to_data_list()
    new_data = []
    for i in range(num_repeat):
        new_data += copy.deepcopy(datas)
    return Batch.from_data_list(new_data)

def get_optimizer(cfg, model):
    return torch.optim.Adam(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=(cfg.beta1, cfg.beta2, )
    )


def get_scheduler(cfg, optimizer):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=cfg.factor,
        patience=cfg.patience,
        min_lr=cfg.min_lr
    )

def get_composed_idx(lig_idx, pkt_idx, lig_batch, pkt_batch):
    compose_batch_idx = (torch.zeros(len(lig_idx)+len(pkt_idx))).long()
    compose_batch_idx[pkt_idx] = pkt_batch
    compose_batch_idx[lig_idx] = lig_batch

    return compose_batch_idx


def get_model_loss(model, batch, config, retrieval_batch_idx, device, test_flag):
    if not test_flag:
        compose_noise = torch.randn_like(batch.compose_pos) * config.train.pos_noise_std
        ret_compose_noise = torch.zeros_like(batch.retrieval_data.compose_pos)
    else:
        compose_noise = torch.zeros_like(batch.compose_pos)
        ret_compose_noise = torch.zeros_like(batch.retrieval_data.compose_pos)


    retrieval_batch_idx = torch.Tensor(retrieval_batch_idx).long().to(device)

    
    loss, loss_frontier, loss_pos, loss_cls, loss_edge, loss_real, loss_fake, loss_surf = model(
            pos_real = batch.pos_real.to(device),
            y_real = batch.cls_real.long().to(device),
            # p_real = batch.ind_real.float(),    # Binary indicators: float
            pos_fake = batch.pos_fake.to(device),

            edge_index_real = torch.stack([batch.real_compose_edge_index_0, batch.real_compose_edge_index_1], dim=0).to(device),
            edge_label = batch.real_compose_edge_type.to(device),
            
            index_real_cps_edge_for_atten = batch.index_real_cps_edge_for_atten.to(device),
            tri_edge_index = batch.tri_edge_index.to(device),
            tri_edge_feat = batch.tri_edge_feat.to(device),

            compose_feature = batch.compose_feature.float().to(device),
            compose_pos = (batch.compose_pos + compose_noise).to(device),
            idx_ligand = batch.idx_ligand_ctx_in_compose.to(device),
            idx_protein = batch.idx_protein_in_compose.to(device),

            y_frontier = batch.ligand_frontier.to(device),
            idx_focal = batch.idx_focal_in_compose.to(device),
            pos_generate = batch.pos_generate.to(device),
            idx_protein_all_mask = batch.idx_protein_all_mask.to(device),
            y_protein_frontier = batch.y_protein_frontier.to(device),

            compose_knn_edge_index = batch.compose_knn_edge_index.to(device),
            compose_knn_edge_feature = batch.compose_knn_edge_feature.to(device),
            real_compose_knn_edge_index = torch.stack([batch.real_compose_knn_edge_index_0, batch.real_compose_knn_edge_index_1], dim=0).to(device),
            fake_compose_knn_edge_index = torch.stack([batch.fake_compose_knn_edge_index_0, batch.fake_compose_knn_edge_index_1], dim=0).to(device),


            compose_batch_idx = get_composed_idx(batch.idx_ligand_ctx_in_compose, batch.idx_protein_in_compose, batch.idx_ligand_ctx_in_compose_batch, batch.idx_protein_in_compose_batch).to(device),
            
            ret_batch_idx = retrieval_batch_idx.to(device),
            ret_compose_feature = batch.retrieval_data.compose_feature.float().to(device),
            ret_compose_pos = (batch.retrieval_data.compose_pos + ret_compose_noise).to(device),
            ret_idx_ligand = batch.retrieval_data.idx_ligand_ctx_in_compose.to(device),
            ret_idx_protein = batch.retrieval_data.idx_protein_in_compose.to(device),

            ret_compose_knn_edge_index = batch.retrieval_data.compose_knn_edge_index.to(device),
            ret_compose_knn_edge_feature = batch.retrieval_data.compose_knn_edge_feature.to(device),
        )
    return loss, loss_frontier, loss_pos, loss_cls, loss_edge, loss_real, loss_fake, loss_surf
