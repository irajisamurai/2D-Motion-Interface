import torch
from src.data.humanml.scripts.motion_process import (process_file, recover_from_ric)
from src.losses.utils import compose_loss
from os.path import join as pjoin
import numpy as np
from tqdm import tqdm

def feats2joints(features,mean,std):
        mean = torch.tensor(mean).to(features)
        std = torch.tensor(std).to(features)
        features = features * std + mean
        return recover_from_ric(features, 22)
    
def train_one_epoch(cfg,vqvae, ref_vqvae, _train_dataloader, recon_losses, loss_fn, optimizer, device):
    vqvae.train()
    ref_vqvae.eval()
    dis_data_root = pjoin(cfg.DATASET.HUMANML3D.MEAN_STD_PATH, 't2m', "VQVAEV3_CB1024_CMT_H1024_NRES3", "meta")
    mean = np.load(pjoin(dis_data_root, "mean.npy"))
    std = np.load(pjoin(dis_data_root, "std.npy"))
    train_total_loss = 0.0
    train_total_encoder_loss = 0.0
    train_total_rst_loss = 0.0
    for i, batch in tqdm(enumerate(_train_dataloader)):
        optimizer.zero_grad()
        feats_3d = batch["motion"].to(device)
        feats_2d = torch.cat([batch["motion_2d"], torch.zeros(batch["motion_2d"].shape[0], batch["motion_2d"].shape[1], 263-batch["motion_2d"].shape[2])], dim=-1).to(device)
        joints_ref = feats2joints(feats_3d,mean,std)
        with torch.no_grad():
            ref_encoded_motions = ref_vqvae.encoder(ref_vqvae.preprocess(feats_3d))
            ref_recon_feats_3d, loss_commit, perplexity = ref_vqvae(feats_3d)
        vqvae_encoded_motions = vqvae.encoder(vqvae.preprocess(feats_2d))
        """with torch.no_grad():
            vqvae_encoded_quantized, loss_commit, perplexity = ref_vqvae.quantizer(vqvae_encoded_motions)
            vqvae_encoded_decoder = ref_vqvae.decoder(vqvae_encoded_quantized)
            vqvae_encoded_out = ref_vqvae.postprocess(vqvae_encoded_decoder)
            joints_rst = feats2joints(vqvae_encoded_out,mean,std)"""
        vqvae_encoded_quantized, loss_commit, perplexity = vqvae.quantizer(vqvae_encoded_motions)
        vqvae_encoded_decoder = vqvae.decoder(vqvae_encoded_quantized)
        vqvae_encoded_out = vqvae.postprocess(vqvae_encoded_decoder)
        joints_rst = feats2joints(vqvae_encoded_out,mean,std)
        rs_set = {
            "m_ref": feats_3d,
            "joints_ref": joints_ref,
            "m_rst": vqvae_encoded_out,
            "joints_rst": joints_rst,
            "loss_commit": loss_commit,
            "perplexity": perplexity,
        }
        
        loss_encoder = loss_fn(vqvae_encoded_motions, ref_encoded_motions)
        loss_rst = recon_losses['losses_' + "train"].update(rs_set)
        #print("loss total:", type(loss), getattr(loss, "requires_grad", None))
        loss_terms = {
            "encoder": loss_encoder,
            "recon": loss_rst,
        }
        
        loss, used_terms = compose_loss(loss_terms, cfg.LOSS.terms)

        loss.backward()
        optimizer.step()
        train_total_loss += loss.item()
        train_total_encoder_loss += loss_encoder.item()
        train_total_rst_loss += loss_rst.item()
    log = {
        "train_loss": train_total_loss / len(_train_dataloader),
        "train_encoder_loss": train_total_encoder_loss / len(_train_dataloader),
        "train_rst_loss": train_total_rst_loss / len(_train_dataloader),
    }
    return log

def eval_one_epoch(cfg,vqvae, ref_vqvae, _val_dataloader, recon_losses, loss_fn, device,mr_metrics=None, tm2t_metric=None):
    val_total_encoder_loss = 0.0
    val_total_rst_loss = 0.0
    correct = 0
    total = 0
    dis_data_root = pjoin(cfg.DATASET.HUMANML3D.MEAN_STD_PATH, 't2m', "VQVAEV3_CB1024_CMT_H1024_NRES3", "meta")
    mean = np.load(pjoin(dis_data_root, "mean.npy"))
    std = np.load(pjoin(dis_data_root, "std.npy"))
    vqvae.eval()
    ref_vqvae.eval()
    for i, batch in enumerate(_val_dataloader):
        feats_3d = batch["motion"].to(device)
        feats_2d = torch.cat([batch["motion_2d"], torch.zeros(batch["motion_2d"].shape[0], batch["motion_2d"].shape[1], 263-batch["motion_2d"].shape[2])], dim=-1).to(device)
        joints_ref = feats2joints(feats_3d,mean,std)
        # motion encode & decode
        with torch.no_grad():
            ref_encoded_motions = ref_vqvae.encoder(ref_vqvae.preprocess(feats_3d))
            ref_code_idx,_ = ref_vqvae.encode(feats_3d)
            ref_recon_feats_3d, loss_commit, perplexity = ref_vqvae(feats_3d)
            N, T, _ = feats_2d.shape
            vqvae_encoded_motions = vqvae.encoder(vqvae.preprocess(feats_2d))
            
            code_idx = ref_vqvae.postprocess(vqvae_encoded_motions)
            code_idx = code_idx.contiguous().view(-1,
                                        code_idx.shape[-1])  # (NT, C)
            code_idx = ref_vqvae.quantizer.quantize(code_idx)
            code_idx = code_idx.view(N, -1)
            
            vqvae_encoded_quantized, loss_commit, perplexity = ref_vqvae.quantizer(vqvae_encoded_motions)
            vqvae_encoded_decoder = ref_vqvae.decoder(vqvae_encoded_quantized)
            vqvae_encoded_out = ref_vqvae.postprocess(vqvae_encoded_decoder)
            joints_rst = feats2joints(vqvae_encoded_out,mean,std)
            
            rs_set = {
                "m_ref": feats_3d,
                "joints_ref": joints_ref,
                "m_rst": vqvae_encoded_out,
                "joints_rst": joints_rst,
                "loss_commit": loss_commit,
                "perplexity": perplexity,
            }
            
            loss_encoder = loss_fn(vqvae_encoded_motions, ref_encoded_motions)
            loss_rst = recon_losses['losses_' + "test"].update(rs_set)
            
            if mr_metrics is not None:
                mr_metrics.update(
                joints_rst = rs_set["joints_rst"],
                joints_ref = rs_set["joints_ref"],
                lengths=batch["length"],
            )
            if tm2t_metric is not None:
                tm2t_metric.update(
                feats_ref=rs_set["m_ref"],
                feats_rst=rs_set["m_rst"],
                lengths_ref=batch["length"],
                lengths_rst=batch["length"],
                word_embs=None,
                pos_ohot=None,
                text_lengths=None,
            )
        correct += (ref_code_idx == code_idx).sum().item()
        val_total_encoder_loss += loss_encoder.item()
        val_total_rst_loss += loss_rst.item()
        total += ref_code_idx.numel()
    
    if mr_metrics is not None and tm2t_metric is not None:
        results = tm2t_metric.compute(sanity_flag=False)
        mr_results = mr_metrics.compute(sanity_flag=False)
        tm2t_metric.reset()
        mr_metrics.reset()
        log = {
            "val_loss": (val_total_encoder_loss + val_total_rst_loss) / len(_val_dataloader),
            "val_encoder_loss": val_total_encoder_loss / len(_val_dataloader),
            "val_rst_loss": val_total_rst_loss / len(_val_dataloader),
            "val_accuracy": correct / total,
            "FID" : results["FID"],
            "Div" : results["Diversity"],
            "MPJPE" : mr_results["MPJPE"].item(),
            "PAMPJPE" : mr_results["PAMPJPE"].item(),
            "ACCEL" : mr_results["ACCEL"].item()
        }
    else:
        log = {
            "val_loss": (val_total_encoder_loss + val_total_rst_loss) / len(_val_dataloader),
            "val_encoder_loss": val_total_encoder_loss / len(_val_dataloader),
            "val_rst_loss": val_total_rst_loss / len(_val_dataloader),
            "val_accuracy": correct / total,
        }
    return log