import torch
from src.losses.mgpt import GPTLosses

def build_loss_functions(cfg,device):
    loss_function_name = cfg.LOSS.ABLATION.ENCODER_LOSS
    if loss_function_name == 'mse':
        loss_fn = torch.nn.MSELoss()
    elif loss_function_name == 'l1':
        loss_fn = torch.nn.L1Loss()
    elif loss_function_name == 'l1_smooth':
        loss_fn = torch.nn.SmoothL1Loss()
    recon_losses = torch.nn.ModuleDict({
            split: GPTLosses(cfg, "vae", 22)
            for split in ["losses_train", "losses_test", "losses_val"]
        }).to(device)
    return recon_losses, loss_fn

def build_optim(cfg,vqvae):
    optim_name = cfg.TRAIN.OPTIM.target
    if optim_name == 'AdamW':
        optimizer = torch.optim.AdamW(vqvae.parameters(), lr=cfg.TRAIN.OPTIM.params.lr, weight_decay=cfg.TRAIN.OPTIM.params.weight_decay, betas=cfg.TRAIN.OPTIM.params.betas)
    elif optim_name == 'SGD':
        optimizer = torch.optim.SGD(vqvae.parameters(), lr=cfg.TRAIN.OPTIM.params.lr, weight_decay=cfg.TRAIN.OPTIM.params.weight_decay, momentum=0.9)
    return optimizer