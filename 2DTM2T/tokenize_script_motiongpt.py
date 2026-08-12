import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from os.path import join as pjoin

import numpy as np
import torch
import codecs as cs
from tqdm import tqdm
from torch.utils.data import DataLoader

import utils.paramUtil as paramUtil
from options.train_options import TrainVQTokenizerOptions
from data.dataset import MotionTokenizeDataset
from src.models.mgpt_vq import VQVae


if __name__ == '__main__':
    parser = TrainVQTokenizerOptions()
    parser.parser.add_argument(
        '--motiongpt_ckpt',
        type=str,
        default=str(ROOT_DIR / 'checkpoints' / 'MotionGPT-base' / 'motiongpt_s3_h3d.tar'),
        help='Path to MotionGPT checkpoint (.tar)',
    )
    opt = parser.parse()
    opt.is_train = False

    opt.device = torch.device("cpu" if opt.gpu_id == -1 else "cuda:" + str(opt.gpu_id))
    if opt.gpu_id != -1:
        torch.cuda.set_device(opt.gpu_id)

    if opt.dataset_name == 't2m':
        opt.data_root = str(ROOT_DIR / 'dataset' / 'HumanML3D')
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        opt.joints_num = 22
        opt.max_motion_length = 196
        kinematic_chain = paramUtil.t2m_kinematic_chain
    elif opt.dataset_name == 'kit':
        opt.data_root = str(ROOT_DIR / 'dataset' / 'KIT-ML')
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        opt.joints_num = 21
        opt.max_motion_length = 196
        kinematic_chain = paramUtil.kit_kinematic_chain
    else:
        raise KeyError('Dataset Does Not Exist')

    # MotionGPT normalization statistics (already stored alongside VQVAEV3 checkpoint)
    mgpt_meta_dir = ROOT_DIR / 'checkpoints' / opt.dataset_name / 'VQVAEV3_CB1024_CMT_H1024_NRES3' / 'meta'
    mean = np.load(str(mgpt_meta_dir / 'mean.npy'))
    std  = np.load(str(mgpt_meta_dir / 'std.npy'))

    all_split_file = pjoin(opt.data_root, 'all.txt')

    # MotionGPT VQ-VAE (code_num=512, ema_reset quantizer)
    vqvae = VQVae(
        nfeats=263,
        quantizer="ema_reset",
        code_num=512,
        code_dim=512,
        output_emb_width=512,
        down_t=2,
        stride_t=2,
        width=512,
        depth=3,
        dilation_growth_rate=3,
        norm="none",
        activation="relu",
    ).to(opt.device)

    print(f'Loading MotionGPT checkpoint from {opt.motiongpt_ckpt}')
    checkpoint = torch.load(opt.motiongpt_ckpt, map_location='cpu', weights_only=True)
    new_state_dict = {
        key.replace("vae.", ""): values
        for key, values in checkpoint["state_dict"].items()
        if "vae" in key
    }
    vqvae.load_state_dict(state_dict=new_state_dict)
    vqvae.eval()

    all_dataset = MotionTokenizeDataset(opt, mean, std, all_split_file)
    all_loader = DataLoader(all_dataset, batch_size=1, num_workers=1, pin_memory=True)

    token_data_dir = pjoin(opt.data_root, opt.name)
    os.makedirs(token_data_dir, exist_ok=True)
    print(f'Token output directory: {token_data_dir}')

    num_replics = 5
    opt.unit_length = 4

    with torch.no_grad():
        # Generate num_replics token sequences per motion to improve robustness
        # (MotionTokenizeDataset introduces slight randomness via random crop)
        for e in range(num_replics):
            print(f'Replication {e + 1}/{num_replics}')
            for i, data in enumerate(tqdm(all_loader)):
                motion, name = data
                motion = motion.detach().to(opt.device).float()
                indices, _ = vqvae.encode(motion)
                indices = list(indices[0].cpu().numpy())
                indices = [str(token) for token in indices]
                with cs.open(pjoin(token_data_dir, '%s.txt' % name[0]), 'a+') as f:
                    if e == num_replics - 1:
                        f.write(' '.join(indices))
                    else:
                        f.write(' '.join(indices) + '\n')
