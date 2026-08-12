import argparse
import sys
from pathlib import Path
import torch
import numpy as np
from options.get_eval_option import get_opt
from models.evaluator_wrapper import EvaluatorModelWrapper
import models.vqvae as vqvae
from options import option
import utils.utils_model as utils_model
import json
from dataloader.eval_loader import M2DT_DATALoader
from utils.evaluate import evaluation_m2dt
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch import cuda

ROOT_DIR = Path(__file__).resolve().parent


class _VQVae2DReconWrapper:
    """Wraps 2DMotionGPT VQVae so encode() returns only indices (same interface as HumanVQVAE)."""
    def __init__(self, inner):
        self._inner = inner

    def encode(self, x):
        idx, _ = self._inner.encode(x.float())
        return idx

    def eval(self):
        self._inner.eval()
        return self

    def cuda(self):
        self._inner.cuda()
        return self


def _run_eval(val_loader, vae, model, logger, tokenizer, args, use_2d):
    bleu1, bleu4, bleu7, rouge, cider, bert_score = [], [], [], [], [], []
    s_bleu1, s_bleu4, s_bleu7, s_rouge, s_cider, s_bert_score = [], [], [], [], [], []
    repeat_time = 1
    for _ in range(repeat_time):
        best_bleu1, best_bleu4, best_bleu7, best_rouge, best_cider, best_bert_score, \
        best_s_bleu1, best_s_bleu4, best_s_bleu7, best_s_rouge, best_s_cider, best_s_bert_score, \
        logger = evaluation_m2dt(
            val_loader, vae, model, logger, tokenizer,
            instruction=args.prompt,
            max_new_tokens=1536,
            use_2d=use_2d,
        )
        bleu1.append(best_bleu1); bleu4.append(best_bleu4); bleu7.append(best_bleu7)
        rouge.append(best_rouge); cider.append(best_cider); bert_score.append(best_bert_score)
        s_bleu1.append(best_s_bleu1); s_bleu4.append(best_s_bleu4); s_bleu7.append(best_s_bleu7)
        s_rouge.append(best_s_rouge); s_cider.append(best_s_cider); s_bert_score.append(best_s_bert_score)
    return {
        'bleu1': np.array(bleu1), 'bleu4': np.array(bleu4), 'bleu7': np.array(bleu7),
        'rouge': np.array(rouge), 'cider': np.array(cider), 'bert_score': np.array(bert_score),
        's_bleu1': np.array(s_bleu1), 's_bleu4': np.array(s_bleu4), 's_bleu7': np.array(s_bleu7),
        's_rouge': np.array(s_rouge), 's_cider': np.array(s_cider), 's_bert_score': np.array(s_bert_score),
        'repeat_time': repeat_time,
    }


def _print_results(tag, r):
    rt = r['repeat_time']
    print(f'\n=== {tag} Sequence-Level ===')
    for k in ['bleu1', 'bleu4', 'bleu7', 'rouge', 'cider', 'bert_score']:
        print(f'{k}: {np.mean(r[k]):.3f}')
    print(f'=== {tag} Snippet-Level ===')
    for k in ['s_bleu1', 's_bleu4', 's_bleu7', 's_rouge', 's_cider', 's_bert_score']:
        print(f'{k[2:]}: {np.mean(r[k]):.3f}')

    def _fmt(keys, prefix=''):
        return ', '.join(
            f"{k.removeprefix(prefix)}. {np.mean(r[k]):.3f}, conf. {np.std(r[k]) * 1.96 / np.sqrt(rt):.3f}"
            for k in keys
        )
    msg = (
        f"[{tag}] Sequence-Level:\n"
        + _fmt(['bleu1', 'bleu4', 'bleu7', 'rouge', 'cider', 'bert_score'])
        + f"\n[{tag}] Snippet-Level:\n"
        + _fmt(['s_bleu1', 's_bleu4', 's_bleu7', 's_rouge', 's_cider', 's_bert_score'], prefix='s_')
    )
    return msg


if __name__ == "__main__":

    parser = option.get_args_parser()

    # set hyperparameters
    parser.add_argument("--model_name", type=str, default="./m2dt-ft-from-t5-base/checkpoint-300000/", help="Trained model name or directory")
    parser.add_argument("--prompt", type=str, default="Generate the motion script: ", help="Motion-to-Detailed Text Prompt")
    parser.add_argument("--vqvae_2drecon_ckpt", type=str, default=None,
                        help="Path to 2DRecon VQ-VAE checkpoint (.tar). When set, the 2D evaluation "
                             "uses the full 2DRecon VQ-VAE (same as training tokenization).")
    parser.add_argument("--vqvae_2d_seed", type=int, default=None,
                        help="Seed used for 2D encoder training. When set, auto-search looks in "
                             "checkpoints/2d_vq_train/{dataname}/seed{N}/ instead of the default directory.")
    args = parser.parse_args()



    # Evaluator Setting
    if args.dataname == 'kit':
        dataset_opt_path = './checkpoints/kit/Comp_v6_KLD005/opt.txt'
        args.nb_joints = 21
    else:
        dataset_opt_path = './checkpoints/t2m/Comp_v6_KLD005/opt.txt'
        args.nb_joints = 22

    wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'))
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)



    # Test set
    val_loader = M2DT_DATALoader(args.dataname, 'test', 32)



    # VQ-VAE (3D weights)
    print('Loading VAE')
    vae = vqvae.HumanVQVAE(args,
                           512,
                           args.code_dim,
                           args.output_emb_width,
                           2,
                           args.stride_t,
                           args.width,
                           3,
                           args.dilation_growth_rate)
    resume_pth = ROOT_DIR / "checkpoints" / "pretrained_vqvae" / f"{args.dataname}.pth"
    ckpt = torch.load(resume_pth, map_location='cpu')
    vae.load_state_dict(ckpt['net'], strict=True)
    vae = vae.cuda().eval()
    print('Loading VAE Done')



    # set logger
    logger = utils_model.get_test_logger(args.model_name, 'test_m2dt_run.log')
    logger.info(json.dumps(vars(args), indent=4, sort_keys=True))



    # motion aware language model setting
    tokenizer = T5Tokenizer.from_pretrained(args.model_name)
    model = T5ForConditionalGeneration.from_pretrained(args.model_name)

    device = 'cuda' if cuda.is_available() else 'cpu'
    model = model.to(device)



    # ---- 3D evaluation ----
    print('\n========== Evaluating 3D (original encoder) ==========')
    res_3d = _run_eval(val_loader, vae, model, logger, tokenizer, args, use_2d=False)
    msg_3d = _print_results('3D', res_3d)
    logger.info(msg_3d)

    # ---- Load 2D encoder (or full 2DRecon VQ-VAE) ----
    if args.vqvae_2drecon_ckpt:
        motiongpt_dir = ROOT_DIR.parent / '2DMotionGPT'
        sys.path.insert(0, str(motiongpt_dir))
        from src.models.mgpt_vq import VQVae as _VQVae2DRecon
        dim_feat = 263 if args.dataname == 't2m' else 251
        vae_2drecon = _VQVae2DRecon(
            nfeats=dim_feat,
            quantizer='ema_reset',
            code_num=512,
            code_dim=512,
            output_emb_width=512,
            down_t=2,
            stride_t=2,
            width=512,
            depth=3,
            dilation_growth_rate=3,
            norm='none',
            activation='relu',
        )
        print(f'\nLoading 2DRecon VQ-VAE checkpoint: {args.vqvae_2drecon_ckpt}')
        ckpt_2drecon = torch.load(args.vqvae_2drecon_ckpt, map_location='cpu', weights_only=False)
        state = ckpt_2drecon.get('model_state_dict', ckpt_2drecon)
        vae_2drecon.load_state_dict(state)
        vae_2d = _VQVae2DReconWrapper(vae_2drecon).cuda().eval()
        print('Loading 2DRecon VQ-VAE Done')
        label_2d = '2DRecon'
    else:
        if args.vqvae_2d_seed is not None:
            ckpt_2d_dir = ROOT_DIR / "checkpoints" / "2d_vq_train" / args.dataname / f"seed{args.vqvae_2d_seed}"
        else:
            ckpt_2d_dir = ROOT_DIR / "checkpoints" / "2d_vq_train" / args.dataname
        candidates = sorted(ckpt_2d_dir.glob("best_2dvq_epoch*_ratio*.pt"))
        if not candidates:
            raise FileNotFoundError(f"No 2D encoder checkpoint found in {ckpt_2d_dir}")
        ckpt_2d_path = candidates[-1]
        print(f'\nLoading 2D encoder checkpoint: {ckpt_2d_path}')
        ckpt_2d = torch.load(ckpt_2d_path, map_location='cpu')
        new_state_dict = {k: v for k, v in ckpt_2d["net"].items() if k.startswith("vqvae.encoder")}
        vae.load_state_dict(new_state_dict, strict=False)
        vae.eval()
        print('Loading 2D encoder weights Done')
        vae_2d = vae
        label_2d = '2D'

    # ---- 2D evaluation ----
    print(f'\n========== Evaluating {label_2d} ==========')
    res_2d = _run_eval(val_loader, vae_2d, model, logger, tokenizer, args, use_2d=True)
    msg_2d = _print_results(label_2d, res_2d)
    logger.info(msg_2d)

    # ---- Comparison ----
    print(f'\n========== Comparison (3D vs {label_2d}) ==========')
    print('Sequence-Level:')
    for k in ['bleu1', 'bleu4', 'bleu7', 'rouge', 'cider', 'bert_score']:
        print(f'  {k}: 3D={np.mean(res_3d[k]):.3f}  {label_2d}={np.mean(res_2d[k]):.3f}  diff={np.mean(res_2d[k]) - np.mean(res_3d[k]):+.3f}')
    print('Snippet-Level:')
    for k in ['s_bleu1', 's_bleu4', 's_bleu7', 's_rouge', 's_cider', 's_bert_score']:
        label = k[2:]
        print(f'  {label}: 3D={np.mean(res_3d[k]):.3f}  {label_2d}={np.mean(res_2d[k]):.3f}  diff={np.mean(res_2d[k]) - np.mean(res_3d[k]):+.3f}')
