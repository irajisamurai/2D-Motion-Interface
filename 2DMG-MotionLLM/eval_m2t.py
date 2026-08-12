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
from dataloader.eval_loader import M2T_DATALoader
from utils.evaluate import evaluation_m2t
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


def _run_eval(val_loader, vae, model, logger, tokenizer, w_vectorizer, eval_wrapper, args, use_2d):
    bleu1, bleu4, rouge, cider, bert_score = [], [], [], [], []
    top1, top2, top3, matching = [], [], [], []
    repeat_time = 1
    for _ in range(repeat_time):
        best_top1, best_top2, best_top3, best_matching, \
        best_bleu1, best_bleu4, best_rouge, best_cider, best_bert_score, \
        logger = evaluation_m2t(val_loader,
                                vae, model,
                                logger,
                                tokenizer,
                                w_vectorizer,
                                eval_wrapper=eval_wrapper,
                                instruction=args.prompt,
                                max_new_tokens=40,
                                use_2d=use_2d)
        bleu1.append(best_bleu1); bleu4.append(best_bleu4)
        rouge.append(best_rouge); cider.append(best_cider)
        bert_score.append(best_bert_score)
        top1.append(best_top1); top2.append(best_top2)
        top3.append(best_top3); matching.append(best_matching)
    return {
        'bleu1': np.array(bleu1), 'bleu4': np.array(bleu4),
        'rouge': np.array(rouge), 'cider': np.array(cider),
        'bert_score': np.array(bert_score),
        'top1': np.array(top1), 'top2': np.array(top2),
        'top3': np.array(top3), 'matching': np.array(matching),
        'repeat_time': repeat_time,
    }


def _print_results(tag, r):
    rt = r['repeat_time']
    print(f'\n=== {tag} Results ===')
    for k in ['bleu1', 'bleu4', 'rouge', 'cider', 'bert_score', 'top1', 'top2', 'top3', 'matching']:
        print(f'{k}: {np.mean(r[k]):.3f}')
    return (
        f"[{tag}] "
        f"bleu1. {np.mean(r['bleu1']):.3f}, conf. {np.std(r['bleu1']) * 1.96 / np.sqrt(rt):.3f}, "
        f"bleu4. {np.mean(r['bleu4']):.3f}, conf. {np.std(r['bleu4']) * 1.96 / np.sqrt(rt):.3f}, "
        f"rouge. {np.mean(r['rouge']):.3f}, conf. {np.std(r['rouge']) * 1.96 / np.sqrt(rt):.3f}, "
        f"cider. {np.mean(r['cider']):.3f}, conf. {np.std(r['cider']) * 1.96 / np.sqrt(rt):.3f}, "
        f"bert_score. {np.mean(r['bert_score']):.3f}, conf. {np.std(r['bert_score']) * 1.96 / np.sqrt(rt):.3f}, "
        f"TOP1. {np.mean(r['top1']):.3f}, conf. {np.std(r['top1']) * 1.96 / np.sqrt(rt):.3f}, "
        f"TOP2. {np.mean(r['top2']):.3f}, conf. {np.std(r['top2']) * 1.96 / np.sqrt(rt):.3f}, "
        f"TOP3. {np.mean(r['top3']):.3f}, conf. {np.std(r['top3']) * 1.96 / np.sqrt(rt):.3f}, "
        f"Matching. {np.mean(r['matching']):.3f}, conf. {np.std(r['matching']) * 1.96 / np.sqrt(rt):.3f}"
    )


if __name__ == "__main__":

    parser = option.get_args_parser()

    # set hyperparameters
    parser.add_argument("--model_name", type=str, default="./m2t-ft-from-t5-base/checkpoint-300000/", help="Trained model name or directory")
    parser.add_argument("--prompt", type=str, default="Generate text: ", help="Motion-to-Text Prompt")
    parser.add_argument("--vqvae_2drecon_ckpt", type=str, default=None,
                        help="Path to 2DRecon VQ-VAE checkpoint (.tar). When set, the 2D evaluation "
                             "uses the full 2DRecon VQ-VAE (same as training tokenization) instead of "
                             "the 3D VQ-VAE with swapped encoder.")
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
    from utils.word_vectorizer import WordVectorizer
    w_vectorizer = WordVectorizer('./glove', 'our_vab')

    val_loader = M2T_DATALoader(args.dataname, 'test', 32, w_vectorizer, unit_length=2 ** args.down_t)



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
    logger = utils_model.get_test_logger(args.model_name, 'test_m2t_run.log')
    logger.info(json.dumps(vars(args), indent=4, sort_keys=True))



    # motion aware language model setting
    tokenizer = T5Tokenizer.from_pretrained(args.model_name)
    model = T5ForConditionalGeneration.from_pretrained(args.model_name)

    device = 'cuda' if cuda.is_available() else 'cpu'
    model = model.to(device)



    # ---- 3D evaluation ----
    print('\n========== Evaluating 3D (original encoder) ==========')
    res_3d = _run_eval(val_loader, vae, model, logger, tokenizer, w_vectorizer, eval_wrapper, args, use_2d=False)
    msg_3d = _print_results('3D', res_3d)
    logger.info(msg_3d)

    # ---- Load 2D encoder (or full 2DRecon VQ-VAE) ----
    if args.vqvae_2drecon_ckpt:
        # Use the same VQ-VAE that generated VQVAE_2DRecon training tokens
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
    res_2d = _run_eval(val_loader, vae_2d, model, logger, tokenizer, w_vectorizer, eval_wrapper, args, use_2d=True)
    msg_2d = _print_results(label_2d, res_2d)
    logger.info(msg_2d)

    # ---- Comparison ----
    print(f'\n========== Comparison (3D vs {label_2d}) ==========')
    for k in ['bleu1', 'bleu4', 'rouge', 'cider', 'bert_score', 'top1', 'top2', 'top3', 'matching']:
        print(f'{k}: 3D={np.mean(res_3d[k]):.3f}  {label_2d}={np.mean(res_2d[k]):.3f}  diff={np.mean(res_2d[k]) - np.mean(res_3d[k]):+.3f}')
