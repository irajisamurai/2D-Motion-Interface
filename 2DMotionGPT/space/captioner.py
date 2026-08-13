"""Inference-only motion captioner: 2D keypoints -> English caption.

This is the whole pipeline the Space needs, with no HumanML3D, no glove,
no pytorch-lightning, no metric packages, and no ViTPose:

    COCO-17 keypoints + confidence
      -> COCO-13
      -> 81-dim features (with conf)  --A_real-->  zero-pad to 263  -.
      -> 68-dim features (no conf)    ------------ zero-pad to 263 --+-> VQ-VAE.encode
                                                                      -> MotionGPT LM (m2t)
                                                                      -> caption

Deterministic: the full clip is used from frame 0 (only truncated to a multiple
of unit_length), and generation is greedy. The caption therefore always matches
the video shown next to it.
"""

import numpy as np
import torch
from omegaconf import OmegaConf
from os.path import join as pjoin
from safetensors.torch import load_file
from transformers import T5Config, T5ForConditionalGeneration

import features2d as F2D
from adapters import build_adapter
from mgpt.mgpt_lm import MLM
from mgpt.mgpt_vq import VQVae

NFEATS_LM = 263


class _T5FromConfigOnly:
    """MLM.__init__ calls T5ForConditionalGeneration.from_pretrained, which would
    fetch ~990MB of flan-t5-base weights that lm.safetensors immediately
    overwrites. Inside this context the architecture is built from config.json
    alone, so the bundle needs no pretrained weights at all.
    """

    def __enter__(self):
        # from_pretrained lives on PreTrainedModel, not on the subclass, so the
        # override is removed again on exit rather than restored.
        self._had_own = "from_pretrained" in T5ForConditionalGeneration.__dict__
        self._orig = T5ForConditionalGeneration.__dict__.get("from_pretrained")

        def _from_config(path, *args, **kwargs):
            return T5ForConditionalGeneration(T5Config.from_pretrained(path))

        T5ForConditionalGeneration.from_pretrained = _from_config
        return self

    def __exit__(self, *exc):
        if self._had_own:
            T5ForConditionalGeneration.from_pretrained = self._orig
        else:
            del T5ForConditionalGeneration.from_pretrained
        return False


class MotionCaptioner:
    def __init__(self, bundle_dir, device="cpu"):
        self.device = torch.device(device)
        cfg = OmegaConf.load(pjoin(bundle_dir, "model_config.yaml"))
        self.cfg = cfg
        self.unit_length = cfg.unit_length
        self.max_motion_length = cfg.max_motion_length

        stats = np.load(pjoin(bundle_dir, "stats.npz"))
        self.mean_2d, self.std_2d = stats["mean_2d"], stats["std_2d"]
        self.mean_est, self.std_est = stats["mean_est"], stats["std_est"]

        # VQ-VAE (encoder + quantizer only; decoder is never called)
        vq = dict(cfg.vq)
        vq.pop("ablation", None)
        self.vqvae = VQVae(**vq).to(self.device)
        missing, unexpected = self.vqvae.load_state_dict(
            load_file(pjoin(bundle_dir, "vqvae.safetensors")), strict=False)
        assert not unexpected, f"unexpected VQ-VAE keys: {unexpected[:5]}"
        assert all("decoder" in k for k in missing), \
            f"VQ-VAE weights missing outside the decoder: {[k for k in missing if 'decoder' not in k][:5]}"

        # Language model
        with _T5FromConfigOnly():
            self.lm = MLM(
                model_path=pjoin(bundle_dir, "flan-t5-base"),
                model_type="t5",
                stage="test",
                motion_codebook_size=cfg.motion_codebook_size,
            ).to(self.device)
        tied = ["language_model.encoder.embed_tokens.weight",
                "language_model.decoder.embed_tokens.weight"]
        missing, unexpected = self.lm.load_state_dict(
            load_file(pjoin(bundle_dir, "lm.safetensors")), strict=False)
        assert not unexpected, f"unexpected LM keys: {unexpected[:5]}"
        assert set(missing) <= set(tied), f"LM weights missing: {sorted(set(missing) - set(tied))[:5]}"
        shared = self.lm.language_model.shared.weight
        for name in ("encoder", "decoder"):
            w = getattr(self.lm.language_model, name).embed_tokens.weight
            assert w.data_ptr() == shared.data_ptr(), \
                f"{name}.embed_tokens is not tied to shared - the dropped copy was needed"

        # Adapter
        self.adapter = build_adapter(cfg.adapter_type, dim=cfg.adapter_dim,
                                     hidden=cfg.adapter_hidden).to(self.device)
        self.adapter.load_state_dict(load_file(pjoin(bundle_dir, "adapter.safetensors")))

        for m in (self.vqvae, self.lm, self.adapter):
            m.eval()

    # -- length handling ---------------------------------------------------

    def clip_length(self, n_frames):
        """Deterministic: keep frame 0 onwards, truncate to a multiple of
        unit_length, cap at max_motion_length."""
        n = min(n_frames, self.max_motion_length)
        return max((n // self.unit_length) * self.unit_length, self.unit_length)

    # -- inference ---------------------------------------------------------

    def _generate(self, feats_263):
        with torch.no_grad():
            tokens, _ = self.vqvae.encode(feats_263)
            out = self.lm.generate_conditional(
                motion_tokens=[tokens[0]],
                lengths=[tokens.shape[1]],
                task="m2t",
                stage="test",
            )
        return out[0], int(tokens.shape[1])

    def _pad(self, x):
        pad = torch.zeros(x.shape[0], x.shape[1], NFEATS_LM - x.shape[2], device=self.device)
        return torch.cat([x, pad], dim=-1)

    def caption(self, kp17, conf17, use_adapter=True):
        """kp17: (T,17,2) pixel coordinates, conf17: (T,17) scores."""
        kp, cf = F2D.to_coco13(np.asarray(kp17, np.float32), np.asarray(conf17, np.float32))
        n = self.clip_length(kp.shape[0])
        kp, cf = kp[:n], cf[:n]

        if use_adapter:
            feat = (F2D.feature_81(kp, cf) - self.mean_est) / self.std_est
            x = torch.from_numpy(feat).float().unsqueeze(0).to(self.device)
            x = self.adapter(x)
        else:
            feat = (F2D.feature_68(kp) - self.mean_2d) / self.std_2d
            x = torch.from_numpy(feat).float().unsqueeze(0).to(self.device)

        text, n_tokens = self._generate(self._pad(x))
        return {"caption": text, "frames_used": n, "n_tokens": n_tokens}

    def caption_json(self, json_path, use_adapter=True):
        kp, cf = F2D.load_vitpose_json(json_path)
        return self.caption(kp, cf, use_adapter=use_adapter)
