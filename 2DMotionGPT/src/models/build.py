from src.models.mgpt_vq import VQVae

def build_vqvae(cfg):
    vqvae = VQVae(
        nfeats=cfg.vq.default.params.nfeats,
        quantizer=cfg.vq.default.params.quantizer,
        code_num=cfg.vq.default.params.code_num,
        code_dim=cfg.vq.default.params.code_dim,
        output_emb_width=cfg.vq.default.params.output_emb_width,
        down_t=cfg.vq.default.params.down_t,
        stride_t=cfg.vq.default.params.stride_t,
        width=cfg.vq.default.params.width,
        depth=cfg.vq.default.params.depth,
        dilation_growth_rate=cfg.vq.default.params.dilation_growth_rate,
        norm=cfg.vq.default.params.norm,
        activation=cfg.vq.default.params.activation
        )
    return vqvae