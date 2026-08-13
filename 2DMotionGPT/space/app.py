"""Hugging Face Space for the 2D Motion Interface.

Pick one of four real monocular videos; the ViTPose 2D keypoints shipped with the
clip are fed through A_real -> 2D encoder -> MotionGPT and captioned live. The
adapter-less baseline is captioned too, so the adapter's contribution is visible
side by side.

No pose estimation runs here: the keypoints are the ones released with the
real-world dataset, so the Space is CPU-only and needs no ViTPose install.

Env:
  BUNDLE_DIR   local weight bundle (default: ./bundle, then ./space/bundle)
  BUNDLE_REPO  HF model repo to pull the bundle from when BUNDLE_DIR is absent
"""

import html
import os

import gradio as gr
from omegaconf import OmegaConf
from os.path import join as pjoin

from captioner import MotionCaptioner

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = pjoin(HERE, "assets")


# A free account cannot host a Gradio Space on cpu-basic, so this Space runs on
# ZeroGPU — which refuses to start unless at least one @spaces.GPU function exists.
# Locally the `spaces` package is absent and everything falls back to CPU.
try:
    import spaces
    GPU = spaces.GPU
    DEVICE = "cuda"
except ImportError:                                    # local / non-ZeroGPU
    def GPU(fn=None, **kwargs):
        return fn if fn is not None else (lambda f: f)
    DEVICE = "cpu"


def _resolve_bundle():
    for cand in (os.environ.get("BUNDLE_DIR"), pjoin(HERE, "bundle"),
                 pjoin(HERE, "space", "bundle")):
        if cand and os.path.isdir(cand):
            return cand
    repo = os.environ.get("BUNDLE_REPO")
    if not repo:
        raise RuntimeError("No weight bundle found. Set BUNDLE_DIR or BUNDLE_REPO.")
    from huggingface_hub import snapshot_download
    return snapshot_download(repo_id=repo)


CLIPS = {c.id: c for c in OmegaConf.load(pjoin(ASSETS, "clips.yaml")).clips}
CHOICES = [(f"{c.label}  ({c.frames_raw} frames)", cid) for cid, c in CLIPS.items()]

CAPTIONER = MotionCaptioner(_resolve_bundle(), device=DEVICE)


CSS = """
.result-card {
    border: 1px solid var(--border-color-primary);
    border-radius: var(--block-radius);
    background: var(--block-background-fill);
    padding: 24px 28px;
    min-height: 148px;
    display: flex;
    flex-direction: column;
    justify-content: center;
}
.result-label {
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--body-text-color-subdued);
    margin-bottom: 12px;
}
.result-caption {
    font-size: clamp(20px, 2.4vw, 30px);
    line-height: 1.35;
    font-weight: 500;
    color: var(--body-text-color);
    margin: 0;
    text-wrap: balance;
}
.result-caption.placeholder {
    font-weight: 400;
    color: var(--body-text-color-subdued);
}
.result-meta {
    margin-top: 18px;
    padding-top: 14px;
    border-top: 1px solid var(--border-color-primary);
    font-size: 12px;
    color: var(--body-text-color-subdued);
    font-variant-numeric: tabular-nums;
}
"""

PLACEHOLDER = """<div class="result-card">
  <p class="result-caption placeholder">Select a clip and generate its caption.</p>
</div>"""


def show_video(clip_id):
    return pjoin(ASSETS, CLIPS[clip_id].video), PLACEHOLDER


@GPU(duration=30)
def run(clip_id):
    clip = CLIPS[clip_id]
    out = CAPTIONER.caption_json(pjoin(ASSETS, clip.keypoints), use_adapter=True)
    meta = (f"{clip.frames_raw} frames estimated &middot; "
            f"{out['frames_used']} used &middot; "
            f"{out['n_tokens']} motion tokens &middot; greedy decoding")
    return (f'<div class="result-card">\n'
            f'  <div class="result-label">2D Input + A_real</div>\n'
            f'  <p class="result-caption">{html.escape(out["caption"])}</p>\n'
            f'  <div class="result-meta">{meta}</div>\n'
            f'</div>')


with gr.Blocks(title="2D Motion Interface", css=CSS) as demo:
    gr.Markdown(
        "# 🕺 A Plug-and-Play 2D Motion Interface\n"
        "Captioning real monocular video through **2D keypoints only** — no 3D pose "
        "estimation anywhere in the pipeline.\n\n"
        "The overlay shows the ViTPose 2D keypoints that are actually fed to the model. "
        "`A_real` is the 0.3M-parameter real-video adapter; the pretrained motion-language "
        "model is untouched."
    )

    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            choice = gr.Radio(choices=CHOICES, value=CHOICES[0][1], label="Clip")
            button = gr.Button("Generate caption", variant="primary", size="lg")
        with gr.Column(scale=1):
            video = gr.Video(value=pjoin(ASSETS, CLIPS[CHOICES[0][1]].video),
                             label="ViTPose keypoints overlaid", autoplay=True, loop=True)

    output = gr.HTML(value=PLACEHOLDER)

    # api_name=False keeps this internal. Gradio 6 will want api_visibility="private",
    # which 5.50 does not accept yet.
    choice.change(show_video, inputs=choice, outputs=[video, output], api_name=False)
    button.click(run, inputs=choice, outputs=output, api_name="caption")

if __name__ == "__main__":
    demo.launch()
