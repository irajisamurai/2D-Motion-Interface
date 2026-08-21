"""Video -> caption, end to end.

    python demo/demo.py --video my_clip.mp4

Two conda environments are involved because mmpose (torch 1.9) and MotionGPT
(torch 2.9) cannot coexist, so this driver runs each stage in its own
interpreter:

    stage 1  openmlab env : video -> 20 fps -> ViTPose -> keypoints.json
    stage 2  mgpt_2d  env : keypoints.json -> A_real -> E_2D -> caption

Interpreters are located automatically; override with the POSE_PYTHON /
CAPTION_PYTHON environment variables or --pose_python / --caption_python.

This driver only uses the standard library, so any python3 can run it.
"""

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
POSE_ENV = "openmmlab"
CAPTION_ENV = "mgpt_2d"


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, help="input video (any fps / resolution)")
    ap.add_argument("--out_dir", default=None,
                    help="output directory (default: demo/outputs/<video name>)")
    ap.add_argument("--gpu_id", type=int, default=0)
    ap.add_argument("--start", type=float, default=0.0, help="trim: start time in seconds")
    ap.add_argument("--duration", type=float, default=None, help="trim: length in seconds")
    ap.add_argument("--mode", default="both", choices=["both", "adapter", "raw"])
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                    help="device for the captioner (pose estimation always uses the GPU)")
    ap.add_argument("--no_overlay", action="store_true", help="skip the skeleton video")
    ap.add_argument("--skip_pose", action="store_true",
                    help="reuse the keypoint JSON already in --out_dir")
    ap.add_argument("--pose_python", default=os.environ.get("POSE_PYTHON"))
    ap.add_argument("--caption_python", default=os.environ.get("CAPTION_PYTHON"))
    return ap.parse_args()


def find_python(env_name, override):
    """Locate an environment's interpreter without needing conda on PATH."""
    if override:
        if not os.path.isfile(override):
            sys.exit(f"[error] not an interpreter: {override}")
        return override
    cands = []
    conda = os.environ.get("CONDA_EXE") or shutil.which("conda")
    if conda:
        base = os.path.dirname(os.path.dirname(conda))
        cands.append(os.path.join(base, "envs", env_name, "bin", "python"))
    cands += sorted(glob.glob(os.path.expanduser(
        f"~/.pyenv/versions/*/envs/{env_name}/bin/python")))
    cands += sorted(glob.glob(os.path.expanduser(
        f"~/*conda*/envs/{env_name}/bin/python")))
    for c in cands:
        if os.path.isfile(c):
            return c
    sys.exit(f"[error] could not find the '{env_name}' environment. "
             f"Pass its interpreter explicitly, e.g.\n"
             f"    --{'pose' if env_name == POSE_ENV else 'caption'}_python "
             f"$(conda run -n {env_name} which python)")


def run(cmd):
    print(f"\n$ {' '.join(cmd)}\n", flush=True)
    if subprocess.run(cmd).returncode != 0:
        sys.exit(f"[error] stage failed: {cmd[1]}")


def main():
    args = parse_args()
    if not os.path.isfile(args.video):
        sys.exit(f"[error] no such video: {args.video}")

    stem = os.path.splitext(os.path.basename(args.video))[0]
    out_dir = args.out_dir or os.path.join(HERE, "outputs", stem)
    os.makedirs(out_dir, exist_ok=True)
    kp_json = os.path.join(out_dir, "keypoints.json")
    result_json = os.path.join(out_dir, "caption.json")

    if not args.skip_pose:
        cmd = [find_python(POSE_ENV, args.pose_python),
               os.path.join(HERE, "run_vitpose_single.py"),
               "--video", os.path.abspath(args.video),
               "--out_json", kp_json,
               "--out_meta", os.path.join(out_dir, "pose_meta.json"),
               "--gpu_id", str(args.gpu_id),
               "--start", str(args.start)]
        if args.duration:
            cmd += ["--duration", str(args.duration)]
        if not args.no_overlay:
            cmd += ["--out_video", os.path.join(out_dir, "overlay.mp4")]
        run(cmd)
    elif not os.path.isfile(kp_json):
        sys.exit(f"[error] --skip_pose given but {kp_json} does not exist")

    run([find_python(CAPTION_ENV, args.caption_python),
         os.path.join(HERE, "caption_from_json.py"),
         "--json", kp_json,
         "--out_json", result_json,
         "--device", args.device,
         "--mode", args.mode])

    result = json.load(open(result_json))
    print("=" * 72)
    print(f"  video   {args.video}")
    for key, label in (("with_adapter", "2D + A_real (ours)"),
                       ("without_adapter", "2D, no adapter")):
        if key in result:
            print(f"  {label:<22}  {result[key]}")
    print("=" * 72)
    print(f"  outputs in {out_dir}")


if __name__ == "__main__":
    main()
