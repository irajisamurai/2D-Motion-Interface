"""Stage 1: arbitrary video -> ViTPose 2D keypoints JSON (openmmlab env).

Differences from the batch script used to build the dataset (run_vitpose.py):
  * one arbitrary user video instead of a directory of rendered clips,
  * a real person detector (rtmdet-m) instead of ``det_model='whole_image'``,
    which only works for rendered videos where the person fills the frame,
  * a single subject is tracked across frames and missing frames are filled,
    so the JSON always has exactly one instance per frame -- the format
    ``space/features2d.load_vitpose_json`` expects.

Run with the openmmlab environment's python:
    python demo/run_vitpose_single.py --video in.mp4 --out_json kp.json
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

# ViTPose-base (COCO). The config name is resolved through mmpose's metafile,
# which also supplies the weights URL, so no absolute paths are needed.
POSE2D_MODEL = "td-hm_ViTPose-base_8xb64-210e_coco-256x192"

# HumanML3D — and therefore everything downstream — is 20 fps.
TARGET_FPS = 20

# COCO-17 skeleton, for the optional overlay video only.
COCO_EDGES = [(15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12),
              (5, 6), (5, 7), (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2),
              (1, 3), (2, 4), (3, 5), (4, 6)]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, help="input video (any fps / resolution)")
    ap.add_argument("--out_json", required=True, help="where to write the keypoint JSON")
    ap.add_argument("--out_video", default=None, help="optional skeleton overlay video")
    ap.add_argument("--out_meta", default=None, help="optional JSON with extraction stats")
    ap.add_argument("--gpu_id", type=int, default=0, help="index within CUDA_VISIBLE_DEVICES")
    ap.add_argument("--start", type=float, default=0.0, help="trim: start time in seconds")
    ap.add_argument("--duration", type=float, default=None, help="trim: length in seconds")
    ap.add_argument("--fps", type=int, default=TARGET_FPS,
                    help=f"resample the video to this fps (default {TARGET_FPS}; "
                         "the model was trained at 20 fps)")
    ap.add_argument("--no_resample", action="store_true",
                    help="feed the video as-is (only if it is already 20 fps)")
    ap.add_argument("--bbox_thr", type=float, default=0.3, help="person detection threshold")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--keep_tmp", action="store_true", help="keep the resampled video")
    return ap.parse_args()


# -- video preprocessing ---------------------------------------------------

def probe(video):
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        sys.exit(f"[error] cannot open video: {video}")
    info = dict(fps=cap.get(cv2.CAP_PROP_FPS),
                frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    return info


def resample(video, out_path, fps, start, duration):
    """ffmpeg-based fps conversion and trimming. Rotation metadata (iPhone
    portrait videos) is baked in by ffmpeg's decoder, so the keypoints come out
    in the orientation a viewer sees."""
    if shutil.which("ffmpeg") is None:
        sys.exit("[error] ffmpeg not found; install it or pass --no_resample")
    cmd = ["ffmpeg", "-y", "-loglevel", "error"]
    if start:
        cmd += ["-ss", str(start)]
    cmd += ["-i", video]
    if duration:
        cmd += ["-t", str(duration)]
    cmd += ["-vf", f"fps={fps}", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", out_path]
    subprocess.run(cmd, check=True)
    return out_path


# -- single-subject selection ---------------------------------------------

def _box(inst):
    b = inst.get("bbox", [[0, 0, 0, 0]])
    # mmpose serialises the bbox as [[x1, y1, x2, y2]]
    return list(b[0]) if isinstance(b[0], (list, tuple)) else list(b)


def _area(inst):
    x1, y1, x2, y2 = _box(inst)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / ua if ua > 0 else 0.0


def select_subject(frames, bbox_thr):
    """One instance per frame: the biggest confident person, then whichever
    detection overlaps the previous pick. Returns (instances, missing_indices)
    with None where nobody was detected."""
    picked, missing, prev = [], [], None
    for i, frame in enumerate(frames):
        cands = [c for c in frame.get("instances", [])
                 if c.get("bbox_score", 1.0) >= bbox_thr]
        if not cands:
            cands = frame.get("instances", [])
        if not cands:
            picked.append(None)
            missing.append(i)
            prev = None
            continue
        if prev is None:
            best = max(cands, key=lambda c: _area(c) * c.get("bbox_score", 1.0))
        else:
            best = max(cands, key=lambda c: _iou(_box(c), prev))
            if _iou(_box(best), prev) < 0.1:      # subject lost -> re-seed
                best = max(cands, key=lambda c: _area(c) * c.get("bbox_score", 1.0))
        prev = _box(best)
        picked.append(best)
    return picked, missing


def fill_gaps(picked):
    """Linearly interpolate keypoints over frames with no detection so the
    downstream feature extraction never sees a hole. Confidence is
    interpolated too, so filled frames stay visibly low-confidence."""
    valid = [i for i, p in enumerate(picked) if p is not None]
    if not valid:
        sys.exit("[error] no person detected in any frame")
    T = len(picked)
    kp = np.zeros((T, 17, 2), np.float32)
    cf = np.zeros((T, 17), np.float32)
    for i in valid:
        kp[i] = np.asarray(picked[i]["keypoints"], np.float32)
        cf[i] = np.asarray(picked[i]["keypoint_scores"], np.float32)
    if len(valid) < T:
        src = np.asarray(valid)
        tgt = np.arange(T)
        for j in range(17):
            for c in range(2):
                kp[:, j, c] = np.interp(tgt, src, kp[src, j, c])
            cf[:, j] = np.interp(tgt, src, cf[src, j])
    boxes = []
    for i in range(T):
        boxes.append(_box(picked[i]) if picked[i] is not None
                     else [float(kp[i, :, 0].min()), float(kp[i, :, 1].min()),
                           float(kp[i, :, 0].max()), float(kp[i, :, 1].max())])
    scores = [float(picked[i].get("bbox_score", 1.0)) if picked[i] is not None else 0.0
              for i in range(T)]
    return kp, cf, boxes, scores


def to_json(kp, cf, boxes, scores):
    return [{"frame_id": i,
             "instances": [{"keypoints": kp[i].tolist(),
                            "keypoint_scores": cf[i].tolist(),
                            "bbox": [boxes[i]],
                            "bbox_score": scores[i]}]}
            for i in range(len(kp))]


# -- overlay ---------------------------------------------------------------

def draw_overlay(video, kp, cf, out_path, fps, kpt_thr=0.3):
    cap = cv2.VideoCapture(video)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    i = 0
    while i < len(kp):
        ok, frame = cap.read()
        if not ok:
            break
        pts, sc = kp[i], cf[i]
        for a, b in COCO_EDGES:
            if sc[a] >= kpt_thr and sc[b] >= kpt_thr:
                cv2.line(frame, tuple(pts[a].astype(int)), tuple(pts[b].astype(int)),
                         (0, 255, 128), 3, cv2.LINE_AA)
        for j in range(17):
            if sc[j] >= kpt_thr:
                cv2.circle(frame, tuple(pts[j].astype(int)), 4, (0, 96, 255), -1, cv2.LINE_AA)
        writer.write(frame)
        i += 1
    cap.release()
    writer.release()


def main():
    args = parse_args()
    if not os.path.isfile(args.video):
        sys.exit(f"[error] no such video: {args.video}")

    src = probe(args.video)
    print(f"[input] {args.video}  {src['width']}x{src['height']}  "
          f"{src['fps']:.2f} fps  {src['frames']} frames")

    tmpdir = tempfile.mkdtemp(prefix="vitpose_demo_")
    try:
        if args.no_resample:
            video = args.video
        else:
            video = resample(args.video, os.path.join(tmpdir, "input_20fps.mp4"),
                             args.fps, args.start, args.duration)
            got = probe(video)
            print(f"[resample] -> {args.fps} fps, {got['frames']} frames "
                  f"({got['frames'] / args.fps:.1f} s)")

        # importing mmpose is slow, so it happens after the cheap checks
        from mmpose.apis.inferencers import MMPoseInferencer
        inferencer = MMPoseInferencer(pose2d=POSE2D_MODEL,
                                      device=f"cuda:{args.gpu_id}")

        raw_dir = os.path.join(tmpdir, "raw")
        for _ in inferencer(inputs=video, pred_out_dir=raw_dir,
                            batch_size=args.batch_size, bbox_thr=args.bbox_thr):
            pass
        raw_path = os.path.join(raw_dir, Path(video).stem + ".json")
        frames = json.load(open(raw_path))

        picked, missing = select_subject(frames, args.bbox_thr)
        kp, cf, boxes, scores = fill_gaps(picked)

        os.makedirs(os.path.dirname(os.path.abspath(args.out_json)) or ".", exist_ok=True)
        json.dump(to_json(kp, cf, boxes, scores), open(args.out_json, "w"))

        n_multi = sum(1 for f in frames if len(f.get("instances", [])) > 1)
        meta = {"source_video": os.path.abspath(args.video),
                "fps": args.fps, "frames": int(len(kp)),
                "duration_sec": round(len(kp) / args.fps, 2),
                "frames_no_detection": len(missing),
                "frames_multi_person": n_multi,
                "mean_keypoint_score": round(float(cf.mean()), 4),
                "width": src["width"], "height": src["height"]}
        print(f"[pose] {meta['frames']} frames, "
              f"{meta['frames_no_detection']} without detection (interpolated), "
              f"{meta['frames_multi_person']} with >1 person, "
              f"mean kpt score {meta['mean_keypoint_score']}")
        print(f"[pose] wrote {args.out_json}")
        if args.out_meta:
            json.dump(meta, open(args.out_meta, "w"), indent=2)

        if args.out_video:
            draw_overlay(video, kp, cf, args.out_video, args.fps)
            print(f"[pose] wrote {args.out_video}")
    finally:
        if args.keep_tmp:
            print(f"[tmp] kept {tmpdir}")
        else:
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
