"""
Inference: extract a pose from an image, score it, and visualise the result.

Usage:
    python score_pose.py path/to/image.png
    python score_pose.py path/to/image.png --checkpoint pose_scorer.pt
    python score_pose.py path/to/image.png --no-rotate
"""

from __future__ import annotations

import argparse
import sys

import cv2
import mediapipe as mp
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np
import torch

from train_bradley_terry import normalize_pose, PoseScorer

_mp_pose    = mp.solutions.pose
_mp_drawing = mp.solutions.drawing_utils

# Catppuccin Mocha palette
C_BG      = '#1e1e2e'
C_SURFACE = '#181825'
C_TEXT    = '#cdd6f4'
C_SUBTEXT = '#6c7086'
C_BLUE    = '#89b4fa'
C_GREEN   = '#a6e3a1'
C_YELLOW  = '#f9e2af'
C_RED     = '#f38ba8'
C_SKEL_BG = (15, 15, 30)
C_JOINT   = (220, 80, 80)
C_BONE    = (100, 160, 240)


# ──────────────────────────────────────── helpers

def _draw_skeleton(img_rgb: np.ndarray, landmarks) -> np.ndarray:
    h, w = img_rgb.shape[:2]
    canvas = np.full((h, w, 3), C_SKEL_BG, dtype=np.uint8)
    if landmarks:
        _mp_drawing.draw_landmarks(
            canvas, landmarks, _mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=_mp_drawing.DrawingSpec(
                color=C_JOINT, thickness=4, circle_radius=4),
            connection_drawing_spec=_mp_drawing.DrawingSpec(
                color=C_BONE, thickness=2))
    return canvas


def _draw_3d_skeleton(ax: plt.Axes,
                      world_lms,
                      title: str = "3D Skeleton") -> None:
    """Plot the 3D world landmark skeleton on a matplotlib 3D axis."""
    xs = [lm.x  for lm in world_lms]
    ys = [-lm.y for lm in world_lms]   # flip Y up
    zs = [lm.z  for lm in world_lms]

    ax.scatter(xs, zs, ys, c='#f38ba8', s=30, zorder=5)
    for start, end in _mp_pose.POSE_CONNECTIONS:
        ax.plot([xs[start], xs[end]],
                [zs[start], zs[end]],
                [ys[start], ys[end]],
                color='#89b4fa', linewidth=1.5)

    ax.set_xlabel('X', color=C_SUBTEXT, fontsize=8)
    ax.set_ylabel('Z', color=C_SUBTEXT, fontsize=8)
    ax.set_zlabel('Y', color=C_SUBTEXT, fontsize=8)
    ax.set_title(title, color=C_TEXT, fontsize=11, pad=6)
    ax.set_facecolor(C_SURFACE)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor(C_SUBTEXT)
    ax.tick_params(colors=C_SUBTEXT, labelsize=7)
    ax.grid(True, color=C_SUBTEXT, alpha=0.3)


def _score_gauge(ax: plt.Axes, quality_pct: float, raw_score: float) -> None:
    """Draw a horizontal score bar with colour coding."""
    ax.set_facecolor(C_SURFACE)

    if quality_pct >= 65:
        color = C_GREEN
    elif quality_pct >= 40:
        color = C_YELLOW
    else:
        color = C_RED

    # Background track
    ax.barh([0], [100], color='#313244', height=0.55, zorder=1)
    # Filled portion
    ax.barh([0], [quality_pct], color=color, height=0.55, zorder=2)

    ax.set_xlim(0, 100)
    ax.set_ylim(-1, 1)
    ax.set_yticks([])
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(['0', '25', '50', '75', '100'], color=C_SUBTEXT, fontsize=9)
    ax.set_xlabel("Quality  (%)", color=C_TEXT, fontsize=10)
    ax.set_title("Pose Quality Score", color=C_TEXT, fontsize=11, pad=8)
    for spine in ax.spines.values():
        spine.set_edgecolor('#45475a')

    ax.text(quality_pct / 2, 0, f"{quality_pct:.1f}%",
            ha='center', va='center', fontsize=22, fontweight='bold',
            color=C_SURFACE if quality_pct > 10 else C_TEXT, zorder=3)
    ax.text(50, -0.7, f"raw logit: {raw_score:+.3f}",
            ha='center', va='center', fontsize=9, color=C_SUBTEXT)

    # Legend patches
    patches = [
        mpatches.Patch(color=C_GREEN,  label='Good  (≥65%)'),
        mpatches.Patch(color=C_YELLOW, label='Fair  (40–65%)'),
        mpatches.Patch(color=C_RED,    label='Poor  (<40%)'),
    ]
    ax.legend(handles=patches, loc='lower center', ncol=3,
              fontsize=8, framealpha=0,
              labelcolor=C_SUBTEXT, bbox_to_anchor=(0.5, -0.35))


# ──────────────────────────────────────── main

def score_image(image_path: str, checkpoint: str, rotate: bool) -> None:
    # ── load model ────────────────────────────────────────────────────
    ckpt = torch.load(checkpoint, map_location='cpu', weights_only=False)
    hidden    = ckpt.get('hidden',    [256, 128, 64])
    input_dim = ckpt.get('input_dim', 132)
    # If the checkpoint recorded a rotate setting, prefer it unless the user
    # explicitly passed --no-rotate on the command line.
    ckpt_rotate = ckpt.get('rotate', rotate)

    model = PoseScorer(input_dim=input_dim, hidden=hidden)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    print(f"Checkpoint : {checkpoint}  "
          f"(epoch {ckpt.get('epoch','?')}, "
          f"val acc {ckpt.get('val_acc', 0):.1%})")
    print(f"Rotation normalisation : {ckpt_rotate}")

    # ── run mediapipe ─────────────────────────────────────────────────
    pose = _mp_pose.Pose(static_image_mode=True, model_complexity=2,
                         enable_segmentation=False, min_detection_confidence=0.5)

    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        sys.exit(f"Cannot read image: {image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)
    pose.close()

    skeleton_2d = _draw_skeleton(img_rgb, results.pose_landmarks)

    if results.pose_world_landmarks is None:
        print("WARNING: No pose detected — cannot compute score.")
        fig, axes = plt.subplots(1, 2, figsize=(10, 5),
                                 facecolor=C_BG)
        for ax, im, ttl in zip(axes,
                                [img_rgb, skeleton_2d],
                                ['Original', 'No pose detected']):
            ax.imshow(im); ax.set_title(ttl, color=C_TEXT); ax.axis('off')
        plt.tight_layout(); plt.show()
        return

    # ── score ──────────────────────────────────────────────────────────
    lms = results.pose_world_landmarks.landmark
    raw_arr = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in lms],
                       dtype=np.float32)
    feat = normalize_pose(raw_arr, rotate=ckpt_rotate)
    feat_t = torch.from_numpy(feat).unsqueeze(0)

    with torch.no_grad():
        raw_score = model(feat_t).item()

    quality_pct = float(torch.sigmoid(torch.tensor(raw_score)).item() * 100)

    print(f"Raw logit  : {raw_score:+.4f}")
    print(f"Quality    : {quality_pct:.1f}%")

    # ── figure ────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(15, 7), facecolor=C_BG)
    # layout: [original | skeleton 2D | 3D skeleton ] on top row
    #         [          score bar spans bottom       ]
    gs = gridspec.GridSpec(2, 3, figure=fig,
                            height_ratios=[3, 1],
                            hspace=0.35, wspace=0.12)

    ax_orig  = fig.add_subplot(gs[0, 0])
    ax_skel2 = fig.add_subplot(gs[0, 1])
    ax_3d    = fig.add_subplot(gs[0, 2], projection='3d')
    ax_bar   = fig.add_subplot(gs[1, :])

    # Original
    ax_orig.imshow(img_rgb)
    ax_orig.set_title("Original Image", color=C_TEXT, fontsize=11, pad=6)
    ax_orig.axis('off')

    # 2D skeleton
    ax_skel2.imshow(skeleton_2d)
    ax_skel2.set_title("MediaPipe 2D Skeleton", color=C_TEXT, fontsize=11, pad=6)
    ax_skel2.axis('off')

    # 3D skeleton
    _draw_3d_skeleton(ax_3d, lms, title="3D World Pose")

    # Score bar
    _score_gauge(ax_bar, quality_pct, raw_score)

    fig.suptitle(f"Yoga Pose Quality Evaluation  —  {Path(image_path).name}",
                 color=C_TEXT, fontsize=13, fontweight='bold', y=1.01)

    plt.show()


# ──────────────────────────────────────── entry point

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Score a yoga pose image")
    p.add_argument('image',        help='Path to the yoga pose image')
    p.add_argument('--checkpoint', default='pose_scorer.pt')
    p.add_argument('--no-rotate',  action='store_true',
                   help='Disable rotation normalisation '
                        '(overridden by checkpoint setting if present)')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    from pathlib import Path
    score_image(args.image, args.checkpoint, rotate=not args.no_rotate)
