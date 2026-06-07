"""
evaluate_dae.py
---------------
Answers the core research question:
  Does filtering MediaPipe landmarks through the DAE improve the
  Bradley-Terry scorer's accuracy at ranking yoga pose quality pairs?

Metric: same as Mustapha used during training —
  accuracy = % of (winner, loser) pairs where score_winner > score_loser

Produces:
  - Console report with overall and per-class accuracy
  - dae_evaluation.png  — two matplotlib figures saved to disk
      Fig 1: overall accuracy bar + pair breakdown
      Fig 2: per-class accuracy comparison (top 20 classes by pair count)

Usage:
    python evaluate_dae.py
    python evaluate_dae.py --prefs preferences_merged.jsonl
    python evaluate_dae.py --scorer pose_scorer.pt --dae dae.pt
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

import matplotlib
matplotlib.use('Agg')   #no display needed; saves to PNG
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import torch.nn as nn

from train_bradley_terry import normalize_pose, PoseScorer
from train_dae import PoseDAE

COLLAPSE_THR = 0.05   #AE output std below this is treated as a collapsed prediction


# ──────────────────────────────────────── DAE helpers

def load_dae(path: str, device: torch.device) -> PoseDAE:
    ckpt  = torch.load(path, map_location=device, weights_only=False)
    model = PoseDAE(
        input_dim=ckpt.get('input_dim', 99),
        latent_dim=ckpt.get('latent_dim', 64),
    ).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f'DAE loaded       epoch={ckpt.get("epoch","?")}  '
          f'val_loss={ckpt.get("val_loss", float("nan")):.5f}  '
          f'latent={ckpt.get("latent_dim", 64)}')
    return model


def apply_dae(world_34: np.ndarray,
              dae: PoseDAE,
              device: torch.device) -> tuple[np.ndarray, float]:
    """
    Filter a (33,4) world-landmark array through the DAE.

    The DAE was trained on 99-dim xyz-only vectors in the space:
      center at mean xyz, scale by max distance from center.
    We apply it as a denoising filter on the xyz channel only,
    then invert the normalization to return to world-meter space.
    The visibility column (col 3) is preserved unchanged because
    the DAE does not model detection confidence.

    Returns the filtered (33,4) array and the AE output std
    (used to detect collapsed predictions).
    """
    xyz    = world_34[:, :3].copy()
    vis    = world_34[:, 3:4]

    #your normalization (same as in train_dae.py _xyz_from_world)
    center = xyz.mean(axis=0)
    xyz_c  = xyz - center
    scale  = np.linalg.norm(xyz_c, axis=1).max() + 1e-8
    vec_99 = (xyz_c / scale).flatten().astype(np.float32)

    with torch.no_grad():
        vec_ae = dae(
            torch.from_numpy(vec_99).unsqueeze(0).to(device)
        ).cpu().squeeze(0).numpy()

    ae_std = float(vec_ae.std())

    if ae_std < COLLAPSE_THR:
        vec_ae = vec_99.copy()   #fallback to raw if AE collapsed

    #inverse normalization: back to world meter space
    xyz_ae = vec_ae.reshape(33, 3) * scale + center
    return np.concatenate([xyz_ae, vis], axis=1).astype(np.float32), ae_std


# ──────────────────────────────────────── scoring

def bt_logit(world_34: np.ndarray,
             scorer: PoseScorer,
             rotate: bool,
             device: torch.device) -> float:
    feat = normalize_pose(world_34, rotate=rotate)
    t    = torch.from_numpy(feat).unsqueeze(0).to(device)
    with torch.no_grad():
        return scorer(t).item()


# ──────────────────────────────────────── evaluation loop

def evaluate(args: argparse.Namespace) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device : {device}\n')

    #load Mustapha's scorer
    ckpt      = torch.load(args.scorer, map_location=device, weights_only=False)
    bt_rotate = ckpt.get('rotate', True)
    scorer    = PoseScorer(
        input_dim=ckpt.get('input_dim', 132),
        hidden=ckpt.get('hidden', [256, 128, 64]),
    ).to(device)
    scorer.load_state_dict(ckpt['model_state'])
    scorer.eval()
    original_val_acc = ckpt.get('val_acc', None)
    print(f'BT scorer loaded  epoch={ckpt.get("epoch","?")}  '
          f'original_val_acc={original_val_acc:.1%}  rotate={bt_rotate}')

    #load DAE
    dae = load_dae(args.dae, device)
    print()

    #load landmark cache with basename fallback for cross-machine paths
    with open(args.cache, 'rb') as f:
        cache = pickle.load(f)
    basename_map = {os.path.basename(k): v for k, v in cache.items()}

    def get_lm(path: str):
        v = cache.get(path)
        if v is not None:
            return v
        return basename_map.get(os.path.basename(path))

    #load preference pairs
    pairs: list[tuple[str, str, str]] = []
    with open(args.prefs) as f:
        for line in f:
            line = line.strip()
            if not line.startswith('{'):
                continue
            try:
                r = json.loads(line)
                pairs.append((r['winner'], r['loser'], r.get('class', 'unknown')))
            except json.JSONDecodeError:
                pass
    print(f'Preference pairs: {len(pairs)}\n')

    #counters
    n_total = n_skipped = n_collapsed = 0
    raw_correct = ae_correct = 0
    both_ok = neither_ok = only_raw_ok = only_ae_ok = 0
    class_stats: dict[str, dict[str, int]] = {}

    for winner_path, loser_path, cls in pairs:
        w_lm = get_lm(winner_path)
        l_lm = get_lm(loser_path)
        if w_lm is None or l_lm is None:
            n_skipped += 1
            continue

        n_total += 1

        #raw scores
        w_raw_logit = bt_logit(w_lm, scorer, bt_rotate, device)
        l_raw_logit = bt_logit(l_lm, scorer, bt_rotate, device)
        raw_ok = w_raw_logit > l_raw_logit

        #AE-filtered scores
        w_ae_lm, w_std = apply_dae(w_lm, dae, device)
        l_ae_lm, l_std = apply_dae(l_lm, dae, device)
        w_ae_logit = bt_logit(w_ae_lm, scorer, bt_rotate, device)
        l_ae_logit = bt_logit(l_ae_lm, scorer, bt_rotate, device)
        ae_ok = w_ae_logit > l_ae_logit

        if w_std < COLLAPSE_THR or l_std < COLLAPSE_THR:
            n_collapsed += 1

        if raw_ok:
            raw_correct += 1
        if ae_ok:
            ae_correct += 1

        if raw_ok and ae_ok:
            both_ok += 1
        elif not raw_ok and not ae_ok:
            neither_ok += 1
        elif raw_ok and not ae_ok:
            only_raw_ok += 1
        else:
            only_ae_ok += 1

        if cls not in class_stats:
            class_stats[cls] = {'n': 0, 'raw': 0, 'ae': 0}
        class_stats[cls]['n']   += 1
        class_stats[cls]['raw'] += int(raw_ok)
        class_stats[cls]['ae']  += int(ae_ok)

    if n_total == 0:
        sys.exit('No pairs could be evaluated — check that paths in the '
                 'preference file match those in the landmark cache.')

    raw_acc = raw_correct / n_total
    ae_acc  = ae_correct  / n_total
    delta   = ae_acc - raw_acc

    #console report
    print('=' * 58)
    print(f'Pairs evaluated        : {n_total}  (skipped: {n_skipped})')
    print(f'Collapsed (fallback)   : {n_collapsed}')
    print('─' * 58)
    if original_val_acc is not None:
        print(f'Original val_acc       : {original_val_acc:.1%}  '
              f'(Mustapha checkpoint, held-out 15%)')
    print(f'Accuracy RAW           : {raw_acc:.1%}  ({raw_correct}/{n_total})')
    print(f'Accuracy AE-filtered   : {ae_acc:.1%}  ({ae_correct}/{n_total})')
    print(f'Delta (AE − Raw)       : {delta:+.1%}')
    print('─' * 58)
    print(f'Both correct           : {both_ok}  ({both_ok/n_total:.1%})')
    print(f'Only RAW correct       : {only_raw_ok}  ({only_raw_ok/n_total:.1%})')
    print(f'Only AE correct        : {only_ae_ok}  ({only_ae_ok/n_total:.1%})')
    print(f'Neither correct        : {neither_ok}  ({neither_ok/n_total:.1%})')
    print('=' * 58)

    #interpret the result
    if delta > 0.01:
        verdict = f'DAE IMPROVES accuracy by {delta:+.1%}'
    elif delta < -0.01:
        verdict = f'DAE HURTS accuracy by {delta:+.1%}'
    else:
        verdict = f'DAE has NEGLIGIBLE effect ({delta:+.1%})'
    print(f'\nVerdict: {verdict}\n')

    # ── visualisation ─────────────────────────────────────────────────

    #Catppuccin Mocha palette (matches Mustapha's score_pose.py)
    C_BG      = '#1e1e2e'
    C_SURFACE = '#181825'
    C_TEXT    = '#cdd6f4'
    C_SUB     = '#6c7086'
    C_GREEN   = '#a6e3a1'
    C_RED     = '#f38ba8'
    C_YELLOW  = '#f9e2af'
    C_BLUE    = '#89b4fa'

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=C_BG)
    fig.suptitle('DAE Impact on Bradley-Terry Pose Quality Scorer',
                 color=C_TEXT, fontsize=14, fontweight='bold', y=1.01)

    #Figure 1 left: overall accuracy comparison
    ax = axes[0]
    ax.set_facecolor(C_SURFACE)
    bars_x   = ['Raw\nMediaPipe', 'AE\nFiltered']
    bars_y   = [raw_acc * 100, ae_acc * 100]
    bar_cols = [C_RED, C_GREEN] if delta >= 0 else [C_GREEN, C_RED]
    b = ax.bar(bars_x, bars_y, color=bar_cols, width=0.4, zorder=2)
    for rect, v in zip(b, bars_y):
        ax.text(rect.get_x() + rect.get_width() / 2, v + 0.8,
                f'{v:.1f}%', ha='center', va='bottom',
                color=C_TEXT, fontsize=13, fontweight='bold')
    if original_val_acc is not None:
        ax.axhline(original_val_acc * 100, color=C_YELLOW,
                   linestyle='--', linewidth=1.5, zorder=3)
        ax.text(1.45, original_val_acc * 100 + 0.5,
                f'Preference model val_acc\n{original_val_acc:.1%}',
                color=C_YELLOW, fontsize=8, va='bottom')
    ax.set_ylim(0, 110)
    ax.set_ylabel('Accuracy (%)', color=C_TEXT)
    ax.set_title('Overall Pair-Ranking Accuracy', color=C_TEXT, pad=10)
    ax.tick_params(colors=C_SUB)
    for sp in ax.spines.values():
        sp.set_edgecolor('#45475a')
    ax.grid(axis='y', color=C_SUB, alpha=0.3, zorder=1)
    delta_label = f'DAE effect: {delta:+.1%}'
    ax.text(0.5, 5, delta_label, ha='center', color=C_TEXT,
            fontsize=11, fontstyle='italic')

    #Figure 1 right: pair breakdown
    ax2 = axes[1]
    ax2.set_facecolor(C_SURFACE)
    breakdown_labels = ['Both\ncorrect', 'Only RAW\ncorrect',
                        'Only AE\ncorrect', 'Neither\ncorrect']
    breakdown_vals   = [both_ok, only_raw_ok, only_ae_ok, neither_ok]
    breakdown_cols   = [C_GREEN, C_RED, C_BLUE, C_SUB]
    b2 = ax2.bar(breakdown_labels, breakdown_vals,
                 color=breakdown_cols, width=0.5, zorder=2)
    for rect, v in zip(b2, breakdown_vals):
        ax2.text(rect.get_x() + rect.get_width() / 2, v + 5,
                 f'{v}\n({v/n_total:.1%})', ha='center', va='bottom',
                 color=C_TEXT, fontsize=9)
    ax2.set_ylabel('Number of pairs', color=C_TEXT)
    ax2.set_title('Pair-Level Breakdown', color=C_TEXT, pad=10)
    ax2.tick_params(colors=C_SUB)
    for sp in ax2.spines.values():
        sp.set_edgecolor('#45475a')
    ax2.grid(axis='y', color=C_SUB, alpha=0.3, zorder=1)

    plt.tight_layout()
    out1 = 'dae_evaluation_overall.png'
    fig.savefig(out1, dpi=150, bbox_inches='tight', facecolor=C_BG)
    print(f'Saved → {out1}')

    #Figure 2: per-class accuracy (top 20 by pair count)
    sorted_cls = sorted(class_stats.items(),
                        key=lambda x: x[1]['n'], reverse=True)[:20]
    cls_names   = [c[:22]                              for c, _ in sorted_cls]
    cls_raw_acc = [s['raw'] / s['n'] * 100             for _, s in sorted_cls]
    cls_ae_acc  = [s['ae']  / s['n'] * 100             for _, s in sorted_cls]
    cls_delta   = [ae - raw for ae, raw in zip(cls_ae_acc, cls_raw_acc)]
    cls_n       = [s['n']                              for _, s in sorted_cls]

    x      = np.arange(len(cls_names))
    width  = 0.35

    fig2, (ax3, ax4) = plt.subplots(2, 1, figsize=(16, 11), facecolor=C_BG)
    fig2.suptitle('Per-Class DAE Impact — Top 20 Classes by Pair Count',
                  color=C_TEXT, fontsize=13, fontweight='bold', y=1.01)

    #top panel: grouped bars per class
    ax3.set_facecolor(C_SURFACE)
    ax3.bar(x - width / 2, cls_raw_acc, width, label='Raw',
            color=C_RED,   alpha=0.85, zorder=2)
    ax3.bar(x + width / 2, cls_ae_acc,  width, label='AE Filtered',
            color=C_GREEN, alpha=0.85, zorder=2)
    ax3.set_xticks(x)
    ax3.set_xticklabels(cls_names, rotation=45, ha='right',
                        color=C_SUB, fontsize=8)
    ax3.set_ylabel('Accuracy (%)', color=C_TEXT)
    ax3.set_ylim(0, 115)
    ax3.tick_params(colors=C_SUB)
    ax3.grid(axis='y', color=C_SUB, alpha=0.3, zorder=1)
    ax3.legend(facecolor=C_SURFACE, labelcolor=C_TEXT, edgecolor='#45475a')
    for sp in ax3.spines.values():
        sp.set_edgecolor('#45475a')
    #annotate number of pairs above each group
    for i, n in enumerate(cls_n):
        ax3.text(i, 106, f'n={n}', ha='center', color=C_SUB, fontsize=7)

    #bottom panel: delta bars
    ax4.set_facecolor(C_SURFACE)
    delta_cols = [C_GREEN if d >= 0 else C_RED for d in cls_delta]
    ax4.bar(x, cls_delta, color=delta_cols, alpha=0.85, zorder=2)
    ax4.axhline(0, color=C_TEXT, linewidth=0.8, zorder=3)
    ax4.set_xticks(x)
    ax4.set_xticklabels(cls_names, rotation=45, ha='right',
                        color=C_SUB, fontsize=8)
    ax4.set_ylabel('Delta AE − Raw (%)', color=C_TEXT)
    ax4.tick_params(colors=C_SUB)
    ax4.grid(axis='y', color=C_SUB, alpha=0.3, zorder=1)
    for sp in ax4.spines.values():
        sp.set_edgecolor('#45475a')
    for i, d in enumerate(cls_delta):
        ax4.text(i, d + (1 if d >= 0 else -2.5), f'{d:+.1f}',
                 ha='center', color=C_TEXT, fontsize=7)

    plt.tight_layout()
    out2 = 'dae_evaluation_per_class.png'
    fig2.savefig(out2, dpi=150, bbox_inches='tight', facecolor=C_BG)
    print(f'Saved → {out2}')
    plt.close('all')


# ──────────────────────────────────────── entry point

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Evaluate DAE impact on Bradley-Terry scorer accuracy')
    p.add_argument('--scorer', default='pose_scorer.pt')
    p.add_argument('--dae',    default='dae_noise001_occ002.pt')
    p.add_argument('--cache',  default='landmark_cache.pkl')
    p.add_argument('--prefs',  default='preferences.jsonl',
                   help='Use preferences_merged.jsonl for the full dataset, '
                        'or preferences.jsonl for the original split only')
    return p.parse_args()


if __name__ == '__main__':
    evaluate(parse_args())