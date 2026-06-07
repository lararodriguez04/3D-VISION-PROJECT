"""
evaluate_pipeline.py
--------------------
Evaluates two-stage pipeline with reranking.

Modes compared:
  A) Raw           : preferences.pt only
  B) DAE + Raw     : DAE denoising + preferences.pt
  C) Rerank        : alpha * preferences.pt + (1-alpha) * missing_joints.pt
  D) DAE + Rerank  : DAE + combined score

Usage:
    python evaluate_pipeline.py
    python evaluate_pipeline.py --alpha 0.7
    python evaluate_pipeline.py --prefs preferences.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from train_bradley_terry import normalize_pose, PoseScorer
from train_dae import PoseDAE

COLLAPSE_THR = 0.05


# ── loaders ───────────────────────────────────────────────────────────

def load_scorer(path: str, device: torch.device) -> tuple[PoseScorer, bool]:
    ckpt  = torch.load(path, map_location=device, weights_only=False)
    model = PoseScorer(
        input_dim=ckpt.get('input_dim', 132),
        hidden=ckpt.get('hidden', [256, 128, 64]),
    ).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f'  {os.path.basename(path):35s} val_acc={ckpt.get("val_acc", float("nan")):.1%}')
    return model, ckpt.get('rotate', True)


def load_dae(path: str, device: torch.device) -> PoseDAE | None:
    if not path or not os.path.exists(path):
        print(f'  DAE not found at {path} — modes B and D skipped.')
        return None
    ckpt  = torch.load(path, map_location=device, weights_only=False)
    model = PoseDAE(
        input_dim=ckpt.get('input_dim', 99),
        latent_dim=ckpt.get('latent_dim', 64),
    ).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f'  {os.path.basename(path):35s} val_loss={ckpt.get("val_loss", float("nan")):.5f}')
    return model


# ── inference helpers ─────────────────────────────────────────────────

def bt_logit(world_34: np.ndarray, scorer: PoseScorer,
             rotate: bool, device: torch.device) -> float:
    feat = normalize_pose(world_34, rotate=rotate)
    t    = torch.from_numpy(feat).unsqueeze(0).to(device)
    with torch.no_grad():
        return scorer(t).item()


def apply_dae(world_34: np.ndarray, dae: PoseDAE,
              device: torch.device) -> np.ndarray:
    xyz    = world_34[:, :3].copy()
    vis    = world_34[:, 3:4]
    center = xyz.mean(axis=0)
    xyz_c  = xyz - center
    scale  = np.linalg.norm(xyz_c, axis=1).max() + 1e-8
    vec_99 = (xyz_c / scale).flatten().astype(np.float32)
    with torch.no_grad():
        vec_ae = dae(
            torch.from_numpy(vec_99).unsqueeze(0).to(device)
        ).cpu().squeeze(0).numpy()
    if vec_ae.std() < COLLAPSE_THR:
        vec_ae = vec_99.copy()
    xyz_ae = vec_ae.reshape(33, 3) * scale + center
    return np.concatenate([xyz_ae, vis], axis=1).astype(np.float32)


def get_lm(path: str, cache: dict, basename_map: dict):
    v = cache.get(path)
    if v is not None:
        return v
    return basename_map.get(os.path.basename(path))


# ── main ─────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}\n')

    print('Loading models...')
    quality_scorer, quality_rotate = load_scorer(args.scorer, device)
    mj_scorer,      mj_rotate      = load_scorer(args.mj,     device)
    dae = load_dae(args.dae, device)
    print()

    with open(args.cache, 'rb') as f:
        cache = pickle.load(f)
    basename_map = {os.path.basename(k): v for k, v in cache.items()}
    print(f'Landmark cache: {len(cache)} entries')

    pairs: list[tuple[str, str, str]] = []
    with open(args.prefs) as f:
        for line in f:
            line = line.strip()
            if not line.startswith('{'):
                continue
            try:
                r = json.loads(line)
                pairs.append((r['winner'], r['loser'], r.get('class', '?')))
            except json.JSONDecodeError:
                pass
    print(f'Preference pairs: {len(pairs)}\n')

    #alpha sweep: find best reranking weight automatically
    alphas     = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    alpha_use  = args.alpha   #fixed alpha for final report, None = auto-sweep

    #collect all logits first (one pass), then sweep alphas
    results_raw = []    #(w_q, l_q, w_mj, l_mj, w_q_dae, l_q_dae, w_mj_dae, l_mj_dae)
    n_skipped = 0

    print('Running inference...')
    for w_path, l_path, _ in pairs:
        w_lm = get_lm(w_path, cache, basename_map)
        l_lm = get_lm(l_path, cache, basename_map)
        if w_lm is None or l_lm is None:
            n_skipped += 1
            continue

        w_q   = bt_logit(w_lm, quality_scorer, quality_rotate, device)
        l_q   = bt_logit(l_lm, quality_scorer, quality_rotate, device)
        w_mj  = bt_logit(w_lm, mj_scorer,      mj_rotate,      device)
        l_mj  = bt_logit(l_lm, mj_scorer,      mj_rotate,      device)

        if dae is not None:
            w_ae    = apply_dae(w_lm, dae, device)
            l_ae    = apply_dae(l_lm, dae, device)
            w_q_dae = bt_logit(w_ae, quality_scorer, quality_rotate, device)
            l_q_dae = bt_logit(l_ae, quality_scorer, quality_rotate, device)
            w_mj_dae = bt_logit(w_ae, mj_scorer, mj_rotate, device)
            l_mj_dae = bt_logit(l_ae, mj_scorer, mj_rotate, device)
        else:
            w_q_dae = l_q_dae = w_mj_dae = l_mj_dae = 0.0

        results_raw.append((w_q, l_q, w_mj, l_mj, w_q_dae, l_q_dae, w_mj_dae, l_mj_dae))

    n_total = len(results_raw)
    print(f'Evaluated: {n_total}  skipped: {n_skipped}\n')

    #mode A: raw quality only (alpha=1.0, no MJ)
    raw_acc = sum(1 for r in results_raw if r[0] > r[1]) / n_total

    #mode B: DAE + quality only
    dae_acc = sum(1 for r in results_raw if r[4] > r[5]) / n_total if dae else raw_acc

    #alpha sweep for reranking — find best alpha
    def rerank_acc(alpha, use_dae=False):
        correct = 0
        for r in results_raw:
            if use_dae:
                w = alpha * r[4] + (1 - alpha) * r[6]
                l = alpha * r[5] + (1 - alpha) * r[7]
            else:
                w = alpha * r[0] + (1 - alpha) * r[2]
                l = alpha * r[1] + (1 - alpha) * r[3]
            correct += int(w > l)
        return correct / n_total

    print('Alpha sweep for reranking (quality weight):')
    print(f'  {"Alpha":>6}  {"Rerank":>8}  {"DAE+Rerank":>12}')
    best_alpha_rerank   = 1.0
    best_alpha_dae_rerank = 1.0
    best_acc_rerank     = raw_acc
    best_acc_dae_rerank = dae_acc

    for a in alphas:
        acc_r   = rerank_acc(a, use_dae=False)
        acc_dr  = rerank_acc(a, use_dae=True) if dae else raw_acc
        marker  = ''
        if acc_r > best_acc_rerank:
            best_acc_rerank   = acc_r
            best_alpha_rerank = a
        if acc_dr > best_acc_dae_rerank:
            best_acc_dae_rerank   = acc_dr
            best_alpha_dae_rerank = a
        print(f'  {a:>6.1f}  {acc_r:>8.1%}  {acc_dr:>12.1%}')

    #use override alpha if provided
    if alpha_use is not None:
        best_alpha_rerank     = alpha_use
        best_alpha_dae_rerank = alpha_use
        best_acc_rerank       = rerank_acc(alpha_use, use_dae=False)
        best_acc_dae_rerank   = rerank_acc(alpha_use, use_dae=True) if dae else raw_acc

    print()
    print('=' * 58)
    print(f'Pairs evaluated : {n_total}  (skipped: {n_skipped})')
    print('─' * 58)
    print(f'{"Mode":<35} {"Accuracy":>9} {"Delta":>8}')
    print('─' * 58)
    modes = [
        ('A) Raw (preferences.pt)',          raw_acc),
        ('B) DAE + preferences.pt',          dae_acc),
        (f'C) Rerank (α={best_alpha_rerank:.1f})',  best_acc_rerank),
        (f'D) DAE + Rerank (α={best_alpha_dae_rerank:.1f})', best_acc_dae_rerank),
    ]
    best_acc = max(m[1] for m in modes)
    for label, acc in modes:
        delta  = acc - raw_acc
        marker = '  <-- best' if acc == best_acc else ''
        print(f'{label:<35} {acc:>9.1%} {delta:>+8.2%}{marker}')
    print('=' * 58)

    # ── chart ────────────────────────────────────────────────────────
    C_BG      = '#1e1e2e'
    C_SURFACE = '#181825'
    C_TEXT    = '#cdd6f4'
    C_SUB     = '#6c7086'
    C_GREEN   = '#a6e3a1'
    C_RED     = '#f38ba8'
    C_BLUE    = '#89b4fa'
    C_MAUVE   = '#cba6f7'
    C_YELLOW  = '#f9e2af'

    labels_short = [
        'A) Raw',
        'B) DAE',
        f'C) Rerank\nα={best_alpha_rerank:.1f}',
        f'D) DAE+Rerank\nα={best_alpha_dae_rerank:.1f}',
    ]
    acc_vals   = [m[1] * 100 for m in modes]
    delta_vals = [(m[1] - raw_acc) * 100 for m in modes]
    colors     = [C_BLUE, C_GREEN, C_MAUVE, C_YELLOW]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), facecolor=C_BG)
    fig.suptitle('Pipeline Comparison — Pair-Ranking Accuracy',
                 color=C_TEXT, fontsize=13, fontweight='bold')

    for ax, vals, title, ylabel in [
        (ax1, acc_vals,   'Absolute Accuracy',       'Accuracy (%)'),
        (ax2, delta_vals, 'Delta vs Raw Baseline',   'Delta (%)'),
    ]:
        ax.set_facecolor(C_SURFACE)
        bar_cols = colors if ax is ax1 else \
                   [C_GREEN if d >= 0 else C_RED for d in delta_vals]
        bars = ax.bar(labels_short, vals, color=bar_cols, width=0.5, zorder=2)
        for rect, v in zip(bars, vals):
            yoff = 0.2 if ax is ax1 else (0.05 if v >= 0 else -0.2)
            ax.text(rect.get_x() + rect.get_width() / 2, v + yoff,
                    f'{v:.1f}%' if ax is ax1 else f'{v:+.2f}%',
                    ha='center', va='bottom', color=C_TEXT,
                    fontsize=10, fontweight='bold')
        if ax is ax1:
            spread = max(acc_vals) - min(acc_vals)
            ax.set_ylim(min(acc_vals) - max(spread, 2), max(acc_vals) + max(spread, 2))
        else:
            ax.axhline(0, color=C_TEXT, linewidth=0.8, zorder=3)
            spread = max(abs(d) for d in delta_vals) or 0.5
            ax.set_ylim(-spread * 2, spread * 2)
        ax.set_title(title, color=C_TEXT, pad=10)
        ax.set_ylabel(ylabel, color=C_TEXT)
        ax.tick_params(colors=C_SUB)
        ax.grid(axis='y', color=C_SUB, alpha=0.3, zorder=1)
        for sp in ax.spines.values():
            sp.set_edgecolor('#45475a')

    plt.tight_layout()
    out = 'pipeline_comparison.png'
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=C_BG)
    plt.show()
    plt.close()
    print(f'\nSaved -> {out}')


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--scorer',  default='preferences.pt')
    p.add_argument('--mj',      default='missing_joints.pt')
    p.add_argument('--dae',     default='dae_noise001_occ002.pt')
    p.add_argument('--cache',   default='landmark_cache.pkl')
    p.add_argument('--prefs',   default='preferences.jsonl')
    p.add_argument('--alpha',   type=float, default=None,
                   help='Fixed reranking alpha (auto-sweep if not set)')
    return p.parse_args()


if __name__ == '__main__':
    main(parse_args())