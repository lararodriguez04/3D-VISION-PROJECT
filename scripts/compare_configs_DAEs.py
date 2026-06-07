"""
compare_configs.py
------------------
Evaluates multiple DAE checkpoints against Mustapha's Bradley-Terry scorer
and produces a side-by-side comparison table + bar chart.

Usage:
    #auto-discovers all dae*.pt files in the current directory
    python compare_configs.py

    #or specify explicitly
    python compare_configs.py --dae dae.pt dae_lownoise.pt dae_verylownoise.pt dae_overfit.pt

    #custom prefs / scorer
    python compare_configs.py --prefs preferences.jsonl --scorer pose_scorer.pt

Output:
    console table with accuracy and delta per config
    compare_configs.png  saved to disk
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import pickle
import sys

import matplotlib
# matplotlib.use('Agg')  #disabled: show window instead of saving only
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from train_bradley_terry import normalize_pose, PoseScorer
from train_dae import PoseDAE

COLLAPSE_THR = 0.05


# ── reuse apply_dae logic inline (avoids circular import with evaluate_dae) ──

def _apply_dae(world_34: np.ndarray, dae: PoseDAE,
               device: torch.device) -> tuple[np.ndarray, float]:
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
    ae_std = float(vec_ae.std())
    if ae_std < COLLAPSE_THR:
        vec_ae = vec_99.copy()
    xyz_ae = vec_ae.reshape(33, 3) * scale + center
    return np.concatenate([xyz_ae, vis], axis=1).astype(np.float32), ae_std


def _bt_logit(world_34: np.ndarray, scorer: nn.Module,
              rotate: bool, device: torch.device) -> float:
    feat = normalize_pose(world_34, rotate=rotate)
    t    = torch.from_numpy(feat).unsqueeze(0).to(device)
    with torch.no_grad():
        return scorer(t).item()


# ── evaluate one checkpoint over all pairs ───────────────────────────

def eval_one(dae_path: str, pairs: list, cache: dict, basename_map: dict,
             scorer: nn.Module, bt_rotate: bool,
             device: torch.device) -> dict:

    ckpt = torch.load(dae_path, map_location=device, weights_only=False)
    dae  = PoseDAE(
        input_dim=ckpt.get('input_dim', 99),
        latent_dim=ckpt.get('latent_dim', 64),
    ).to(device)
    dae.load_state_dict(ckpt['model_state'])
    dae.eval()

    noise    = ckpt.get('noise_std', '?')
    epoch    = ckpt.get('epoch', '?')
    val_loss = ckpt.get('val_loss', float('nan'))

    n_total = raw_correct = ae_correct = n_skipped = n_collapsed = 0

    for winner_path, loser_path, _ in pairs:
        w_lm = cache.get(winner_path)
        if w_lm is None:
            w_lm = basename_map.get(os.path.basename(winner_path))
        l_lm = cache.get(loser_path)
        if l_lm is None:
            l_lm = basename_map.get(os.path.basename(loser_path))

        if w_lm is None or l_lm is None:
            n_skipped += 1
            continue

        n_total += 1

        raw_ok = _bt_logit(w_lm, scorer, bt_rotate, device) > \
                 _bt_logit(l_lm, scorer, bt_rotate, device)

        w_ae, w_std = _apply_dae(w_lm, dae, device)
        l_ae, l_std = _apply_dae(l_lm, dae, device)
        ae_ok = _bt_logit(w_ae, scorer, bt_rotate, device) > \
                _bt_logit(l_ae, scorer, bt_rotate, device)

        if w_std < COLLAPSE_THR or l_std < COLLAPSE_THR:
            n_collapsed += 1

        raw_correct += int(raw_ok)
        ae_correct  += int(ae_ok)

    if n_total == 0:
        return None

    raw_acc = raw_correct / n_total
    ae_acc  = ae_correct  / n_total

    return {
        'name':       os.path.basename(dae_path),
        'noise':      noise,
        'epoch':      epoch,
        'val_loss':   val_loss,
        'n_total':    n_total,
        'n_skipped':  n_skipped,
        'n_collapsed':n_collapsed,
        'raw_acc':    raw_acc,
        'ae_acc':     ae_acc,
        'delta':      ae_acc - raw_acc,
    }


# ── main ─────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    #load scorer
    ckpt      = torch.load(args.scorer, map_location=device, weights_only=False)
    bt_rotate = ckpt.get('rotate', True)
    scorer    = PoseScorer(
        input_dim=ckpt.get('input_dim', 132),
        hidden=ckpt.get('hidden', [256, 128, 64]),
    ).to(device)
    scorer.load_state_dict(ckpt['model_state'])
    scorer.eval()
    original_val_acc = ckpt.get('val_acc', None)
    print(f'BT scorer: val_acc={original_val_acc:.1%}  rotate={bt_rotate}')

    #load cache
    with open(args.cache, 'rb') as f:
        cache = pickle.load(f)
    basename_map = {os.path.basename(k): v for k, v in cache.items()}

    #load pairs
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
    print(f'Pairs: {len(pairs)}\n')

    #discover checkpoints
    dae_paths = args.dae if args.dae else sorted(glob.glob('dae*.pt'))
    if not dae_paths:
        sys.exit('No dae*.pt files found. Pass --dae explicitly or run train_dae.py first.')
    print(f'Checkpoints to evaluate: {len(dae_paths)}')
    for p in dae_paths:
        print(f'  {p}')
    print()

    #evaluate each
    results = []
    for path in dae_paths:
        print(f'Evaluating {path} ...', end=' ', flush=True)
        r = eval_one(path, pairs, cache, basename_map,
                     scorer, bt_rotate, device)
        if r is None:
            print('SKIPPED (no pairs matched)')
            continue
        results.append(r)
        print(f'raw={r["raw_acc"]:.1%}  ae={r["ae_acc"]:.1%}  delta={r["delta"]:+.1%}')

    if not results:
        sys.exit('No results — check that cache paths match preference paths.')

    #sort by delta descending
    results.sort(key=lambda x: x['delta'], reverse=True)

    #console table
    print()
    print(f'{"Config":<30} {"Noise":>6} {"Epoch":>6} {"ValLoss":>8} '
          f'{"Raw":>7} {"AE":>7} {"Delta":>7} {"Collapsed":>10}')
    print('─' * 85)
    for r in results:
        marker = ' <-- best' if r == results[0] else ''
        print(f'{r["name"]:<30} {str(r["noise"]):>6} {str(r["epoch"]):>6} '
              f'{r["val_loss"]:>8.5f} {r["raw_acc"]:>7.1%} {r["ae_acc"]:>7.1%} '
              f'{r["delta"]:>+7.1%} {r["n_collapsed"]:>10}{marker}')

    #all configs share the same raw_acc (same scorer, same pairs) — sanity check
    raw_accs = [r['raw_acc'] for r in results]
    if max(raw_accs) - min(raw_accs) > 0.001:
        print('\nWARNING: raw_acc differs across configs — unexpected, check pairs.')

    # ── chart ────────────────────────────────────────────────────────
    C_BG      = '#1e1e2e'
    C_SURFACE = '#181825'
    C_TEXT    = '#cdd6f4'
    C_SUB     = '#6c7086'
    C_GREEN   = '#a6e3a1'
    C_RED     = '#f38ba8'
    C_YELLOW  = '#f9e2af'
    C_BLUE    = '#89b4fa'

    names      = [r['name'].replace('.pt', '') for r in results]
    ae_accs    = [r['ae_acc']  * 100 for r in results]
    deltas     = [r['delta']   * 100 for r in results]
    raw_acc_pct = results[0]['raw_acc'] * 100   #same for all

    x     = np.arange(len(results))
    width = 0.45

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(10, len(results) * 2), 10),
                                    facecolor=C_BG)
    fig.suptitle('DAE Configuration Comparison — Bradley-Terry Accuracy',
                 color=C_TEXT, fontsize=13, fontweight='bold')

    #top panel: AE accuracy per config + raw baseline
    ax1.set_facecolor(C_SURFACE)
    bars = ax1.bar(x, ae_accs, width,
                   color=[C_GREEN if d >= 0 else C_RED for d in deltas],
                   alpha=0.85, zorder=2)
    ax1.axhline(raw_acc_pct, color=C_YELLOW, linestyle='--',
                linewidth=1.5, zorder=3, label=f'Raw baseline {raw_acc_pct:.1f}%')
    if original_val_acc is not None:
        ax1.axhline(original_val_acc * 100, color=C_BLUE, linestyle=':',
                    linewidth=1.5, zorder=3,
                    label=f'Preference Models val_acc {original_val_acc:.1%}')
    for rect, v in zip(bars, ae_accs):
        ax1.text(rect.get_x() + rect.get_width() / 2, v + 0.3,
                 f'{v:.1f}%', ha='center', va='bottom',
                 color=C_TEXT, fontsize=9, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=25, ha='right', color=C_SUB, fontsize=9)
    ax1.set_ylabel('AE Accuracy (%)', color=C_TEXT)
    ax1.set_ylim(0, 100)
    ax1.tick_params(colors=C_SUB)
    ax1.grid(axis='y', color=C_SUB, alpha=0.3, zorder=1)
    ax1.legend(facecolor=C_SURFACE, labelcolor=C_TEXT, edgecolor='#45475a')
    for sp in ax1.spines.values():
        sp.set_edgecolor('#45475a')

    #bottom panel: delta per config
    ax2.set_facecolor(C_SURFACE)
    delta_colors = [C_GREEN if d >= 0 else C_RED for d in deltas]
    bars2 = ax2.bar(x, deltas, width, color=delta_colors, alpha=0.85, zorder=2)
    ax2.axhline(0, color=C_TEXT, linewidth=0.8, zorder=3)
    for rect, d in zip(bars2, deltas):
        spread = max(abs(d) for d in deltas) or 1.0
        ypos = d + spread * 0.05 if d >= 0 else d - spread * 0.15
        ax2.text(rect.get_x() + rect.get_width() / 2, ypos,
                 f'{d:+.2f}%', ha='center', va='bottom',
                 color=C_TEXT, fontsize=9, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=25, ha='right', color=C_SUB, fontsize=9)
    spread = max(abs(d) for d in deltas) or 0.5
    ax2.set_ylim(-spread * 1.6, spread * 1.6)
    ax2.set_ylabel('Delta AE − Raw (%)', color=C_TEXT)
    ax2.tick_params(colors=C_SUB)
    ax2.grid(axis='y', color=C_SUB, alpha=0.3, zorder=1)
    for sp in ax2.spines.values():
        sp.set_edgecolor('#45475a')

    plt.tight_layout()
    out = 'compare_configs.png'
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor=C_BG)
    plt.show()
    plt.close()
    print(f'\nSaved -> {out}')
    print(f'Best config: {results[0]["name"]}  delta={results[0]["delta"]:+.1%}')


# ── entry point ───────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Compare multiple DAE configurations')
    p.add_argument('--dae',     nargs='+', default=None,
                   help='DAE checkpoint files. If omitted, auto-discovers dae*.pt')
    p.add_argument('--scorer',  default='pose_scorer.pt')
    p.add_argument('--cache',   default='landmark_cache.pkl')
    p.add_argument('--prefs',   default='preferences.jsonl')
    return p.parse_args()


if __name__ == '__main__':
    main(parse_args())