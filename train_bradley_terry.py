"""
Bradley-Terry preference learning for yoga pose quality.

Pipeline:
  1. Read preferences.jsonl  (winner / loser image paths)
  2. Run MediaPipe on every unique image and cache landmarks to disk
  3. Normalise each pose:  Y-flip  →  centre at hip  →  scale by torso  →  align torso to +Y
  4. Train an MLP that scores each pose; loss = BCEWithLogits(score_A - score_B, 1)
  5. Validate on held-out pairs (metric: % where score_winner > score_loser)

Usage:
    python train_bradley_terry.py
    python train_bradley_terry.py --epochs 80 --lr 1e-3 --no-rotate
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

CACHE_FILE  = 'landmark_cache.pkl'
CHECKPOINT  = 'pose_scorer.pt'
HISTORY_OUT = 'training_history.json'

# ──────────────────────────────────────── normalisation

def normalize_pose(raw: np.ndarray, rotate: bool = True) -> np.ndarray:
    """
    raw : (33, 4) float32  — MediaPipe pose_world_landmarks [x, y, z, visibility]

    Steps
    -----
    1. Flip Y so the axis points up  (MediaPipe world coords have Y pointing down)
    2. Centre: subtract hip midpoint (landmarks 23 & 24)
    3. Scale : divide by torso length  ||hip_mid → shoulder_mid||
    4. Rotate: align the hip→shoulder direction to +Y in the XY plane

    Returns flattened (132,) float32.
    """
    xyz = raw[:, :3].copy()
    vis = raw[:, 3:4]

    # 1. Y-flip  (so "up" = +Y, consistent with standard Cartesian)
    xyz[:, 1] *= -1.0

    # 2. Centre at hip midpoint
    hip_mid = (xyz[23] + xyz[24]) / 2.0
    xyz -= hip_mid

    # 3. Scale by torso length (hip → shoulder midpoint distance)
    shoulder_mid = (xyz[11] + xyz[12]) / 2.0
    torso_len = float(np.linalg.norm(shoulder_mid))
    if torso_len > 1e-6:
        xyz /= torso_len

    # 4. Rotate in XY plane so torso direction → +Y
    if rotate:
        shoulder_mid = (xyz[11] + xyz[12]) / 2.0   # recompute after scale
        dx, dy = float(shoulder_mid[0]), float(shoulder_mid[1])
        angle = np.arctan2(dx, dy)                  # signed angle from +Y
        cos_a = float(np.cos(-angle))
        sin_a = float(np.sin(-angle))
        R = np.array([[cos_a, -sin_a, 0.0],
                      [sin_a,  cos_a, 0.0],
                      [0.0,    0.0,   1.0]], dtype=np.float32)
        xyz = (R @ xyz.T).T

    out = np.concatenate([xyz, vis], axis=1)         # (33, 4)
    return out.flatten().astype(np.float32)           # (132,)


# ──────────────────────────────────────── landmark extraction & cache

def _run_mediapipe(image_path: str, pose) -> np.ndarray | None:
    """Return (33, 4) world landmarks or None if pose not detected."""
    img = cv2.imread(image_path)
    if img is None:
        return None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)
    if results.pose_world_landmarks is None:
        return None
    lms = results.pose_world_landmarks.landmark
    return np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in lms],
                    dtype=np.float32)


def build_landmark_cache(image_paths: list[str]) -> dict[str, np.ndarray | None]:
    existing: dict[str, np.ndarray | None] = {}
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'rb') as f:
            existing = pickle.load(f)

    missing = [p for p in image_paths if p not in existing]
    if not missing:
        print(f"Landmark cache: all {len(existing)} entries already cached.")
        return existing

    print(f"Extracting landmarks for {len(missing)} images  "
          f"({len(existing)} already cached)…")
    _mp_pose = mp.solutions.pose   #imported here to avoid crash when only normalize_pose is imported
    pose = _mp_pose.Pose(static_image_mode=True, model_complexity=2,
                         enable_segmentation=False, min_detection_confidence=0.5)
    for path in tqdm(missing, unit='img'):
        existing[path] = _run_mediapipe(path, pose)
    pose.close()

    with open(CACHE_FILE, 'wb') as f:
        pickle.dump(existing, f)
    print(f"Landmark cache saved → {CACHE_FILE}  ({len(existing)} total entries)")
    return existing


# ──────────────────────────────────────── dataset

class PairDataset(Dataset):
    """Each sample: (winner_features, loser_features) both (132,) float32."""

    def __init__(self, pairs: list[tuple[str, str]],
                 cache: dict[str, np.ndarray | None],
                 rotate: bool = True) -> None:
        self.samples: list[tuple[np.ndarray, np.ndarray]] = []
        skipped = 0
        for winner, loser in pairs:
            w_raw = cache.get(winner)
            l_raw = cache.get(loser)
            if w_raw is None or l_raw is None:
                skipped += 1
                continue
            self.samples.append((normalize_pose(w_raw, rotate),
                                  normalize_pose(l_raw, rotate)))
        if skipped:
            print(f"  Skipped {skipped} pairs — pose not detected in ≥1 image.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        w, l = self.samples[idx]
        return torch.from_numpy(w), torch.from_numpy(l)


# ──────────────────────────────────────── model

class PoseScorer(nn.Module):
    """MLP that maps a normalised pose vector (132,) → scalar quality score."""

    def __init__(self, input_dim: int = 132,
                 hidden: list[int] | None = None,
                 dropout: float = 0.1) -> None:
        super().__init__()
        if hidden is None:
            hidden = [256, 128, 64]
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h),
                       nn.GELU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)   # (B,)


# ──────────────────────────────────────── loss & evaluation

def bradley_terry_loss(score_winner: torch.Tensor,
                       score_loser: torch.Tensor) -> torch.Tensor:
    """p(A wins) = σ(sA − sB).  Loss = BCEWithLogits(sA − sB, 1)."""
    logit = score_winner - score_loser
    return F.binary_cross_entropy_with_logits(logit, torch.ones_like(logit))


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader,
             device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss = n = correct = 0
    for w_feat, l_feat in loader:
        w_feat, l_feat = w_feat.to(device), l_feat.to(device)
        sw, sl = model(w_feat), model(l_feat)
        total_loss += bradley_terry_loss(sw, sl).item() * len(w_feat)
        correct     += (sw > sl).sum().item()
        n           += len(w_feat)
    return total_loss / n, correct / n


# ──────────────────────────────────────── training loop

def train(args: argparse.Namespace) -> list[dict]:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device : {device}")

    # ── load preferences ──────────────────────────────────────────────
    pairs: list[tuple[str, str]] = []
    bad = 0
    with open(args.prefs) as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith('{'):   # skip nulls / garbage
                bad += 1
                continue
            try:
                rec = json.loads(line)
                pairs.append((rec['winner'], rec['loser']))
            except json.JSONDecodeError:
                bad += 1
    if bad:
        print(f"Skipped {bad} malformed lines in {args.prefs}")
    print(f"Preference pairs loaded : {len(pairs)}")

    # ── landmark cache ────────────────────────────────────────────────
    all_paths = list({p for pair in pairs for p in pair})
    cache = build_landmark_cache(all_paths)

    # ── dataset / split ───────────────────────────────────────────────
    rotate = not args.no_rotate
    dataset = PairDataset(pairs, cache, rotate=rotate)
    n_total = len(dataset)
    if n_total == 0:
        sys.exit("No valid pairs — check that MediaPipe detected poses.")

    n_val   = max(1, int(n_total * 0.15))
    n_train = n_total - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42))

    pin = device.type == 'cuda'
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                               shuffle=True,  num_workers=0, pin_memory=pin)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                               shuffle=False, num_workers=0, pin_memory=pin)
    print(f"Dataset  : {n_total} valid pairs  →  train {n_train}  /  val {n_val}")

    # ── model ─────────────────────────────────────────────────────────
    model = PoseScorer(input_dim=132, hidden=args.hidden,
                       dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model    : {n_params:,} parameters  "
          f"hidden={args.hidden}  dropout={args.dropout}  wd={args.weight_decay}\n")

    optimizer = torch.optim.AdamW(model.parameters(),
                                   lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    best_val_acc = 0.0
    history: list[dict] = []

    header = (f"{'Epoch':>6}  {'Train Loss':>10}  "
              f"{'Val Loss':>10}  {'Val Acc':>8}  {'LR':>10}")
    print(header)
    print("─" * len(header))

    for epoch in range(1, args.epochs + 1):
        # ── train ─────────────────────────────────────────────────────
        model.train()
        total_loss = n = 0
        for w_feat, l_feat in train_loader:
            w_feat, l_feat = w_feat.to(device), l_feat.to(device)
            optimizer.zero_grad()
            loss = bradley_terry_loss(model(w_feat), model(l_feat))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(w_feat)
            n          += len(w_feat)
        scheduler.step()

        train_loss           = total_loss / n
        val_loss, val_acc    = evaluate(model, val_loader, device)
        lr_now               = scheduler.get_last_lr()[0]

        row = {'epoch': epoch, 'train_loss': train_loss,
               'val_loss': val_loss, 'val_acc': val_acc, 'lr': lr_now}
        history.append(row)

        star = " ★" if val_acc > best_val_acc else ""
        print(f"{epoch:>6}  {train_loss:>10.4f}  {val_loss:>10.4f}  "
              f"{val_acc:>7.1%}  {lr_now:>10.2e}{star}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'model_state': model.state_dict(),
                'val_acc':     val_acc,
                'epoch':       epoch,
                'rotate':      rotate,
                'hidden':      args.hidden,
                'dropout':     args.dropout,
                'input_dim':   132,
            }, args.checkpoint)

    print(f"\nBest val accuracy : {best_val_acc:.1%}  →  {args.checkpoint}")

    with open(HISTORY_OUT, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Training history  →  {HISTORY_OUT}")

    return history


# ──────────────────────────────────────── entry point

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Bradley-Terry pose quality training")
    p.add_argument('--prefs',        default='preferences.jsonl',
                   help='Preference labels file')
    p.add_argument('--checkpoint',   default=CHECKPOINT,
                   help='Output checkpoint path (default: pose_scorer.pt)')
    p.add_argument('--epochs',       type=int,   default=60)
    p.add_argument('--batch-size',   type=int,   default=64)
    p.add_argument('--lr',           type=float, default=3e-4)
    p.add_argument('--dropout',      type=float, default=0.1,
                   help='Dropout rate in each MLP block (default: 0.1)')
    p.add_argument('--weight-decay', type=float, default=1e-4,
                   help='AdamW weight decay (default: 1e-4)')
    p.add_argument('--hidden',       type=int,   nargs='+', default=[256, 128, 64],
                   help='Hidden layer sizes (default: 256 128 64)')
    p.add_argument('--no-rotate',    action='store_true',
                   help='Skip rotation normalisation')
    return p.parse_args()


if __name__ == '__main__':
    train(parse_args())