"""
train_dae.py
------------
Trains a Denoising Autoencoder (DAE) on the yoga pose landmark cache
produced by train_bradley_terry.py.

The DAE operates on the xyz coordinates of MediaPipe world landmarks
after applying Mustapha's normalize_pose() preprocessing, which
produces 99-dim vectors (33 joints x 3 coords, visibility dropped).
The DAE learns to recover clean poses from noisy/occluded inputs.

Usage:
    python train_dae.py
    python train_dae.py --epochs 200 --latent 64 --noise 0.04
    python train_dae.py --cache landmark_cache.pkl --out dae.pt

Output:
    dae.pt   trained DAE checkpoint
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

from train_bradley_terry import normalize_pose

DAE_CHECKPOINT = 'dae.pt'

#MediaPipe 33-joint peripheral indices (wrists, hands, ankles, feet)
#these joints collapse first under standard MSE so they receive 3x weight
PERIPHERAL_INDICES = [15, 16, 17, 18, 19, 20, 21, 22, 27, 28, 29, 30, 31, 32]

#representative bone pairs for anatomical length consistency loss
BONE_PAIRS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (29, 31),
    (24, 26), (26, 28), (28, 30), (30, 32),
]


# ──────────────────────────────────────── architecture

class _ResBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim), nn.LayerNorm(dim), nn.GELU(),
            nn.Linear(dim, dim), nn.LayerNorm(dim),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #additive skip prevents variance collapse in the bottleneck
        return self.act(x + self.block(x))


class PoseDAE(nn.Module):
    """
    Denoising Autoencoder for 99-dim normalized pose vectors.

    Architecture choices vs a plain MLP:
    - LayerNorm instead of BatchNorm1d: safe at batch_size=1 during inference,
      no running statistics that pull single samples toward the training mean.
    - ResBlocks with additive skips: gradient path of magnitude 1 regardless
      of bottleneck width; peripheral joints cannot be silently dropped.
    - Input skip (0.15 weight): output is anchored to the input scale so the
      network cannot collapse all predictions to the mean pose.
    - GELU instead of ReLU: no hard zero for negative coords (which are common
      after centering), preventing dead neurons in the bottleneck.
    """

    def __init__(self, input_dim: int = 99, latent_dim: int = 64) -> None:
        super().__init__()
        self.enc_proj = nn.Sequential(
            nn.Linear(input_dim, 128), nn.LayerNorm(128), nn.GELU(),
        )
        self.enc_res = _ResBlock(128)
        self.enc_out = nn.Linear(128, latent_dim)

        self.dec_proj = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.LayerNorm(128), nn.GELU(),
        )
        self.dec_res = _ResBlock(128)
        self.dec_out = nn.Linear(128, input_dim)

        #bypass path: 15% weighted shortcut from input to output
        #prevents the AE from producing a constant mean-pose regardless of input
        self.skip = nn.Linear(input_dim, input_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.enc_out(self.enc_res(self.enc_proj(x)))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.dec_out(self.dec_res(self.dec_proj(z)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x)) + 0.15 * self.skip(x)


# ──────────────────────────────────────── loss

def _build_weight_vector(input_dim: int, device: torch.device) -> torch.Tensor:
    #3x weight on peripheral joints so they are not sacrificed to fit the torso
    w = torch.ones(input_dim, dtype=torch.float32)
    for ji in PERIPHERAL_INDICES:
        w[ji * 3: ji * 3 + 3] = 3.0
    return w.to(device)


def _weighted_mse(output: torch.Tensor, target: torch.Tensor,
                  weights: torch.Tensor) -> torch.Tensor:
    return ((output - target) ** 2 * weights).mean()


def _per_joint_var_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    #per-joint variance deficit: forces every joint to maintain realistic spread,
    #not just the aggregate — prevents the network gaming a global std constraint
    out_j = output.view(-1, 33, 3)
    tgt_j = target.view(-1, 33, 3)
    return torch.clamp(tgt_j.var(dim=0) - out_j.var(dim=0), min=0).mean()


def _bone_length_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    #penalizes bone segment length distortion; prevents the decoder from
    #placing joints at mean positions even if individual positions look OK
    out_j = output.view(-1, 33, 3)
    tgt_j = target.view(-1, 33, 3)
    total = torch.tensor(0.0, device=output.device)
    for a, b in BONE_PAIRS:
        total = total + torch.abs(
            torch.norm(out_j[:, a] - out_j[:, b], dim=1) -
            torch.norm(tgt_j[:, a] - tgt_j[:, b], dim=1)
        ).mean()
    return total / len(BONE_PAIRS)


# ──────────────────────────────────────── data helpers

def _xyz_from_world(world_34: np.ndarray) -> np.ndarray:
    """Extract and normalize xyz from a (33,4) world-landmark array -> (99,)."""
    xyz    = world_34[:, :3].copy()
    center = xyz.mean(axis=0)
    xyz_c  = xyz - center
    scale  = np.linalg.norm(xyz_c, axis=1).max() + 1e-8
    return (xyz_c / scale).flatten().astype(np.float32)


def _generate_noisy(clean: np.ndarray,
                    noise_std: float,
                    occlusion_prob: float,
                    rng: np.random.Generator) -> np.ndarray:
    """Add Gaussian noise and random joint occlusions to a clean vector."""
    noisy = clean + rng.normal(0, noise_std, clean.shape).astype(np.float32)
    #occlude random joints by zeroing all 3 of their coordinates
    for ji in range(33):
        if rng.random() < occlusion_prob:
            noisy[ji * 3: ji * 3 + 3] = 0.0
    return noisy


def build_dataset(cache: dict,
                  noise_std: float,
                  occlusion_prob: float,
                  augment_factor: int = 5) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build (noisy_input, clean_target) pairs from the landmark cache.
    Each clean vector is augmented `augment_factor` times with different noise
    to increase effective dataset size.
    """
    rng        = np.random.default_rng(42)
    clean_vecs = []
    for arr in cache.values():
        if arr is None:
            continue
        clean_vecs.append(_xyz_from_world(arr))

    if not clean_vecs:
        sys.exit('No valid landmarks in cache — run train_bradley_terry.py first.')

    print(f'Clean vectors extracted from cache: {len(clean_vecs)}')
    clean_arr = np.stack(clean_vecs)          #(N, 99)

    noisy_list, clean_list = [], []
    for _ in range(augment_factor):
        for cv in clean_arr:
            noisy_list.append(_generate_noisy(cv, noise_std, occlusion_prob, rng))
            clean_list.append(cv)

    noisy_t = torch.from_numpy(np.stack(noisy_list))
    clean_t = torch.from_numpy(np.stack(clean_list))
    print(f'Training pairs generated: {len(noisy_t)}  '
          f'(augment x{augment_factor}, noise_std={noise_std}, '
          f'occlusion_prob={occlusion_prob})')
    return noisy_t, clean_t


# ──────────────────────────────────────── training

def train(args: argparse.Namespace) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device : {device}')

    #load Mustapha's landmark cache
    if not os.path.exists(args.cache):
        sys.exit(f'Cache not found: {args.cache}  '
                 f'Run train_bradley_terry.py first to generate it.')
    with open(args.cache, 'rb') as f:
        cache = pickle.load(f)
    print(f'Landmark cache loaded: {len(cache)} entries  '
          f'({sum(1 for v in cache.values() if v is None)} undetected)')

    noisy_t, clean_t = build_dataset(cache, args.noise, args.occlusion)

    #train/val split
    full_ds  = TensorDataset(noisy_t, clean_t)
    n_val    = max(1, int(len(full_ds) * 0.1))
    n_train  = len(full_ds) - n_val
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False)
    print(f'Dataset: {len(full_ds)} pairs  →  train {n_train} / val {n_val}\n')

    #model + optimizer
    model     = PoseDAE(input_dim=99, latent_dim=args.latent).to(device)
    weights   = _build_weight_vector(99, device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    n_params = sum(p.numel() for p in model.parameters())
    print(f'PoseDAE: {n_params:,} parameters  (latent_dim={args.latent})')

    header = (f"{'Epoch':>6}  {'TrainLoss':>10}  {'ValLoss':>10}  "
              f"{'wMSE':>8}  {'VarL':>8}  {'BoneL':>8}  {'LR':>9}")
    print(header)
    print('─' * len(header))

    best_val = float('inf')

    for epoch in range(1, args.epochs + 1):
        model.train()
        tot = tot_mse = tot_var = tot_bone = n = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            out = model(batch_x)

            loss_mse  = _weighted_mse(out, batch_y, weights)
            loss_var  = _per_joint_var_loss(out, batch_y)
            loss_bone = _bone_length_loss(out, batch_y)
            loss      = loss_mse + 0.3 * loss_var + 0.1 * loss_bone

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            b = len(batch_x)
            tot      += loss.item()      * b
            tot_mse  += loss_mse.item()  * b
            tot_var  += loss_var.item()  * b
            tot_bone += loss_bone.item() * b
            n        += b

        scheduler.step()

        #validation
        model.eval()
        val_loss = val_n = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                out       = model(batch_x)
                l         = _weighted_mse(out, batch_y, weights)
                val_loss += l.item() * len(batch_x)
                val_n    += len(batch_x)
        val_loss /= val_n

        star = ' ★' if val_loss < best_val else ''
        if epoch % 10 == 0 or epoch == 1 or val_loss < best_val:
            print(f'{epoch:>6}  {tot/n:>10.5f}  {val_loss:>10.5f}  '
                  f'{tot_mse/n:>8.5f}  {tot_var/n:>8.5f}  {tot_bone/n:>8.5f}  '
                  f'{scheduler.get_last_lr()[0]:>9.2e}{star}')

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                'model_state': model.state_dict(),
                'input_dim':   99,
                'latent_dim':  args.latent,
                'val_loss':    best_val,
                'epoch':       epoch,
                'noise_std':   args.noise,
            }, args.out)

    print(f'\nBest val loss : {best_val:.5f}  →  {args.out}')


# ──────────────────────────────────────── entry point

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Train the Pose Denoising Autoencoder')
    p.add_argument('--cache',       default='landmark_cache.pkl')
    p.add_argument('--out',         default=DAE_CHECKPOINT)
    p.add_argument('--epochs',      type=int,   default=150)
    p.add_argument('--batch-size',  type=int,   default=32)
    p.add_argument('--lr',          type=float, default=8e-4)
    p.add_argument('--latent',      type=int,   default=64)
    p.add_argument('--noise',       type=float, default=0.03,
                   help='Gaussian noise std added to clean vectors')
    p.add_argument('--occlusion',   type=float, default=0.05,
                   help='Probability of zeroing each joint during training')
    return p.parse_args()


if __name__ == '__main__':
    train(parse_args())
