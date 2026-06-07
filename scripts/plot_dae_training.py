"""
plot_dae_training.py
--------------------
Generates dae_training_curve.png from the training log of the best DAE
(dae_noise001_occ002.pt, noise=0.01, occlusion=0.02, 150 epochs, GPU run).

Usage:
    python plot_dae_training.py
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LOG = [
    (1,   0.02910, 0.00612),
    (2,   0.00953, 0.00395),
    (3,   0.00768, 0.00366),
    (4,   0.00668, 0.00312),
    (5,   0.00606, 0.00258),
    (6,   0.00560, 0.00253),
    (7,   0.00522, 0.00219),
    (8,   0.00490, 0.00207),
    (9,   0.00472, 0.00190),
    (10,  0.00451, 0.00181),
    (11,  0.00435, 0.00171),
    (12,  0.00422, 0.00167),
    (14,  0.00393, 0.00147),
    (16,  0.00371, 0.00145),
    (17,  0.00358, 0.00137),
    (20,  0.00332, 0.00123),
    (22,  0.00318, 0.00116),
    (24,  0.00305, 0.00115),
    (26,  0.00295, 0.00112),
    (27,  0.00291, 0.00112),
    (28,  0.00286, 0.00106),
    (30,  0.00277, 0.00100),
    (32,  0.00269, 0.00097),
    (33,  0.00267, 0.00096),
    (34,  0.00264, 0.00095),
    (35,  0.00261, 0.00094),
    (36,  0.00257, 0.00094),
    (37,  0.00254, 0.00090),
    (38,  0.00252, 0.00090),
    (39,  0.00249, 0.00088),
    (40,  0.00246, 0.00090),
    (43,  0.00240, 0.00088),
    (45,  0.00235, 0.00086),
    (46,  0.00233, 0.00084),
    (48,  0.00230, 0.00084),
    (49,  0.00228, 0.00081),
    (50,  0.00226, 0.00084),
    (55,  0.00217, 0.00077),
    (57,  0.00213, 0.00077),
    (59,  0.00210, 0.00075),
    (60,  0.00209, 0.00076),
    (62,  0.00205, 0.00074),
    (63,  0.00203, 0.00074),
    (66,  0.00198, 0.00073),
    (68,  0.00196, 0.00071),
    (69,  0.00194, 0.00071),
    (70,  0.00193, 0.00072),
    (71,  0.00191, 0.00070),
    (73,  0.00189, 0.00069),
    (75,  0.00187, 0.00067),
    (80,  0.00181, 0.00066),
    (82,  0.00178, 0.00065),
    (85,  0.00175, 0.00065),
    (87,  0.00173, 0.00065),
    (88,  0.00172, 0.00063),
    (90,  0.00170, 0.00064),
    (92,  0.00168, 0.00063),
    (95,  0.00166, 0.00063),
    (97,  0.00164, 0.00062),
    (98,  0.00163, 0.00062),
    (100, 0.00162, 0.00062),
    (101, 0.00161, 0.00061),
    (103, 0.00159, 0.00061),
    (105, 0.00158, 0.00060),
    (107, 0.00157, 0.00060),
    (108, 0.00156, 0.00060),
    (109, 0.00155, 0.00060),
    (110, 0.00155, 0.00060),
    (111, 0.00154, 0.00060),
    (115, 0.00152, 0.00059),
    (120, 0.00150, 0.00059),
    (121, 0.00149, 0.00059),
    (125, 0.00147, 0.00058),
    (130, 0.00146, 0.00058),
    (131, 0.00146, 0.00058),
    (140, 0.00143, 0.00058),
    (150, 0.00142, 0.00058),
]

epochs     = [r[0] for r in LOG]
train_loss = [r[1] for r in LOG]
val_loss   = [r[2] for r in LOG]

best_val   = min(val_loss)
best_epoch = epochs[val_loss.index(best_val)]

C_BG      = '#1e1e2e'
C_SURFACE = '#181825'
C_TEXT    = '#cdd6f4'
C_SUB     = '#6c7086'
C_BLUE    = '#89b4fa'
C_GREEN   = '#a6e3a1'
C_YELLOW  = '#f9e2af'

fig, ax = plt.subplots(figsize=(9, 4), facecolor=C_BG)
ax.set_facecolor(C_SURFACE)

ax.plot(epochs, train_loss, color=C_BLUE,  linewidth=2, label='Train Loss')
ax.plot(epochs, val_loss,   color=C_GREEN, linewidth=2, label='Val Loss')

ax.axvline(best_epoch, color=C_YELLOW, linestyle='--', linewidth=1.2, alpha=0.7)
ax.annotate(
    f'Best val loss {best_val:.5f}\n(epoch {best_epoch})',
    xy=(best_epoch, best_val),
    xytext=(best_epoch + 6, best_val + 0.0006),
    color=C_YELLOW, fontsize=8,
    arrowprops=dict(arrowstyle='->', color=C_YELLOW, lw=1.2)
)

ax.set_xlabel('Epoch', color=C_TEXT, fontsize=10)
ax.set_ylabel('Loss',  color=C_TEXT, fontsize=10)
ax.set_title('DAE Training Convergence  (noise=0.01, occlusion=0.02)',
             color=C_TEXT, fontsize=11, fontweight='bold')
ax.tick_params(colors=C_SUB)
ax.legend(facecolor=C_SURFACE, labelcolor=C_TEXT, edgecolor='#45475a')
for sp in ax.spines.values():
    sp.set_edgecolor('#45475a')
ax.grid(color=C_SUB, alpha=0.2)

plt.tight_layout()
out = 'dae_training_curve.png'
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=C_BG)
plt.show()
plt.close()
print(f'Saved -> {out}')