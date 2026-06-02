# Yoga Pose Quality Scorer

A preference-learning system that scores yoga pose quality using your own aesthetic judgement. You label pairs of poses as better/worse, train a [Bradley-Terry](https://en.wikipedia.org/wiki/Bradley%E2%80%93Terry_model) model on those preferences, and get a scalar quality score (0–100%) for any new pose image.

**Pipeline overview**

```
Yoga images  →  Label pairs (Tkinter app)  →  Train Bradley-Terry MLP  →  Score new images
```

---

## Requirements

**Python 3.10+** is recommended.

```bash
pip install mediapipe opencv-python torch numpy tqdm matplotlib
```

The rest of the dependencies in `requirements_3d.txt` are for the broader 3D-vision experiments in this repo and are not needed for the scoring pipeline.

---

## Dataset

The labeling app and training scripts expect the [Yoga Pose Image Classification Dataset](https://www.kaggle.com/datasets/shrutisaxena/yoga-pose-image-classification-dataset) from Kaggle. Download it with:

```bash
python download_data.py
```

The dataset will be placed at:
```
~/.cache/kagglehub/datasets/shrutisaxena/yoga-pose-image-classification-dataset/versions/1/dataset/
```

107 pose classes, ~5994 images total.

---

## Step 1 — Collect preference labels

Run the interactive labeling app:

```bash
python preference_labeler.py
```

The app shows **pairs of images from the same pose class**. Each pair displays:
- A filled, part-coloured body render (SMPL-style) at the top, so you can judge body alignment clearly even for complex or inverted poses
- The original 2D photo at the bottom

Press `←` to prefer the left pose, `→` to prefer the right. Labels are saved to `preferences.jsonl` automatically — you can quit and resume at any time without losing progress.

**Target:** ~5000 comparisons for a reliable model. Each session appends new pairs to the file.

### Label file format

Each line in `preferences.jsonl` is:

```json
{"winner": "/path/to/image_A.jpg", "loser": "/path/to/image_B.jpg", "class": "warrior_i", "timestamp": "..."}
```

---

## Step 2 — Train the model

```bash
python train_bradley_terry.py
```

This will:
1. Run MediaPipe on every unique image and cache 3D world landmarks to `landmark_cache.pkl` (fast on re-runs)
2. Normalise each pose: Y-flip → centre at hip midpoint → scale by torso length → align torso to +Y
3. Train a 132→256→128→64→1 MLP with Bradley-Terry loss: `BCEWithLogits(score_A − score_B, 1)`
4. Save the best checkpoint (by validation accuracy) to `pose_scorer.pt`
5. Write epoch-by-epoch metrics to `training_history.json`

### Options

| Flag | Default | Description |
|---|---|---|
| `--prefs` | `preferences.jsonl` | Path to the labels file |
| `--epochs` | `60` | Number of training epochs |
| `--batch-size` | `64` | Batch size |
| `--lr` | `3e-4` | Initial learning rate |
| `--no-rotate` | off | Disable XY-plane rotation normalisation |

Example — train on a custom labels file for 80 epochs:

```bash
python train_bradley_terry.py --prefs my_labels.jsonl --epochs 80
```

### What to expect

A clean, consistently-labeled dataset of ~4000–5000 pairs typically reaches **~76–78% validation accuracy**. Human agreement on subjective pose quality is estimated at 80–85%, so this is near human-level performance.

If you have multiple label files with different labeling criteria (e.g., collected before and after changing the visualizer), train on them separately first to check consistency before merging.

---

## Step 3 — Score a pose image

```bash
python score_pose.py path/to/your/image.jpg
```

This opens a 4-panel figure:

| Panel | Content |
|---|---|
| Original image | The input photo |
| 2D skeleton | MediaPipe landmark overlay |
| 3D world pose | Interactive 3D skeleton (matplotlib) |
| Score bar | Quality % with colour coding |

Score colour bands:
- **Green** — Good (≥ 65%)
- **Yellow** — Fair (40–65%)
- **Red** — Poor (< 40%)

### Options

```bash
python score_pose.py image.jpg --checkpoint pose_scorer.pt   # use a specific checkpoint
python score_pose.py image.jpg --no-rotate                   # disable rotation normalisation
```

The checkpoint embeds the rotation setting used during training; `--no-rotate` only overrides if the checkpoint does not record it.

---

## Files

| File | Description |
|---|---|
| `preference_labeler.py` | Tkinter labeling app |
| `train_bradley_terry.py` | Training script + `normalize_pose`, `PoseScorer` |
| `score_pose.py` | Inference and visualisation |
| `preferences.jsonl` | Human preference labels |
| `landmark_cache.pkl` | Cached MediaPipe 3D landmarks (auto-generated) |
| `pose_scorer.pt` | Best model checkpoint |
| `training_history.json` | Per-epoch train/val metrics |

---

## How it works

**Bradley-Terry model:** given a pair of poses A and B, the probability that A is preferred is:

```
p(A wins) = σ(s_A − s_B)
```

where `s_A`, `s_B` are scalar quality scores output by the MLP and `σ` is the sigmoid function. The loss is `BCEWithLogitsLoss(s_A − s_B, 1)` for every labeled (winner, loser) pair.

**Pose normalisation** makes scores view-invariant:
1. Flip Y axis (MediaPipe world coords have Y pointing down)
2. Centre at hip midpoint (landmarks 23 & 24)
3. Scale by torso length (hip → shoulder midpoint distance)
4. Rotate in XY plane to align hip→shoulder to +Y

The normalised pose is a 132-dimensional vector (33 landmarks × 4 values: x, y, z, visibility) fed into the MLP.
