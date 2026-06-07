import argparse
import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageTk
import json
import os
import random
from pathlib import Path
from datetime import datetime
import threading
from collections import OrderedDict

DATASET_DIR = '/home/mustapha/.cache/kagglehub/datasets/shrutisaxena/yoga-pose-image-classification-dataset/versions/1/dataset/'
DISPLAY_W, DISPLAY_H = 400, 300  # per-panel dimensions
CACHE_LIMIT = 400

# Labeling modes — each maps to a different output file, question, and hint
MODES: dict[str, dict] = {
    'visual': {
        'output':   'preferences.jsonl',
        'target':   5000,
        'question': 'Which pose looks visually better?',
        'hint':     'Overall impression — body shape, balance, clarity of form',
    },
    'missing': {
        'output':   'missing_joints.jsonl',
        'target':   2000,
        'question': 'Which pose has more complete, visible joints?',
        'hint':     'Prefer the pose where more body parts are clearly detected',
    },
    'quality': {
        'output':   'pose_quality.jsonl',
        'target':   5000,
        'question': 'Which pose shows better joint alignment?',
        'hint':     'Knee over ankle · spine neutral · full limb extension',
    },
}

# ──────────────────────────────────────── SMPL-style body renderer

# RGB colors — warm tones = person's left side, cool = right (standard SMPL convention)
_BODY_RGB: dict[str, tuple[int, int, int]] = {
    'bg':      ( 15,  15,  30),
    'head':    (240, 190, 145),
    'neck':    (210, 165, 125),
    'torso':   ( 95, 155, 240),
    'l_up':    (255, 140,  50),   # left upper arm
    'l_lo':    (255, 180, 100),   # left forearm
    'l_hand':  (255, 215, 155),
    'r_up':    ( 80, 205, 115),   # right upper arm
    'r_lo':    (125, 225, 158),
    'r_hand':  (165, 240, 200),
    'l_thi':   (230,  72,  72),   # left thigh
    'l_shin':  (245, 118, 118),
    'l_foot':  (205,  52,  52),
    'r_thi':   (150,  82, 235),   # right thigh
    'r_shin':  (188, 128, 250),
    'r_foot':  (130,  62, 215),
}


def _make_bgr(rgb):
    return (rgb[2], rgb[1], rgb[0])


def _dim(rgb, factor=0.65):
    return (int(rgb[0]*factor), int(rgb[1]*factor), int(rgb[2]*factor))


def _highlight(rgb):
    """Brighter BGR highlight for the 3D-cylinder stripe."""
    r, g, b = rgb
    return (min(255, int(b*1.38+22)),
            min(255, int(g*1.38+22)),
            min(255, int(r*1.38+22)))


def _capsule(canvas, p1, p2, rgb, radius):
    """
    Filled capsule with a central highlight stripe.
    canvas is BGR; rgb is an RGB tuple.
    """
    a = (int(p1[0]), int(p1[1]))
    b = (int(p2[0]), int(p2[1]))
    bgr = _make_bgr(rgb)
    cv2.line(canvas, a, b, bgr, radius * 2, cv2.LINE_AA)
    cv2.circle(canvas, a, radius, bgr, -1, cv2.LINE_AA)
    cv2.circle(canvas, b, radius, bgr, -1, cv2.LINE_AA)
    if radius >= 5:
        hr  = max(1, radius // 3)
        hbr = _highlight(rgb)
        cv2.line(canvas, a, b, hbr, hr * 2, cv2.LINE_AA)
        cv2.circle(canvas, a, hr, hbr, -1, cv2.LINE_AA)
        cv2.circle(canvas, b, hr, hbr, -1, cv2.LINE_AA)


def render_body_smpl(img_rgb, results):
    """
    SMPL-style part-colored body visualization.

    All major segments are always drawn — MediaPipe's near-zero visibility
    just means the landmark is on the occluded side; coordinates are still
    valid. Occluded segments (vis < 0.30) are dimmed to ~50% brightness so
    the viewer sees them as "behind" the body without them disappearing.
    Hand/foot tips use a 0.15 threshold since tiny extrapolated extremities
    look wrong. Depth order is determined per-side from z-anchor joints.

    Returns RGB ndarray the same size as img_rgb.
    """
    h, w = img_rgb.shape[:2]
    canvas = np.full((h, w, 3), _make_bgr(_BODY_RGB['bg']), dtype=np.uint8)

    if not results.pose_landmarks:
        return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

    lms = results.pose_landmarks.landmark

    def pt(i):
        return np.array([lms[i].x * w, lms[i].y * h])

    def vis(i):
        return float(lms[i].visibility)

    def color(key, v):
        rgb = _BODY_RGB[key]
        return rgb if v >= 0.30 else _dim(rgb)

    def seg(p1, p2, key, radius, v1, v2):
        _capsule(canvas, p1, p2, color(key, min(v1, v2)), radius)

    torso_h = float(np.linalg.norm((pt(11)+pt(12))/2 - (pt(23)+pt(24))/2))
    R = max(8, int(torso_h / 8))

    # Depth order: side with larger z is further from camera
    left_back = (lms[11].z + lms[23].z) > (lms[12].z + lms[24].z)

    def draw_left_arm():
        seg(pt(11), pt(13), 'l_up',   int(R*0.82), vis(11), vis(13))
        seg(pt(13), pt(15), 'l_lo',   int(R*0.67), vis(13), vis(15))
        for tip in [17, 19, 21]:
            if vis(15) > 0.15 and vis(tip) > 0.15:
                seg(pt(15), pt(tip), 'l_hand', int(R*0.44), vis(15), vis(tip))

    def draw_right_arm():
        seg(pt(12), pt(14), 'r_up',   int(R*0.82), vis(12), vis(14))
        seg(pt(14), pt(16), 'r_lo',   int(R*0.67), vis(14), vis(16))
        for tip in [18, 20, 22]:
            if vis(16) > 0.15 and vis(tip) > 0.15:
                seg(pt(16), pt(tip), 'r_hand', int(R*0.44), vis(16), vis(tip))

    def draw_left_leg():
        seg(pt(23), pt(25), 'l_thi',  int(R*1.02), vis(23), vis(25))
        seg(pt(25), pt(27), 'l_shin', int(R*0.82), vis(25), vis(27))
        for tip in [29, 31]:
            if vis(27) > 0.15 and vis(tip) > 0.15:
                seg(pt(27), pt(tip), 'l_foot', int(R*0.50), vis(27), vis(tip))

    def draw_right_leg():
        seg(pt(24), pt(26), 'r_thi',  int(R*1.02), vis(24), vis(26))
        seg(pt(26), pt(28), 'r_shin', int(R*0.82), vis(26), vis(28))
        for tip in [30, 32]:
            if vis(28) > 0.15 and vis(tip) > 0.15:
                seg(pt(28), pt(tip), 'r_foot', int(R*0.50), vis(28), vis(tip))

    # Back limbs first
    if left_back:
        draw_left_arm();  draw_left_leg()
    else:
        draw_right_arm(); draw_right_leg()

    # Torso — always render; dim if average visibility is low.
    # In profile / side-view poses the shoulder width collapses to near-zero
    # and fillConvexPoly degenerates to a hairline.  When that happens, fall
    # back to a thick capsule down the spine centre so the torso is always
    # clearly visible regardless of camera angle.
    tl, tr, bl, br = pt(11), pt(12), pt(23), pt(24)
    avg_vis   = (vis(11)+vis(12)+vis(23)+vis(24)) / 4
    tc_rgb    = _BODY_RGB['torso'] if avg_vis >= 0.30 else _dim(_BODY_RGB['torso'])
    shldr_w   = float(np.linalg.norm(tl - tr))
    if shldr_w >= torso_h * 0.15:          # full polygon when body faces camera
        cv2.fillConvexPoly(canvas, np.array([tl, tr, br, bl], dtype=np.int32),
                           _make_bgr(tc_rgb))
    else:                                   # profile / edge-on: draw a spine capsule
        spine_r = max(int(R * 0.75), 6)
        mid_top = ((tl+tr)/2).astype(np.int32)
        mid_bot = ((bl+br)/2).astype(np.int32)
        _capsule(canvas, mid_top, mid_bot, tc_rgb, spine_r)
    # Spine highlight
    mid_top = ((tl+tr)/2).astype(np.int32)
    mid_bot = ((bl+br)/2).astype(np.int32)
    tw = max(4, int(shldr_w * 0.18))
    cv2.line(canvas, tuple(mid_top), tuple(mid_bot), _highlight(tc_rgb), tw*2, cv2.LINE_AA)

    # Neck
    if vis(9) > 0.05 and vis(10) > 0.05:
        neck_base = (pt(11)+pt(12)) / 2
        chin      = (pt(9)+pt(10))  / 2
        seg(neck_base, chin, 'neck', int(R*0.52), vis(9), vis(10))

    # Front limbs
    if left_back:
        draw_right_arm(); draw_right_leg()
    else:
        draw_left_arm();  draw_left_leg()

    # Head
    if vis(7) > 0.05 or vis(8) > 0.05:
        ear_mid = ((pt(7)+pt(8))/2).astype(np.int32)
        head_r  = max(int(R*1.3), int(np.linalg.norm(pt(7)-pt(8))*0.68))
        cv2.circle(canvas, tuple(ear_mid), head_r,            _make_bgr(_BODY_RGB['head']), -1, cv2.LINE_AA)
        cv2.circle(canvas, tuple(ear_mid), int(head_r*0.6),   _highlight(_BODY_RGB['head']),  -1, cv2.LINE_AA)
        cv2.circle(canvas, tuple(ear_mid), head_r,            _make_bgr(_BODY_RGB['neck']),   2,  cv2.LINE_AA)
    elif vis(0) > 0.05:
        nose   = pt(0).astype(np.int32)
        head_r = int(R*1.6)
        center = tuple(nose - np.array([0, head_r//2]))
        cv2.circle(canvas, center, head_r,           _make_bgr(_BODY_RGB['head']), -1, cv2.LINE_AA)
        cv2.circle(canvas, center, int(head_r*0.6),  _highlight(_BODY_RGB['head']),  -1, cv2.LINE_AA)

    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)



class LRUCache:
    def __init__(self, maxsize):
        self.maxsize = maxsize
        self._cache = OrderedDict()

    def get(self, key):
        if key not in self._cache:
            return None
        self._cache.move_to_end(key)
        return self._cache[key]

    def put(self, key, value):
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = value
        if len(self._cache) > self.maxsize:
            self._cache.popitem(last=False)


class PoseLabelingApp:
    def __init__(self, root, mode_cfg: dict):
        self.root = root
        self.mode_cfg = mode_cfg
        self.output_file = mode_cfg['output']
        self.target = mode_cfg['target']
        self.root.title(f"Yoga Pose Labeler — {mode_cfg['output']}")
        self.root.configure(bg='#1e1e2e')
        self.root.resizable(False, False)

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5
        )

        self.pose_cache = LRUCache(CACHE_LIMIT)
        self.cache_lock = threading.Lock()

        self.dataset = self._load_dataset()
        self.pair_queue = []
        self.completed = 0
        self.skipped = 0
        self.current_pair = None
        self._prefetch_thread = None
        self._prefetch_cache = {}  # path -> (skel, detected) pre-fetched result

        self._load_progress()
        self._build_ui()
        self._generate_pairs(300)
        self._load_next_pair()

    # ------------------------------------------------------------------ dataset

    def _load_dataset(self):
        dataset = {}
        base = Path(DATASET_DIR)
        exts = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
        for class_dir in sorted(base.iterdir()):
            if class_dir.is_dir():
                images = [p for p in class_dir.iterdir() if p.suffix in exts]
                if len(images) >= 2:
                    dataset[class_dir.name] = images
        return dataset

    def _load_progress(self):
        if os.path.exists(self.output_file):
            with open(self.output_file, 'r') as f:
                self.completed = sum(1 for line in f if line.strip())

    def _generate_pairs(self, n=200):
        classes = list(self.dataset.keys())
        new_pairs = []
        for _ in range(n):
            cls = random.choice(classes)
            images = self.dataset[cls]
            a, b = random.sample(images, 2)
            new_pairs.append((cls, a, b))
        random.shuffle(new_pairs)
        self.pair_queue.extend(new_pairs)

    # ------------------------------------------------------------------ pose

    def _extract_pose(self, image_path):
        """Return (body_rgb_ndarray, detected_bool). Uses LRU cache."""
        key = str(image_path)
        cached = self.pose_cache.get(key)
        if cached is not None:
            return cached

        img = cv2.imread(key)
        if img is None:
            blank = np.full((DISPLAY_H, DISPLAY_W, 3),
                            _BODY_RGB['bg'], dtype=np.uint8)
            result = (blank, False)
            with self.cache_lock:
                self.pose_cache.put(key, result)
            return result

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_results = self.pose.process(img_rgb)
        detected = mp_results.pose_landmarks is not None
        body_img = render_body_smpl(img_rgb, mp_results)

        result = (body_img, detected)
        with self.cache_lock:
            self.pose_cache.put(key, result)
        return result

    def _letterbox(self, img_rgb, tw, th):
        """Resize preserving aspect ratio and pad to (tw x th) with dark fill."""
        h, w = img_rgb.shape[:2]
        scale = min(tw / w, th / h)
        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_AREA)
        out = np.full((th, tw, 3), (15, 15, 30), dtype=np.uint8)
        y0 = (th - nh) // 2
        x0 = (tw - nw) // 2
        out[y0:y0 + nh, x0:x0 + nw] = resized
        return out

    # ------------------------------------------------------------------ UI build

    def _build_ui(self):
        # ---- header ----
        hdr = tk.Frame(self.root, bg='#1e1e2e')
        hdr.pack(fill='x', padx=20, pady=(12, 4))

        tk.Label(hdr, text=self.mode_cfg['question'],
                 font=('Helvetica', 18, 'bold'),
                 bg='#1e1e2e', fg='#cdd6f4').pack(side='left')

        self.class_label = tk.Label(hdr, text="",
                                    font=('Helvetica', 13, 'italic'),
                                    bg='#1e1e2e', fg='#89b4fa')
        self.class_label.pack(side='right')

        # ---- progress ----
        prog_frame = tk.Frame(self.root, bg='#1e1e2e')
        prog_frame.pack(fill='x', padx=20, pady=4)

        self.progress_var = tk.DoubleVar(value=self.completed)
        bar = ttk.Progressbar(prog_frame, variable=self.progress_var,
                               maximum=self.target)
        bar.pack(side='left', fill='x', expand=True)

        self.progress_label = tk.Label(prog_frame,
                                       text=f"{self.completed}/{self.target}",
                                       font=('Helvetica', 11),
                                       bg='#1e1e2e', fg='#a6e3a1', width=12)
        self.progress_label.pack(side='right')

        tk.Label(self.root, text=self.mode_cfg['hint'],
                 font=('Helvetica', 10, 'italic'),
                 bg='#1e1e2e', fg='#6c7086').pack(pady=(0, 4))

        # ---- comparison panels ----
        panels_frame = tk.Frame(self.root, bg='#1e1e2e')
        panels_frame.pack(fill='both', expand=True, padx=15, pady=8)

        self.left_panel = self._make_side_panel(panels_frame, 'A')
        self.left_panel['frame'].pack(side='left', padx=6)

        # VS divider
        vs_col = tk.Frame(panels_frame, bg='#1e1e2e', width=50)
        vs_col.pack(side='left', fill='y')
        vs_col.pack_propagate(False)
        tk.Label(vs_col, text="VS", font=('Helvetica', 22, 'bold'),
                 bg='#1e1e2e', fg='#f38ba8').place(relx=0.5, rely=0.5, anchor='center')

        self.right_panel = self._make_side_panel(panels_frame, 'B')
        self.right_panel['frame'].pack(side='left', padx=6)

        # ---- buttons ----
        btn_frame = tk.Frame(self.root, bg='#1e1e2e')
        btn_frame.pack(pady=12)

        self.btn_a = tk.Button(
            btn_frame, text="◀  Choose A   [←]",
            font=('Helvetica', 13, 'bold'),
            bg='#89b4fa', fg='#1e1e2e', activebackground='#74c7ec',
            bd=0, padx=22, pady=12, cursor='hand2',
            command=lambda: self._choose('A')
        )
        self.btn_a.pack(side='left', padx=12)

        self.btn_skip = tk.Button(
            btn_frame, text="Skip  [Space]",
            font=('Helvetica', 11),
            bg='#45475a', fg='#cdd6f4', activebackground='#585b70',
            bd=0, padx=16, pady=12, cursor='hand2',
            command=self._skip
        )
        self.btn_skip.pack(side='left', padx=12)

        self.btn_b = tk.Button(
            btn_frame, text="[→]   Choose B  ▶",
            font=('Helvetica', 13, 'bold'),
            bg='#a6e3a1', fg='#1e1e2e', activebackground='#94e2d5',
            bd=0, padx=22, pady=12, cursor='hand2',
            command=lambda: self._choose('B')
        )
        self.btn_b.pack(side='left', padx=12)

        # ---- status bar ----
        self.status_var = tk.StringVar(value="Ready.")
        tk.Label(self.root, textvariable=self.status_var,
                 font=('Helvetica', 9), bg='#181825', fg='#6c7086',
                 anchor='w').pack(fill='x', side='bottom')

        # ---- key bindings ----
        self.root.bind('<Left>', lambda _: self._choose('A'))
        self.root.bind('<Right>', lambda _: self._choose('B'))
        self.root.bind('<space>', lambda _: self._skip())
        self.root.bind('<Escape>', lambda _: self._confirm_quit())
        self.root.protocol('WM_DELETE_WINDOW', self._confirm_quit)

        self._disable_buttons()

    def _make_side_panel(self, parent, label):
        frame = tk.Frame(parent, bg='#181825', pady=6)

        title = tk.Label(frame, text=f"Pose {label}",
                         font=('Helvetica', 13, 'bold'),
                         bg='#181825', fg='#cdd6f4')
        title.pack()

        tk.Label(frame, text="Body Visualization",
                 font=('Helvetica', 8), bg='#181825', fg='#585b70').pack()

        skel_lbl = tk.Label(frame, bg='#0f0f1e',
                             width=DISPLAY_W, height=DISPLAY_H)
        skel_lbl.pack(padx=4, pady=(2, 6))

        tk.Label(frame, text="Original Image",
                 font=('Helvetica', 8), bg='#181825', fg='#585b70').pack()

        img_lbl = tk.Label(frame, bg='#0f0f1e',
                            width=DISPLAY_W, height=DISPLAY_H)
        img_lbl.pack(padx=4, pady=(2, 4))

        return {'frame': frame, 'skel': skel_lbl, 'img': img_lbl}

    # ------------------------------------------------------------------ image helpers

    def _photo(self, arr):
        return ImageTk.PhotoImage(Image.fromarray(arr))

    def _set_panel(self, panel, image_path):
        """Fill a panel's skeleton and image widgets. Returns detected bool."""
        raw = cv2.imread(str(image_path))
        if raw is None:
            blank = np.full((DISPLAY_H, DISPLAY_W, 3), (15, 15, 30), dtype=np.uint8)
            ph = self._photo(blank)
            panel['skel'].configure(image=ph); panel['skel'].image = ph
            panel['img'].configure(image=ph);  panel['img'].image = ph
            return False

        img_rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        skel_raw, detected = self._extract_pose(image_path)

        img_disp  = self._letterbox(img_rgb,  DISPLAY_W, DISPLAY_H)
        skel_disp = self._letterbox(skel_raw, DISPLAY_W, DISPLAY_H)

        ph_img  = self._photo(img_disp)
        ph_skel = self._photo(skel_disp)

        panel['skel'].configure(image=ph_skel); panel['skel'].image = ph_skel
        panel['img'].configure(image=ph_img);   panel['img'].image = ph_img
        return detected

    # ------------------------------------------------------------------ flow

    def _load_next_pair(self):
        if self.completed >= self.target:
            messagebox.showinfo("Complete!",
                                f"You've finished {self.completed} comparisons.\nGreat work!")
            self.root.quit()
            return

        if len(self.pair_queue) < 50:
            self._generate_pairs(200)

        self.current_pair = self.pair_queue.pop(0)
        cls, img_a, img_b = self.current_pair

        self.class_label.config(text=cls)
        self.status_var.set("Processing…  please wait")
        self._disable_buttons()
        self.root.update_idletasks()

        det_a = self._set_panel(self.left_panel,  img_a)
        det_b = self._set_panel(self.right_panel, img_b)

        notes = []
        if not det_a: notes.append("A: no pose detected")
        if not det_b: notes.append("B: no pose detected")
        status = "  |  ".join(notes) if notes else \
                 f"Completed: {self.completed}   Skipped: {self.skipped}"
        self.status_var.set(status)

        self._enable_buttons()
        self._update_progress()

        # Pre-fetch next pair in background
        self._start_prefetch()

    def _start_prefetch(self):
        if not self.pair_queue:
            return
        _, next_a, next_b = self.pair_queue[0]

        def prefetch():
            self._extract_pose(next_a)
            self._extract_pose(next_b)

        t = threading.Thread(target=prefetch, daemon=True)
        t.start()

    def _choose(self, side):
        if self.current_pair is None:
            return
        cls, img_a, img_b = self.current_pair
        winner, loser = (str(img_a), str(img_b)) if side == 'A' else (str(img_b), str(img_a))

        record = {
            "winner": winner,
            "loser":  loser,
            "class":  cls,
            "timestamp": datetime.now().isoformat()
        }
        with open(self.output_file, 'a') as f:
            f.write(json.dumps(record) + '\n')

        self.completed += 1
        self._load_next_pair()

    def _skip(self):
        if self.current_pair is None:
            return
        self.skipped += 1
        self._load_next_pair()

    # ------------------------------------------------------------------ helpers

    def _update_progress(self):
        self.progress_var.set(self.completed)
        self.progress_label.config(text=f"{self.completed}/{self.target}")

    def _disable_buttons(self):
        for w in (self.btn_a, self.btn_b, self.btn_skip):
            w.config(state='disabled')

    def _enable_buttons(self):
        for w in (self.btn_a, self.btn_b, self.btn_skip):
            w.config(state='normal')

    def _confirm_quit(self):
        if messagebox.askokcancel("Quit", f"Quit? Progress ({self.completed} comparisons) is saved."):
            self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description="Yoga pose preference labeler")
    parser.add_argument(
        '--mode', choices=list(MODES.keys()), default='visual',
        help=(
            'visual  → preferences.jsonl   (overall visual quality)\n'
            'missing → missing_joints.jsonl (joint completeness)\n'
            'quality → pose_quality.jsonl   (geometric joint alignment)'
        )
    )
    args = parser.parse_args()
    mode_cfg = MODES[args.mode]
    print(f"Mode    : {args.mode}")
    print(f"Output  : {mode_cfg['output']}")
    print(f"Target  : {mode_cfg['target']}")
    print(f"Hint    : {mode_cfg['hint']}")

    root = tk.Tk()
    root.geometry("950x870")
    app = PoseLabelingApp(root, mode_cfg)
    root.mainloop()


if __name__ == '__main__':
    main()
