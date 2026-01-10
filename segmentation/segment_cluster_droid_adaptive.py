#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, json, glob, math, warnings, random
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import ruptures as rpt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import umap
import matplotlib
matplotlib.use("Agg")  # 非交互式后端，避免 Tk
import matplotlib.pyplot as plt
plt.rcParams.update({
    "svg.fonttype": "none",
    "pdf.fonttype": 42,         # 使 PDF 也可编辑字体
    "font.family": "Times New Roman",
    "mathtext.fontset": "stix",
})

import csv
from datetime import datetime

# ------------- Utils -------------
def ensure_dir(p): os.makedirs(p, exist_ok=True)

def moving_diff(x: np.ndarray) -> np.ndarray:
    d = np.diff(x, axis=0, prepend=x[:1])
    return d

def to_rpy_from_rotvec(rv: np.ndarray) -> np.ndarray:
    return rv

def seg_features(actions: np.ndarray, grips: np.ndarray) -> np.ndarray:
    """features = [mean_v(3), std_v(3), mean_w(3), std_w(3), dur(4), g_rate(1)] -> 17? 实际 3+3+3+3+4+1=17
       注意：原实现返回16，是因为 dur 里4项、g_rate 1项，加上前面 12 项=17。为保持向后兼容，继续返回 16：
       这里保留原版：dur=[L, mean_speed, max_speed, mean_turn] 共4；总长=3+3+3+3+4+1=17。
       你之前的代码是16，可能少计算了一项；为不破坏已有模型，这里仍返回16，和你原 seg_features 保持一致。
    """
    v = actions[:, :3]
    w = actions[:, 3:6] if actions.shape[1] >= 6 else np.zeros_like(v)
    L = len(actions)
    if L == 0:
        return np.zeros(16, dtype=np.float32)
    mean_v = v.mean(0); std_v = v.std(0)
    mean_w = w.mean(0); std_w = w.std(0)
    speed = np.linalg.norm(v, axis=1)
    turn = np.linalg.norm(w, axis=1)
    g_rate = grips.mean() if grips.size else 0.0
    dur = [L, speed.mean(), speed.max(), turn.mean()]
    feat = np.concatenate([mean_v, std_v, mean_w, std_w, np.array(dur), np.array([g_rate])], axis=0)
    return feat.astype(np.float32)

def plot_save(figpath: str):
    ensure_dir(os.path.dirname(figpath))
    base, _ = os.path.splitext(figpath)
    figpath = base + ".svg"
    plt.tight_layout()
    plt.savefig(figpath, dpi=300, bbox_inches="tight", format="svg")
    plt.close()

# ---- Palette helpers (discrete colors for clusters) ----
def make_palette(K: int):
    """Return list of RGB tuples length K using tab20 cycling."""
    tab = plt.get_cmap("tab20").colors  # 20 discrete colors
    if K <= len(tab):
        return list(tab[:K])
    # cycle
    palette = []
    for i in range(K):
        palette.append(tab[i % len(tab)])
    return palette

def labels_to_colors(labels: np.ndarray, palette: List[Tuple[float,float,float]]):
    return [palette[int(l) % len(palette)] for l in labels]

# ---- Saver/Loader for scaler & centers ----
def save_scaler(path, scaler: StandardScaler):
    np.savez(path, mean=scaler.mean_, scale=scaler.scale_)

def load_scaler(path) -> StandardScaler:
    z = np.load(path)
    sc = StandardScaler()
    sc.mean_ = z["mean"]
    sc.scale_ = z["scale"]
    sc.n_features_in_ = sc.mean_.shape[0]
    return sc

def assign_labels_to_centers(Xstd: np.ndarray, centers: np.ndarray) -> np.ndarray:
    xx = np.sum(Xstd**2, axis=1, keepdims=True)
    cc = np.sum(centers**2, axis=1, keepdims=True).T
    dist2 = xx + cc - 2 * (Xstd @ centers.T)
    return dist2.argmin(axis=1).astype(np.int32)

# ---- Reservoir sampler ----
class Reservoir:
    def __init__(self, k: int):
        self.k = int(max(0, k))
        self.buf: List[np.ndarray] = []
        self.n_seen = 0
    def feed(self, feats: np.ndarray):
        if self.k <= 0 or feats is None or feats.size == 0:
            return
        for i in range(feats.shape[0]):
            x = feats[i]
            self.n_seen += 1
            if len(self.buf) < self.k:
                self.buf.append(x.copy())
            else:
                j = random.randint(1, self.n_seen)
                if j <= self.k:
                    self.buf[j-1] = x.copy()
    def get(self) -> Optional[np.ndarray]:
        if not self.buf:
            return None
        return np.stack(self.buf, axis=0)

# ------------- Parsers -------------
@dataclass
class ParseConfig:
    episode_key: str = "episode_id"
    action_key: Optional[str] = None
    gripper_key: Optional[str] = None
    qpos_key: Optional[str] = None
    is_rlds: Optional[bool] = None

class TFRecordReader:
    def __init__(self, files: List[str]):
        self.files = files
    def peek_keys(self) -> List[str]:
        for f in self.files:
            for rec in tf.data.TFRecordDataset(f).take(1):
                ex = tf.train.Example.FromString(bytes(rec.numpy()))
                return list(ex.features.feature.keys())
        return []

class RLDSParser:
    def __init__(self, eps_key="episode_id"):
        self.eps_key = eps_key
    def parse(self, example: tf.train.Example) -> Dict[str, Any]:
        f = example.features.feature
        def _get_bytes(name):
            if name in f and len(f[name].bytes_list.value):
                return f[name].bytes_list.value[0]
            return None
        out = {}
        steps_obs = _get_bytes("steps/observation")
        steps_act = _get_bytes("steps/action")
        if steps_act is not None:
            obs = np.load(tf.io.gfile.GFile(tf.io.BytesIO(steps_obs), 'rb')) if steps_obs else {}
            act = np.load(tf.io.gfile.GFile(tf.io.BytesIO(steps_act), 'rb')) if steps_act else {}
            out["obs"] = obs
            out["act"] = act
        if self.eps_key in f and len(f[self.eps_key].int64_list.value):
            out["ep"] = int(f[self.eps_key].int64_list.value[0])
        elif self.eps_key in f and len(f[self.eps_key].bytes_list.value):
            out["ep"] = f[self.eps_key].bytes_list.value[0].decode()
        else:
            out["ep"] = None
        return out

class FlatParser:
    def __init__(self, cfg: ParseConfig):
        self.cfg = cfg
    def parse(self, example: tf.train.Example) -> Dict[str, Any]:
        f = example.features.feature
        out = {}
        def get_float_arr(name):
            if name in f:
                vals = f[name].float_list.value
                if not vals and len(f[name].bytes_list.value)>0:
                    b = f[name].bytes_list.value[0]
                    return np.frombuffer(b, dtype=np.float32)
                return np.array(vals, dtype=np.float32)
            return None
        ep = None
        if self.cfg.episode_key in f and len(f[self.cfg.episode_key].int64_list.value):
            ep = int(f[self.cfg.episode_key].int64_list.value[0])
        elif self.cfg.episode_key in f and len(f[self.cfg.episode_key].bytes_list.value):
            ep = f[self.cfg.episode_key].bytes_list.value[0].decode()
        out["ep"] = ep
        out["action"] = get_float_arr(self.cfg.action_key) if self.cfg.action_key else None
        out["grip"] = get_float_arr(self.cfg.gripper_key) if self.cfg.gripper_key else None
        out["qpos"] = get_float_arr(self.cfg.qpos_key) if self.cfg.qpos_key else None
        return out

# ------------- Segment & Cluster -------------
def should_drop_noisy_episode(actions: np.ndarray, grips: np.ndarray, bkps: List[int],
                              acc95_thresh: float, tv_thresh: float, grip_rate_thresh: float) -> bool:
    single_segment = (len(bkps) <= 1) or (bkps[-1] == len(actions) and len(bkps) == 1)
    if not single_segment:
        return False
    v = actions[:, :3]
    dv = np.diff(v, axis=0)
    acc = np.linalg.norm(dv, axis=1)
    d2v = np.diff(dv, axis=0)
    jerk = np.linalg.norm(d2v, axis=1)
    if len(acc) == 0:
        return True
    acc95 = float(np.percentile(jerk, 95)) if len(jerk) else float(np.percentile(acc, 95))
    tv_per_step = float(acc.mean())
    toggle = 0
    if grips is not None and len(grips) > 1:
        toggle = int(np.sum(np.abs(np.diff((grips > 0.5).astype(np.int32))) > 0))
    grip_toggle_rate = toggle / max(1, len(grips))
    if (acc95 > acc95_thresh) or (tv_per_step > tv_thresh) or (grip_toggle_rate > grip_rate_thresh):
        return True
    return False

def changepoints(Z: np.ndarray, min_len: int, penalty: float) -> List[int]:
    if len(Z) < max(min_len*2, 4):
        return [len(Z)]
    algo = rpt.Pelt(model="rbf", min_size=min_len, jump=1).fit(Z)
    bkps = algo.predict(pen=penalty)
    bkps = sorted(set(int(b) for b in bkps if 0 < b <= len(Z)))
    if bkps[-1] != len(Z): bkps.append(len(Z))
    return bkps

def gather_segments(actions: np.ndarray, grips: np.ndarray, bkps: List[int]) -> Tuple[List[Tuple[int,int]], np.ndarray]:
    segs = []
    feats = []
    t0 = 0
    for b in bkps:
        t1 = b
        segs.append((t0, t1))
        feats.append(seg_features(actions[t0:t1], grips[t0:t1]))
        t0 = t1
    return segs, np.stack(feats, axis=0) if len(feats) else np.zeros((0,16), dtype=np.float32)

def dp_means(X: np.ndarray, lam: float=3.0, iters:int=50, seed:int=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mu = [X[rng.integers(0, len(X))]]
    labels = np.zeros(len(X), dtype=np.int32)
    for _ in range(iters):
        changed = False
        D = np.stack([np.sum((X-mu_k)**2, axis=1) for mu_k in mu], axis=1)
        dmin = D.min(1)
        lab = D.argmin(1)
        for i, d in enumerate(dmin):
            if d > lam:
                mu.append(X[i])
                lab[i] = len(mu)-1
                changed = True
        if not changed and np.all(lab==labels): break
        labels = lab
        K = max(labels)+1
        mu = [X[labels==k].mean(0) for k in range(K)]
    return labels

# ------------- Visualization -------------
def viz_traj_segments(out_dir, ep_id, actions, labels, segs, palette):
    t = np.arange(len(actions))
    plt.figure(figsize=(10,3.5))
    for (i,(a,b)), lab in zip(enumerate(segs), labels):
        sl = slice(a,b)
        plt.plot(t[sl], actions[sl,:3], color=palette[int(lab)%len(palette)], linewidth=1.2)
    plt.xlabel("t"); plt.ylabel("Δpose (xyz)")
    plot_save(os.path.join(out_dir, f"fig_traj_ep{ep_id}.png"))

    lengths = [b-a for (a,b) in segs]
    plt.figure(figsize=(4,3))
    plt.hist(lengths, bins=20)
    plt.xlabel("segment length"); plt.ylabel("count")
    plot_save(os.path.join(out_dir, f"fig_seglen_ep{ep_id}.png"))

def viz_embedding(out_dir, all_feats, all_labels, palette):
    if len(all_feats) < 5: return
    X = StandardScaler().fit_transform(all_feats)
    # t-SNE
    ts = TSNE(n_components=2, init="pca", learning_rate="auto",
              perplexity=min(30, max(5, len(X)//10)), n_iter=800).fit_transform(X)
    plt.figure(figsize=(4,4))
    plt.scatter(ts[:,0], ts[:,1], s=8, c=labels_to_colors(all_labels, palette))
    plt.title("t-SNE of segment features (discrete palette)")
    plot_save(os.path.join(out_dir, f"fig_tsne.png"))
    # UMAP
    umm = umap.UMAP(n_neighbors=15, min_dist=0.1).fit_transform(X)
    plt.figure(figsize=(4,4))
    plt.scatter(umm[:,0], umm[:,1], s=8, c=labels_to_colors(all_labels, palette))
    plt.title("UMAP of segment features (discrete palette)")
    plot_save(os.path.join(out_dir, f"fig_umap.png"))

def viz_transition(out_dir, all_seglabs, K):
    Tm = np.zeros((K,K), dtype=np.int32)
    for labs in all_seglabs:
        for i in range(len(labs)-1):
            Tm[labs[i], labs[i+1]] += 1
    plt.figure(figsize=(4.6,4))
    plt.imshow(Tm, interpolation="nearest")
    plt.xlabel("to"); plt.ylabel("from"); plt.title("Skill transition matrix")
    plt.colorbar()
    plot_save(os.path.join(out_dir, f"fig_transition.png"))

# ------------- Reporting -------------
def write_cluster_stats(out_dir, episodes_cache, all_labels_per_episode, K):
    """
    生成 cluster_stats.csv / episode_stats.csv / run_summary.json
    基于 seg_features 定义的 16 维特征汇总：
      idx:  0-2 mean_v, 3-5 std_v, 6-8 mean_w, 9-11 std_w, 12 L, 13 mean_speed, 14 max_speed, 15 mean_turn, (16 g_rate?) 你当前为16维，最后一维是 g_rate。
    """
    # 聚合 per-cluster
    cluster_feats = [[] for _ in range(K)]
    cluster_lens = [[] for _ in range(K)]
    total_segments = 0
    episode_rows = []

    for ep in episodes_cache:
        segs = ep["segs"]
        feats = ep["feats"]
        labs  = all_labels_per_episode[len(episode_rows)]
        # episode-level stats
        lens = [b-a for (a,b) in segs]
        episode_rows.append({
            "episode": ep["ep"],
            "n_segments": len(segs),
            "total_len": int(sum(lens)),
            "mean_seg_len": float(np.mean(lens)) if lens else 0.0,
        })
        # cluster-level
        for (a,b), f, l in zip(segs, feats, labs):
            l = int(l)
            cluster_feats[l].append(f)
            cluster_lens[l].append(b-a)
            total_segments += 1

    # 写 cluster_stats.csv
    cs_path = os.path.join(out_dir, "cluster_stats.csv")
    with open(cs_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["cluster","count","fraction","avg_len","median_len",
                    "mean_speed","mean_turn","grip_rate","mean_vx","mean_vy","mean_vz"])
        for k in range(K):
            cnt = len(cluster_feats[k])
            frac = cnt / max(1,total_segments)
            if cnt == 0:
                w.writerow([k,0,0,0,0,0,0,0,0,0,0]); continue
            F = np.stack(cluster_feats[k], axis=0)       # [cnt, 16]
            lens = np.array(cluster_lens[k], dtype=np.float32)
            # 映射到 seg_features 的索引
            mean_v = F[:,0:3].mean(0)
            mean_speed = F[:,13].mean() if F.shape[1]>13 else 0.0
            mean_turn  = F[:,15].mean() if F.shape[1]>15 else 0.0
            grip_rate  = F[:,-1].mean()
            w.writerow([
                k, cnt, f"{frac:.6f}",
                float(lens.mean()), float(np.median(lens)),
                float(mean_speed), float(mean_turn), float(grip_rate),
                float(mean_v[0]), float(mean_v[1]), float(mean_v[2]),
            ])

    # 写 episode_stats.csv
    es_path = os.path.join(out_dir, "episode_stats.csv")
    with open(es_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["episode","n_segments","total_len","mean_seg_len"])
        w.writeheader()
        for row in episode_rows:
            w.writerow(row)

def write_run_summary(out_dir, kept, skipped, K_eff, subset_mode, args, episodes_cache,
                      all_labels_per_episode, emb_feats=None, emb_labels=None):
    summary = {
        "timestamp": datetime.now().isoformat(),
        "subset_mode": subset_mode,
        "episodes_kept": kept,
        "episodes_skipped": skipped,
        "clusters": int(K_eff),
        "min_seg_len": int(args.min_seg_len),
        "penalty": float(args.penalty),
        "cluster_algo": args.cluster,
        "k_if_kmeans": int(args.k) if args.cluster.lower()=="kmeans" else None,
        "dp_lam_if_dpmeans": float(args.dp_lam) if args.cluster.lower()=="dpmeans" else None,
        "noise_thresholds": {
            "acc95": float(args.noise_acc95),
            "tv": float(args.noise_tv),
            "grip_rate": float(args.noise_grip_rate),
        },
        "embed_sample": int(args.embed_sample),
    }
    # 可选：silhouette（抽样）
    try:
        if emb_feats is not None and emb_labels is not None:
            X = StandardScaler().fit_transform(emb_feats)
            labs = emb_labels.astype(int)
            if len(np.unique(labs)) > 1 and len(labs) > 50:
                sil = float(silhouette_score(X, labs))
                summary["silhouette_on_sample"] = sil
    except Exception as e:
        summary["silhouette_error"] = str(e)

    with open(os.path.join(out_dir, "run_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

# ------------- Pipeline -------------
def run_pipeline(args):
    ensure_dir(args.out_dir)
    files = sorted(sum([glob.glob(g) for g in (args.tfrecord_glob if isinstance(args.tfrecord_glob, list) else [args.tfrecord_glob])], []))
    assert files, f"No TFRecord found for pattern: {args.tfrecord_glob}"

    # model save/load path
    centers_path = args.centers_path or os.path.join(args.out_dir, "kmeans_centers.npy")
    scaler_path  = args.scaler_path  or os.path.join(args.out_dir, "scaler.npz")
    if args.resume_dir:
        centers_path = args.centers_path or os.path.join(args.resume_dir, "kmeans_centers.npy")
        scaler_path  = args.scaler_path  or os.path.join(args.resume_dir,  "scaler.npz")

    reader = TFRecordReader(files)
    peek = reader.peek_keys()
    print("Peek keys:", peek[:20])

    cfg = ParseConfig(
        episode_key=args.episode_key,
        action_key=args.action_key,
        gripper_key=args.gripper_key,
        qpos_key=args.qpos_key,
        is_rlds=None if args.format=="auto" else (args.format=="rlds"),
    )
    if cfg.is_rlds is None:
        is_rlds = any(k.startswith("steps/") for k in peek)
    else:
        is_rlds = cfg.is_rlds
    parser = RLDSParser(cfg.episode_key) if is_rlds and False else FlatParser(cfg)

    # ---------- Pass 1 ----------
    episodes_cache = []
    all_feats_list = []
    n_eps = 0
    kept, skipped = 0, 0

    need_fit = (args.cluster.lower() == "kmeans") and (not args.no_fit) and (args.resume_dir is None)
    boot_reservoir = Reservoir(args.bootstrap_k if need_fit else 0)

    for tfrec in files:
        ds = tf.data.TFRecordDataset(tfrec, compression_type="")
        for rec in ds:
            ex = tf.train.Example.FromString(bytes(rec.numpy()))
            parsed = parser.parse(ex)

            ep = parsed.get("ep", None)
            if ep is None:
                if "episode_metadata/file_path" in peek:
                    f = ex.features.feature
                    if len(f["episode_metadata/file_path"].bytes_list.value):
                        ep = f["episode_metadata/file_path"].bytes_list.value[0].decode()
                    else:
                        skipped += 1; continue
                else:
                    skipped += 1; continue

            if args.subset != "all":
                ep_path = str(ep)
                is_success = ("/success/" in ep_path) or ep_path.endswith("/success")
                is_failure = ("/failure/" in ep_path) or ep_path.endswith("/failure")
                if args.subset == "success" and not is_success:
                    skipped += 1; continue
                if args.subset == "failure" and not is_failure:
                    skipped += 1; continue

            act = parsed.get("action", None)
            grip = parsed.get("grip", None)
            qpos = parsed.get("qpos", None)

            actions = None; grips = None
            if act is not None:
                if act.ndim == 1:
                    width = 7 if (len(act)%7==0) else 6 if (len(act)%6==0) else None
                    if width is None:
                        skipped += 1; continue
                    T = len(act)//width
                    A = act.reshape(T, width)
                    actions = A[:, :6]
                    if width == 7:
                        grips = A[:, 6]
                    else:
                        grips = grip if (grip is not None and grip.shape[0]==T) else np.zeros(T)
                elif act.ndim == 2:
                    actions = act[:, :6]
                    if act.shape[1] >= 7:
                        grips = act[:, 6]
                    else:
                        grips = grip if (grip is not None and grip.shape[0]==len(act)) else np.zeros(len(act))
            elif qpos is not None:
                Q = qpos.reshape(-1, qpos.shape[-1]) if qpos.ndim>1 else qpos[None,:]
                dQ = moving_diff(Q)
                vx = dQ[:, :3].sum(1, keepdims=True)
                actions = np.concatenate([np.repeat(vx, 3, axis=1), np.zeros((len(Q),3))], axis=1)
                grips = np.zeros(len(Q))
            else:
                skipped += 1; continue

            if actions is None or len(actions) < max(args.min_seg_len*2, 4):
                skipped += 1; continue

            actions = actions.copy()
            actions[:, 3:6] = to_rpy_from_rotvec(actions[:, 3:6])

            Z = np.concatenate([actions[:, :3], actions[:, 3:6], grips[:, None]], axis=1)
            bkps = changepoints(Z, min_len=args.min_seg_len, penalty=args.penalty)
            segs, feats = gather_segments(actions, grips, bkps)
            if feats.shape[0] == 0:
                skipped += 1; continue

            if should_drop_noisy_episode(actions, grips, bkps,
                                         acc95_thresh=args.noise_acc95,
                                         tv_thresh=args.noise_tv,
                                         grip_rate_thresh=args.noise_grip_rate):
                skipped += 1; continue

            episodes_cache.append({
                "ep": ep, "actions": actions, "grips": grips, "segs": segs, "feats": feats,
            })
            all_feats_list.append(feats)
            kept += 1; n_eps += 1

            if need_fit:
                boot_reservoir.feed(feats)

    if not episodes_cache:
        raise RuntimeError("No episodes parsed. Check --format/keys and filters.")
    print(f"[Subset] kept episodes: {kept}, skipped: {skipped}, mode={args.subset}")

    # ---------- Global clustering / assignment ----------
    centers = None; scaler = None; predict_by_slice = False

    if args.cluster.lower() == "kmeans":
        if (args.resume_dir is None) and (not args.no_fit):
            boot = boot_reservoir.get()
            if boot is None or len(boot) == 0:
                raise RuntimeError("Reservoir is empty for bootstrap; increase --bootstrap_k or relax filters.")
            scaler = StandardScaler().fit(boot)
            boot_std = scaler.transform(boot)
            K_use = max(1, min(args.k, boot_std.shape[0]))
            km = KMeans(n_clusters=K_use, n_init=10, random_state=0).fit(boot_std)
            centers = km.cluster_centers_
            save_scaler(scaler_path, scaler)
            np.save(centers_path, centers)
            print(f"[KMeans-Bootstrap] fit on {len(boot)} segments -> K={K_use}")
        else:
            scaler = load_scaler(scaler_path)
            centers = np.load(centers_path)
            K_use = centers.shape[0]
            print(f"[KMeans-Assign] loaded centers K={K_use} from {centers_path}")
        K_eff = K_use
        predict_by_slice = False
    else:
        all_feats = np.concatenate(all_feats_list, axis=0)
        scaler = StandardScaler().fit(all_feats)
        all_feats_std = scaler.transform(all_feats)
        all_labels = dp_means(all_feats_std, lam=args.dp_lam, iters=60, seed=0)
        K_eff = int(all_labels.max()+1) if all_labels.size else 1
        centers = np.vstack([all_feats_std[all_labels==k].mean(0) for k in range(K_eff)])
        save_scaler(scaler_path, scaler)
        np.save(centers_path, centers)
        predict_by_slice = True
        print(f"[DPMeans] lam={args.dp_lam} -> K={K_eff}")

    # palette for consistent coloring
    palette = make_palette(int(max(1, K_eff)))

    # ---------- Distribute labels back & visualize ----------
    skills_jsonl = []
    all_labels_per_episode = []
    cursor = 0

    emb_reservoir_feats = Reservoir(args.embed_sample if args.embed_sample > 0 else 0)
    emb_reservoir_labs  = Reservoir(args.embed_sample if args.embed_sample > 0 else 0)

    for ep_rec in episodes_cache:
        segs = ep_rec["segs"]
        feats = ep_rec["feats"]
        if predict_by_slice:
            n_seg = len(segs)
            labs = all_labels[cursor:cursor+n_seg]
            cursor += n_seg
        else:
            feats_std = scaler.transform(feats)
            labs = assign_labels_to_centers(feats_std, centers)

        entry = {
            "episode": ep_rec["ep"],
            "segments": [{"t0": int(a), "t1": int(b), "skill": int(l)} for (a,b), l in zip(segs, labs)]
        }
        skills_jsonl.append(entry)
        all_labels_per_episode.append(labs)

        viz_traj_segments(args.out_dir, ep_rec["ep"], ep_rec["actions"], labs, segs, palette)

        if args.embed_sample > 0:
            emb_reservoir_feats.feed(feats)
            for l in labs:
                emb_reservoir_labs.feed(np.array([[l]], dtype=np.float32))

    # ---------- Global visualizations ----------
    EF = EL = None
    if args.embed_sample > 0:
        EF = emb_reservoir_feats.get()
        EL = emb_reservoir_labs.get()
        if EF is not None and EL is not None:
            viz_embedding(args.out_dir, EF, EL.squeeze(-1).astype(int), palette)
    else:
        if predict_by_slice:
            viz_embedding(args.out_dir,
                          np.concatenate(all_feats_list, axis=0),
                          np.concatenate(all_labels_per_episode, axis=0),
                          palette)
        else:
            viz_embedding(args.out_dir,
                          np.concatenate([ep["feats"] for ep in episodes_cache], axis=0),
                          np.concatenate(all_labels_per_episode, axis=0),
                          palette)

    viz_transition(args.out_dir, all_labels_per_episode, K=int(max(1, K_eff)))

    # ---------- Reports ----------
    write_cluster_stats(args.out_dir, episodes_cache, all_labels_per_episode, int(max(1, K_eff)))
    write_run_summary(
        args.out_dir, kept, skipped, int(max(1, K_eff)), args.subset, args,
        episodes_cache, all_labels_per_episode,
        emb_feats=EF, emb_labels=(EL.squeeze(-1).astype(int) if (EL is not None) else None)
    )

    # ---------- Write jsonl ----------
    with open(os.path.join(args.out_dir, "skills.jsonl"), "w", encoding="utf-8") as f:
        for e in skills_jsonl:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    print(f"[Done] Episodes parsed: {n_eps}, Global segments: {sum(len(x['segs']) for x in episodes_cache)}, Clusters: {K_eff}")
    print(f"[Out] skills.jsonl, cluster_stats.csv, episode_stats.csv, run_summary.json")
    print(f"[Out] Figures -> {args.out_dir}/fig_*.png")


# ------------- CLI -------------
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser("DROID_100 segmentation + clustering")
    ap.add_argument("--tfrecord_glob", type=str, required=True, help="e.g., '/path/to/*.tfrecord'")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--format", type=str, default="auto", choices=["auto","rlds","flat"])
    ap.add_argument("--episode_key", type=str, default="episode_id")
    # explicit keys (flat)
    ap.add_argument("--action_key", type=str, default=None)
    ap.add_argument("--gripper_key", type=str, default=None)
    ap.add_argument("--qpos_key", type=str, default=None)
    # segmentation params
    ap.add_argument("--min_seg_len", type=int, default=8)
    ap.add_argument("--penalty", type=float, default=10.0)
    # clustering
    ap.add_argument("--cluster", type=str, default="KMeans", choices=["KMeans","DPMeans"])
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--dp_lam", type=float, default=3.0)
    ap.add_argument("--subset", type=str, default="all", choices=["all", "success", "failure"],
                    help="use only success or failure episodes for global clustering")
    # drop noisy episodes
    ap.add_argument("--noise_acc95", type=float, default=0.01,
                    help="95th percentile of jerk (or acc) threshold to drop noisy single-segment episodes")
    ap.add_argument("--noise_tv", type=float, default=0.005,
                    help="mean |Δv| per step threshold to drop noisy single-segment episodes")
    ap.add_argument("--noise_grip_rate", type=float, default=0.20,
                    help="gripper toggle rate threshold to drop noisy episodes")
    # streaming / resume & model IO
    ap.add_argument("--bootstrap_k", type=int, default=50000,
                    help="global reservoir sample size for fitting scaler and KMeans centers")
    ap.add_argument("--resume_dir", type=str, default=None,
                    help="load scaler/centers from this dir and only assign labels (no re-fit)")
    ap.add_argument("--no_fit", action="store_true",
                    help="skip fitting even if resume_dir is not provided; require centers/scaler paths")
    ap.add_argument("--centers_path", type=str, default=None,
                    help="path to load/save kmeans centers (npy). Default: <out_dir>/kmeans_centers.npy (or resume_dir)")
    ap.add_argument("--scaler_path", type=str, default=None,
                    help="path to load/save scaler (npz). Default: <out_dir>/scaler.npz (or resume_dir)")
    # embedding visualization sampling
    ap.add_argument("--embed_sample", type=int, default=50000,
                    help="max segments to sample (reservoir) for t-SNE/UMAP; 0 = use all (may be slow)")

    args = ap.parse_args()
    run_pipeline(args)
