"""
check_splits.py  —  Visualize centroid associations at division frames.

For every frame where a cell divides, produces a 3D scatter figure showing
centroids of both frames with lineage edges drawn between matched cells.

Usage:
    python check_splits.py sim_graph.json Features.json
    python check_splits.py sim_graph.json Features.json -o output_dir/
    python check_splits.py sim_graph.json Features.json --frames 24 65 70
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_simgraph(path):
    with open(path) as f:
        d = json.load(f)
    nodes = [tuple(n) for n in d["Nodes"]]
    edges = [(tuple(s), tuple(t)) for s, t in d["Edges"] if t[0] > s[0]]
    return nodes, edges


def load_features(path):
    with open(path) as f:
        d = json.load(f)
    return d["centroids"]   # list[feat_idx][label_idx] -> [x, y, z]


# ---------------------------------------------------------------------------
# Frame offset detection
# ---------------------------------------------------------------------------

def detect_offset(nodes, centroids):
    """Find feat_offset such that centroids[frame - feat_offset] matches the
    cell count in the SimGraph at that frame.

    Tries offset = 0 and offset = min_simgraph_frame; picks whichever gives
    a better match at the first three SimGraph frames.
    """
    from collections import Counter
    frame_counts = Counter(n[0] for n in nodes)
    min_frame = min(frame_counts)
    check_frames = sorted(frame_counts)[:3]

    def score(offset):
        s = 0
        for f in check_frames:
            idx = f - offset
            if 0 <= idx < len(centroids):
                nz = sum(1 for c in centroids[idx] if c != [0, 0, 0])
                s += abs(nz - frame_counts[f])
        return s

    offset0 = score(0)
    offset_min = score(min_frame)
    chosen = 0 if offset0 <= offset_min else min_frame
    print(f"  Frame offset: {chosen}  (score@0={offset0}, score@{min_frame}={offset_min})")
    return chosen


# ---------------------------------------------------------------------------
# Division-frame detection
# ---------------------------------------------------------------------------

def find_division_frames(edges):
    """Return sorted list of frames where at least one parent divides (≥2 children)."""
    children = defaultdict(list)
    for src, dst in edges:
        children[src].append(dst)
    div_frames = set()
    for node, kids in children.items():
        if len(kids) >= 2:
            div_frames.add(node[0])
    return sorted(div_frames)


# ---------------------------------------------------------------------------
# Per-frame centroid lookup
# ---------------------------------------------------------------------------

def frame_centroids(centroids, feat_idx):
    """Return dict {label: np.array([x,y,z])} for a given feature index."""
    if feat_idx < 0 or feat_idx >= len(centroids):
        return {}
    result = {}
    for i, c in enumerate(centroids[feat_idx]):
        if c != [0, 0, 0]:
            result[i + 1] = np.array(c, dtype=float)
    return result


# ---------------------------------------------------------------------------
# Figure for one division event
# ---------------------------------------------------------------------------

def plot_division(frame_t, edges_t, centroids, feat_offset, out_path, config_name):
    frame_t1 = frame_t + 1

    pts_t  = frame_centroids(centroids, frame_t  - feat_offset)
    pts_t1 = frame_centroids(centroids, frame_t1 - feat_offset)

    if not pts_t or not pts_t1:
        print(f"  Skipping frame {frame_t}: missing centroid data", file=sys.stderr)
        return

    # Count divisions in this transition
    parent_kids = defaultdict(list)
    for src, dst in edges_t:
        parent_kids[src[1]].append(dst[1])
    n_div = sum(1 for kids in parent_kids.values() if len(kids) >= 2)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    # Frame t — blue
    coords_t = np.array(list(pts_t.values()))
    ax.scatter(coords_t[:, 0], coords_t[:, 1], coords_t[:, 2],
               c="steelblue", s=80, zorder=5, label=f"Frame {frame_t}", depthshade=False)
    for lbl, pt in pts_t.items():
        ax.text(pt[0], pt[1], pt[2], f" {lbl}", fontsize=7, color="steelblue")

    # Frame t+1 — red
    coords_t1 = np.array(list(pts_t1.values()))
    ax.scatter(coords_t1[:, 0], coords_t1[:, 1], coords_t1[:, 2],
               c="tomato", s=80, zorder=5, label=f"Frame {frame_t1}", depthshade=False)
    for lbl, pt in pts_t1.items():
        ax.text(pt[0], pt[1], pt[2], f" {lbl}", fontsize=7, color="tomato")

    # Lineage edges — thin black lines; thicker for division edges
    for src, dst in edges_t:
        p1 = pts_t.get(src[1])
        p2 = pts_t1.get(dst[1])
        if p1 is None or p2 is None:
            continue
        is_div = len(parent_kids[src[1]]) >= 2
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                color="black" if not is_div else "darkorange",
                linewidth=2.0 if is_div else 0.8,
                alpha=0.8 if is_div else 0.4)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    div_str = f"{n_div} division{'s' if n_div != 1 else ''}"
    ax.set_title(f"{config_name}  —  Frame {frame_t} → {frame_t1}  ({div_str})",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")

    plt.tight_layout()
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
        print(f"  Saved → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def infer_config_name(sim_path):
    p = Path(sim_path).resolve().parent
    return p.parent.name if p.name.lower() == "simgraphs" else p.name


def main():
    parser = argparse.ArgumentParser(
        description="Plot centroid associations at division frames.")
    parser.add_argument("simgraph",  help="SimGraph JSON file")
    parser.add_argument("features",  help="Features.json file")
    parser.add_argument("-o", "--output", default=None,
                        help="Save PNGs to this directory instead of showing interactively")
    parser.add_argument("--frames", nargs="+", type=int, default=None,
                        help="Only plot specific division frames (default: all)")
    args = parser.parse_args()

    config = infer_config_name(args.simgraph)
    print(f"Config: {config}")

    nodes, edges = load_simgraph(args.simgraph)
    print(f"SimGraph: {len(nodes)} nodes, {len(edges)} forward edges")

    centroids = load_features(args.features)
    print(f"Features: {len(centroids)} frames × {len(centroids[0])} max labels")

    feat_offset = detect_offset(nodes, centroids)

    div_frames = find_division_frames(edges)
    print(f"Division frames: {div_frames}")

    if args.frames:
        div_frames = [f for f in div_frames if f in args.frames]
        print(f"  (filtered to: {div_frames})")

    save_dir = Path(args.output) if args.output else None
    if save_dir:
        print(f"Saving to: {save_dir}")
    else:
        print("Mode: interactive (close each window to advance)")

    # Build per-frame edge lookup
    edges_by_frame = defaultdict(list)
    for src, dst in edges:
        edges_by_frame[src[0]].append((src, dst))

    for frame_t in div_frames:
        out_path = (save_dir / f"splits_frame{frame_t:05d}.png") if save_dir else None
        plot_division(frame_t, edges_by_frame[frame_t], centroids,
                      feat_offset, out_path, config)

    if save_dir:
        print("Done.")
    else:
        plt.show()


if __name__ == "__main__":
    main()
