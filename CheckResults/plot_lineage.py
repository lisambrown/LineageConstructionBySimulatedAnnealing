"""
plot_lineage.py  —  Visualize a cell lineage tree from SimGraph JSON files.

Usage:
    python plot_lineage.py path/to/sim_graph.json [sim_graph2.json ...]
    python plot_lineage.py path/to/SimGraphs/          # reads all *.json in dir
    python plot_lineage.py --config path/to/config.yaml

Output: saves lineage_tree.png (and shows interactively if a display is available).
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Graph loading
# ---------------------------------------------------------------------------

def load_simgraph(path):
    """Load one SimGraph or LineageGraph JSON → (nodes, forward_edges).

    Accepts two formats:
      SimGraph:     {"Nodes": [[frame, label], ...], "Edges": [[[f1,l1],[f2,l2]], ...]}
      LineageGraph: {"G_based_on_nn": {"Nodes": [{"Name":"FFF_LLL"}, ...],
                                       "Edges": [{"EndNodes":["FFF_LLL","FFF_LLL"]}, ...]}}

    Returns:
      nodes: list of (frame, label) tuples
      edges: list of ((frame_src, lbl_src), (frame_dst, lbl_dst)) forward edges
    """
    with open(path) as f:
        d = json.load(f)

    # Unwrap optional top-level wrapper (LineageGraph format)
    if "G_based_on_nn" in d:
        d = d["G_based_on_nn"]

    raw_nodes = d["Nodes"]
    raw_edges = d["Edges"]

    # Detect format by inspecting first node
    if raw_nodes and isinstance(raw_nodes[0], dict):
        # LineageGraph format: {"Name": "FFF_LLL"}
        def parse_name(s):
            return (int(s[:3]), int(s[-3:]))
        nodes = [parse_name(n["Name"]) for n in raw_nodes]
        edges = []
        for e in raw_edges:
            s = parse_name(e["EndNodes"][0])
            t = parse_name(e["EndNodes"][1])
            if t[0] > s[0]:
                edges.append((s, t))
            elif s[0] > t[0]:
                edges.append((t, s))
    else:
        # SimGraph format: nodes are [frame, label] lists
        nodes = [tuple(n) for n in raw_nodes]
        edges = [
            (tuple(s), tuple(t))
            for s, t in raw_edges
            if t[0] > s[0]
        ]

    return nodes, edges


def load_all(paths):
    """Merge multiple SimGraph files into one node/edge set."""
    all_nodes = set()
    all_edges = set()
    for p in paths:
        nodes, edges = load_simgraph(p)
        all_nodes.update(nodes)
        all_edges.update(edges)
    return sorted(all_nodes), sorted(all_edges)


# ---------------------------------------------------------------------------
# Tree layout
# ---------------------------------------------------------------------------

def build_layout(nodes, edges, lineage_gap=1):
    """Assign (x=frame, y=lineage_position) to every node.

    Each root's subtree is laid out as a contiguous band so lineages
    never overlap. A gap of `lineage_gap` rows is inserted between bands.

    Algorithm:
      1. Within each root's subtree, leaves get sequential y values.
      2. Internal nodes get y = mean of children's y values.
      3. Children are sorted by label for a deterministic, stable layout.
    """
    children = defaultdict(list)
    parents  = defaultdict(list)
    for src, dst in edges:
        children[src].append(dst)
        parents[dst].append(src)

    # Sort children by label so the tree is deterministically ordered.
    for n in list(children):
        children[n].sort()

    all_frames = sorted(set(n[0] for n in nodes))
    min_frame  = all_frames[0]

    # Roots: nodes with no parent.
    roots_unordered = sorted(n for n in nodes if not parents[n])

    def first_div_frame(root):
        """Earliest frame in root's subtree where a cell divides (≥2 children)."""
        stack = [root]
        earliest = float('inf')
        while stack:
            node = stack.pop()
            if len(children[node]) >= 2:
                earliest = min(earliest, node[0])
            stack.extend(children[node])
        return earliest

    # Process latest-dividers first (→ low y = bottom), earliest-dividers last (→ high y = top).
    # Never-dividing cells (inf) go to the very bottom.
    roots = sorted(roots_unordered, key=lambda r: (-first_div_frame(r), r))

    y_pos        = {}
    leaf_counter = [0.0]

    def assign_y(node):
        if node in y_pos:
            return y_pos[node]
        kids = children[node]
        if not kids:
            y_pos[node] = leaf_counter[0]
            leaf_counter[0] += 1.0
        else:
            kid_ys = [assign_y(k) for k in kids]
            y_pos[node] = float(np.mean(kid_ys))
        return y_pos[node]

    for root in roots:
        assign_y(root)
        leaf_counter[0] += lineage_gap   # gap between root lineages

    # Any orphan nodes not reachable from any root
    for node in sorted(nodes):
        if node not in y_pos:
            assign_y(node)
            leaf_counter[0] += lineage_gap

    return y_pos, children, parents, roots


def assign_root_colors(roots, children, cmap_name="tab20"):
    """Give each root a distinct color; descendants inherit it."""
    cmap = plt.get_cmap(cmap_name)
    n = max(len(roots), 1)
    root_color = {r: cmap(i / n) for i, r in enumerate(roots)}

    color = {}
    def propagate(node, c):
        color[node] = c
        for kid in children[node]:
            propagate(kid, c)

    for r, c in root_color.items():
        propagate(r, c)

    return color


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw_lineage(nodes, edges, y_pos, children, parents, root_colors,
                 ax, min_frame):
    """Draw horizontal cell segments and vertical division lines."""
    drawn_edges = set()

    for node in nodes:
        frame, label = node
        x = frame - min_frame      # shift so x starts at 0
        y = y_pos[node]
        color = root_colors.get(node, "gray")
        kids = children[node]

        # How far does this cell persist before splitting/ending?
        if kids:
            # extends to the children's frame
            x_end = kids[0][0] - min_frame
        else:
            x_end = x + 1         # leaf: show a unit-length stub

        # Horizontal bar for cell lifetime
        ax.plot([x, x_end], [y, y], color=color, linewidth=2.5,
                solid_capstyle="butt")

        # Dot at birth (except roots which start at frame 0)
        if not parents[node]:
            ax.plot(x, y, "o", color=color, markersize=5, zorder=3)

        # Division: vertical line connecting parent to two children
        if len(kids) == 2:
            y_kids = [y_pos[k] for k in kids]
            ax.plot([x_end, x_end], [min(y_kids), max(y_kids)],
                    color=color, linewidth=1.5, zorder=2)
            for kid in kids:
                ax.plot(x_end, y_pos[kid], "o", color=color,
                        markersize=4, zorder=3)
        elif len(kids) == 1:
            # Straight continuation — nothing extra needed
            pass
        elif len(kids) > 2:
            # Multi-way split (unusual) — same treatment
            y_kids = [y_pos[k] for k in kids]
            ax.plot([x_end, x_end], [min(y_kids), max(y_kids)],
                    color=color, linewidth=1.5, zorder=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def collect_json_files(paths):
    files = []
    for p in paths:
        pp = Path(p)
        if pp.is_dir():
            files.extend(sorted(pp.glob("sim_graph*.json")))
        elif pp.suffix == ".json":
            files.append(pp)
        else:
            print(f"Skipping {p} (not a JSON file or directory)", file=sys.stderr)
    return files


def infer_config_name(files):
    """Derive a config name from the input file paths.

    Uses the parent directory of the first file; if that directory is named
    'SimGraphs' (case-insensitive), goes up one more level.
    """
    parent = Path(files[0]).resolve().parent
    if parent.name.lower() == "simgraphs":
        parent = parent.parent
    return parent.name


def main():
    parser = argparse.ArgumentParser(description="Plot cell lineage from SimGraph JSON.")
    parser.add_argument("inputs", nargs="+",
                        help="SimGraph JSON file(s) or directory containing them")
    parser.add_argument("-o", "--output", default="lineage_tree.png",
                        help="Output image path (default: lineage_tree.png)")
    parser.add_argument("--title", default=None,
                        help="Plot title (default: derived from input directory)")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--no-show", action="store_true",
                        help="Don't open interactive window")
    args = parser.parse_args()

    files = collect_json_files(args.inputs)
    if not files:
        sys.exit("No SimGraph JSON files found.")
    print(f"Loading {len(files)} file(s)...")

    config = infer_config_name(files)
    title  = args.title if args.title else f"Cell Lineage — {config}"

    nodes, edges = load_all(files)
    print(f"  {len(nodes)} nodes, {len(edges)} forward edges")

    if not nodes:
        sys.exit("No nodes found.")

    y_pos, children, parents, roots = build_layout(nodes, edges)
    root_colors = assign_root_colors(roots, children)

    min_frame = min(n[0] for n in nodes)
    max_frame = max(n[0] for n in nodes)
    n_frames   = max_frame - min_frame + 1
    y_vals     = list(y_pos.values())
    y_span     = max(y_vals) - min(y_vals) if len(y_vals) > 1 else 1

    # Figure sizing: scale with number of frames and leaf count
    n_leaves = sum(1 for n in nodes if not children[n])
    fig_w = max(10, n_frames * 0.5)
    fig_h = max(6,  n_leaves * 0.22)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    draw_lineage(nodes, edges, y_pos, children, parents, root_colors,
                 ax, min_frame)

    # Axes
    ax.set_xlabel("Frame", fontsize=11)
    ax.set_xlim(-0.5, n_frames + 0.5)
    # Thin out x-ticks so labels don't crowd: aim for at most ~25 ticks
    tick_step = max(1, int(np.ceil(n_frames / 25)))
    ticks = list(range(0, n_frames + 1, tick_step))
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(min_frame + i) for i in ticks],
                        rotation=45, ha="right", fontsize=8)
    ax.set_yticks([])
    ax.set_ylim(min(y_vals) - 1, max(y_vals) + 1)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.spines[["top", "right", "left"]].set_visible(False)

    # Cells-per-frame annotation along top — only when count changes
    frame_counts = defaultdict(int)
    for node in nodes:
        frame_counts[node[0]] += 1
    sorted_frames = sorted(frame_counts)
    prev_count = None
    for frame in sorted_frames:
        count = frame_counts[frame]
        if count != prev_count:
            ax.text(frame - min_frame, max(y_vals) + 0.5, str(count),
                    ha="center", va="bottom", fontsize=7, color="gray")
            # Draw a faint vertical rule at each count-change frame
            ax.axvline(frame - min_frame, color="lightgray", linewidth=0.5,
                       linestyle="--", zorder=0)
            prev_count = count

    plt.tight_layout()
    out = args.output
    plt.savefig(out, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved → {out}")

    if not args.no_show:
        try:
            plt.show()
        except Exception:
            pass


if __name__ == "__main__":
    main()
