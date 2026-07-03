"""
EvaluateLineage.py  —  Compare a result lineage against ground truth.

Loads the GT from Data/Mouse/{config}/GroundTruth/LineageGraph.json and
a result sim_graph from Data/Mouse/{config}/Results/, then prints the
cost breakdown and per-frame accuracy (1-to-1 matches, correct splits).
No simulated annealing is run.

Usage:
    python EvaluateLineage.py -c 220827_stack1 -r sim_graph_050_079.json
    python EvaluateLineage.py -c 220827_stack1 -r /full/path/to/sim_graph.json
"""

import copy
import json
import os
import yaml
import argparse
from collections import Counter

import sim_anneal
from NewCostFunction import MyCostParams, GeneralCostFunc
from EvaluateCost import EvaluateCost
from sim_anneal_utils import (
    GetNucleiPerFrame, tracksInitFromGT, tracksInitFromSimGraph, Eval
)

# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate a result lineage against ground truth.')
    parser.add_argument('-c', '--config_name', type=str, required=True,
                        help='configuration name (e.g. 220827_stack1)')
    parser.add_argument('-r', '--result_graph', type=str, required=True,
                        help='result sim_graph JSON filename or full path')
    args = parser.parse_args()

    config_name = args.config_name
    print('Config:', config_name)

    # ---------------------------------------------------------------------------
    # Paths
    # ---------------------------------------------------------------------------

    config_path = '../Config'

    full_config = os.path.join(config_path, config_name + '_config.yaml')
    with open(full_config) as f:
        config_opts = yaml.safe_load(f)

    data_path     = config_opts.get('data_path', os.path.join('../Data/Mouse', config_name))
    features_file = config_opts.get('features_file', 'Features.json')
    start_info    = config_opts.get('register_begin_frame', 0)

    out_path = os.path.join(data_path, 'Results')
    os.makedirs(out_path, exist_ok=True)

    # ---------------------------------------------------------------------------
    # Load features
    # ---------------------------------------------------------------------------

    feature_path = os.path.join(data_path, features_file)
    with open(feature_path) as f:
        data = json.load(f)
    data1 = data   # same file holds mean_intensity20, std_intensity, etc.

    # ---------------------------------------------------------------------------
    # Load result sim_graph → determine frame range
    # ---------------------------------------------------------------------------

    result_path = args.result_graph
    if not os.path.isabs(result_path):
        result_path = os.path.join(out_path, result_path)

    print('Loading result:', result_path)
    with open(result_path) as f:
        result_sg = json.load(f)

    result_frame_counts = Counter(n[0] for n in result_sg['Nodes'])
    result_frames  = sorted(result_frame_counts)
    start_frame    = result_frames[0]
    end_frame      = result_frames[-1]
    nframes        = end_frame - start_frame + 1
    result_nucCounts = [result_frame_counts[f] for f in result_frames]

    print(f'Result covers frames {start_frame}–{end_frame} ({nframes} frames)')
    print('Result cell counts:', result_nucCounts)

    # ---------------------------------------------------------------------------
    # Load ground-truth lineage, filter to result frame range
    # ---------------------------------------------------------------------------

    gt_path = os.path.join(data_path, 'GroundTruth', 'LineageGraph.json')
    with open(gt_path) as f:
        gt_raw = json.load(f)

    # Handle optional top-level wrapper key
    if 'G_based_on_nn' in gt_raw:
        gt_raw = gt_raw['G_based_on_nn']

    gt_nodes_all = gt_raw['Nodes']
    gt_edges_all = gt_raw['Edges']

    # Result sim_graph uses features-space frame numbers; GT lineage uses
    # GT-space frame numbers shifted by start_info (register_begin_frame).
    gt_start_frame = start_frame + start_info
    gt_end_frame   = end_frame   + start_info

    print(f'GT frame range (adjusted for register_begin_frame={start_info}): '
          f'{gt_start_frame}–{gt_end_frame}')

    gt_nodes = [n for n in gt_nodes_all
                if gt_start_frame <= int(n['Name'][:3]) <= gt_end_frame]
    gt_edges = [e for e in gt_edges_all
                if gt_start_frame <= int(e['EndNodes'][0][:3]) <= gt_end_frame
                and gt_start_frame <= int(e['EndNodes'][1][:3]) <= gt_end_frame]

    [gt_nucCounts, list_nuclei_labels] = GetNucleiPerFrame(
        gt_start_frame, gt_end_frame, gt_nodes, number_of_excludes_per_frame=0)

    print('GT cell counts:    ', gt_nucCounts)

    # Trim leading and trailing frames where GT has 0 cells (GT doesn't cover that range).
    skip = next((i for i, c in enumerate(gt_nucCounts) if c > 0), 0)
    trim = next((i for i, c in enumerate(reversed(gt_nucCounts)) if c > 0), 0)
    if skip > 0:
        print(f'Skipping {skip} leading frame(s) with 0 GT cells; '
              f'result start {start_frame}→{start_frame+skip}, '
              f'GT start {gt_start_frame}→{gt_start_frame+skip}')
    if trim > 0:
        print(f'Trimming {trim} trailing frame(s) with 0 GT cells; '
              f'result end {end_frame}→{end_frame-trim}, '
              f'GT end {gt_end_frame}→{gt_end_frame-trim}')
    if skip > 0 or trim > 0:
        gt_start_frame   += skip
        gt_end_frame     -= trim
        start_frame      += skip
        end_frame        -= trim
        nframes           = end_frame - start_frame + 1
        gt_nucCounts      = gt_nucCounts[skip: len(gt_nucCounts) - trim if trim else None]
        result_nucCounts  = result_nucCounts[skip: len(result_nucCounts) - trim if trim else None]
        result_frames     = result_frames[skip: len(result_frames) - trim if trim else None]

    # ---------------------------------------------------------------------------
    # Cost parameters (from config)
    # ---------------------------------------------------------------------------

    sa = config_opts['simulated_annealing']
    support_split = sa.get('support_split', True)

    # ---------------------------------------------------------------------------
    # Build GT tracks
    # ---------------------------------------------------------------------------

    gt_tracks = sim_anneal.Tracks(gt_nucCounts, supportSplit=support_split)
    tracksInitFromGT(gt_tracks, data, data1, gt_edges,
                     gt_start_frame, list_nuclei_labels, exclude_labels={}, start_info=start_info)
    gt_tracks.validate(computeKLeaf=support_split)

    # ---------------------------------------------------------------------------
    # Build result tracks
    # ---------------------------------------------------------------------------

    result_tracks = sim_anneal.Tracks(result_nucCounts, supportSplit=support_split)
    tracksInitFromSimGraph(result_tracks, data, data1,
                           result_sg['Nodes'], result_sg['Edges'],
                           start_frame, start_info=0)  # sim_graph stores features-space frames
    result_tracks.validate(computeKLeaf=support_split)
    splitWt         = sa['splitWt']
    mdDistWt        = sa['mdDistWt']
    symWt           = sa['symWt']
    angWt           = sa['angWt']
    aspWt           = sa['aspWt']
    meanIWt         = sa['meanIWt']
    stdIWt          = sa['stdIWt']
    centWt          = sa['centWt'];  centNoSplitWt  = sa['centNoSplitWt']
    centSplitMDWt   = sa['centSplitMDWt'];  centSplitDDWt = sa['centSplitDDWt']
    centSplitMDDist = sa['centSplitMDDist'];  centSplitDDDist = sa['centSplitDDDist']
    volWt           = sa['volWt'];   volNoSplitWt   = sa['volNoSplitWt']
    volSplitWt      = sa['volSplitWt']
    volNoSplitMult  = sa['volNoSplitMult'];  volSplitMult = sa['volSplitMult']

    MyCP = MyCostParams(
        mdDistWt, symWt, angWt, aspWt, meanIWt, stdIWt,
        centWt, centNoSplitWt, centSplitMDWt, centSplitDDWt,
        centSplitMDDist, centSplitDDDist,
        volWt, volNoSplitWt, volSplitWt, volNoSplitMult, volSplitMult,
        splitWt)

    # ---------------------------------------------------------------------------
    # Evaluate
    # ---------------------------------------------------------------------------

    print('\n--- Cost breakdown (Result vs GT) ---')
    print(f'{"":20s}  {"Result":>10s}  {"GT":>10s}')
    costs = EvaluateCost(result_tracks, gt_tracks, MyCP, splitWt)

    print('\n--- Per-frame accuracy ---')
    [total_good_1to1, total_1to1, total_good_splits, total_splits] = \
        Eval(nframes, gt_tracks, result_tracks, start_frame)

    print(f'\nSummary: 1-to-1 {total_good_1to1}/{total_1to1}  '
          f'splits {total_good_splits}/{total_splits}')

    # ---------------------------------------------------------------------------
    # Save evaluation summary
    # ---------------------------------------------------------------------------

    output = {
        'config_name':        config_name,
        'result_graph':       result_path,
        'start_frame':        start_frame,
        'end_frame':          end_frame,
        'Config_Params':      MyCP.__dict__,
        'costs':              costs,
        'total_good_1to1':    total_good_1to1,
        'total_1to1':         total_1to1,
        'total_good_splits':  total_good_splits,
        'total_splits':       total_splits,
    }

    result_stem = os.path.splitext(os.path.basename(result_path))[0]
    eval_out = os.path.join(out_path, f'eval_{result_stem}.json')
    with open(eval_out, 'w') as f:
        json.dump(output, f, indent=4)
    print(f'\nSaved evaluation → {eval_out}')
