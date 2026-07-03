# main simulated annealing for frame segment with balanced splits
# input start/end and config file

import math
import random
import numpy as np

import sim_anneal
from sim_anneal_utils import *
from NewCostFunction import  DisplayAllSplits, MyCostParams, GeneralCostFunc, DisplayTree
from EvaluateCost import *
import json
import os
import yaml
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_name', type=str,  default='.',
                        help='configuration name')
    parser.add_argument('-s', '--start_frame', type=int,  default='0',
                        help='first frame for annealing')
    parser.add_argument('-e', '--end_frame', type=int,  default='10',
                        help='last frame for annealing')
    parser.add_argument('-p', '--phase', type=str, default='16to32',
                        help='lineage stage: 16to32 (default) or 8to16')
    args = parser.parse_args()

start_frame = args.start_frame
end_frame = args.end_frame
nframes = end_frame - start_frame + 1

config_path = '../Config'
config_name = args.config_name
print('Config ',config_name)
Output_info = {}
Output_info['config_name'] = config_name

# this could be parameters for simulated annealing
full_config_name = config_name + '_config.yaml'
with open(os.path.join(config_path,full_config_name), 'r') as file:
    config_opts = yaml.safe_load(file)
#print(config_opts)

register_start_frame = 0 #config_opts['register_begin_frame'] # used in InitGT (so need to set start_frame accordingly)

data_path = config_opts.get('data_path', '../Data/Mouse/' + config_name + '/')
out_path = os.path.join(data_path,'Results')
if (not os.path.exists(out_path)):
    os.mkdir(out_path)
features_file = config_opts.get('features_file', 'Features.json')
feature_path = os.path.join(data_path, features_file)
fid = open(feature_path, "r")
data = json.load(fid)

# make NucCounts (sequence of number of nuclei in this frame segment)
nucCounts = []
list_nuclei_labels = {} # key is frame number
for iframe in range(start_frame, start_frame + nframes):
    s = np.asarray(data['volumes'][iframe])
    ind = np.nonzero(s)
    nNuc = int(ind[0].shape[0])
    nucCounts.append(nNuc)
    labels =  list(ind[0])
    #print(len(labels),nNuc)
    list_nuclei_labels[iframe] = labels

print('lineage start_frame, nframes', start_frame, nframes)
print('Nuclear Counts ',nucCounts)

sa_key = 'simulated_annealing' if args.phase == '16to32' else f'simulated_annealing_{args.phase}'
sa = config_opts[sa_key]
print(f'SA phase: {args.phase}  (config key: {sa_key})')
nepochs       = sa['nepochs']
splitWt       = sa['splitWt']
mdDistWt      = sa['mdDistWt']
symWt         = sa['symWt']
angWt         = sa['angWt']
aspWt         = sa['aspWt']
meanIWt       = sa['meanIWt']
stdIWt        = sa['stdIWt']
centWt        = sa['centWt'];  centNoSplitWt = sa['centNoSplitWt']
centSplitMDWt = sa['centSplitMDWt'];  centSplitDDWt = sa['centSplitDDWt']
centSplitMDDist = sa['centSplitMDDist'];  centSplitDDDist = sa['centSplitDDDist']
volWt         = sa['volWt'];   volNoSplitWt  = sa['volNoSplitWt']
volSplitWt    = sa['volSplitWt']
volNoSplitMult = sa['volNoSplitMult'];  volSplitMult = sa['volSplitMult']
mdDistSquared  = sa.get('mdDistSquared', False)
mdDistRef      = sa.get('mdDistRef', 0.0)
angTarget      = sa.get('angTarget', 156) * math.pi / 180
aspTarget      = sa.get('aspTarget', 1.8)
earlyStopIter  = sa.get('earlyStopIter', 500)
support_split  = sa.get('support_split', True)

print('constructing tracks ...',start_frame ,start_frame + nframes)
tracks = sim_anneal.Tracks(nucCounts, supportSplit=support_split)

print('Load features into tracks')
tracksInit(tracks, data, start_frame, list_nuclei_labels, register_start_frame)
sim_anneal.tracksShuffle(tracks)
tracks.validate()

MyCP = MyCostParams(mdDistWt, symWt, angWt, aspWt, meanIWt, stdIWt,
                    centWt, centNoSplitWt, centSplitMDWt, centSplitDDWt,
                    centSplitMDDist, centSplitDDDist,
                    volWt, volNoSplitWt, volSplitWt, volNoSplitMult, volSplitMult,
                    splitWt, mdDistSquared=mdDistSquared, mdDistRef=mdDistRef,
                    angTarget=angTarget, aspTarget=aspTarget)

Output_info = {}
Output_info['nepochs'] = nepochs
print('number of epochs', nepochs)

# When supportSplit=False, splitWt must be None (not 0) for anneal/tracksCost
anneal_splitWt = splitWt if support_split else None

cost = sim_anneal.tracksCost(tracks, daughterCostFunc=GeneralCostFunc(MyCP), splitWt=anneal_splitWt)
print('cost before annealing', cost)


tracks = sim_anneal.anneal (tracks, epochs = nepochs, daughterCostFunc = GeneralCostFunc(MyCP), splitWt = anneal_splitWt, earlyStopIter = earlyStopIter)
cost = sim_anneal.tracksCost (tracks, daughterCostFunc = GeneralCostFunc(MyCP), splitWt = anneal_splitWt)
print('cost after annealing ',cost)

# display all splits
print('Stats of final lineage')
DisplayAllSplits(tracks, nframes, start_frame)

Output_info['Config_Params'] = MyCP.__dict__
Output_info['Sim_full_cost'] = cost

costs = EvaluateSimCost(tracks, MyCP, splitWt)
Output_info['costs'] = costs

# output each lineage for input into matlab based on tracks
mat_graph = OutputGraph(tracks, start_frame)
start_str = '%03d' % start_frame
end_str = '%03d' % end_frame
fid = open(os.path.join(out_path,'sim_graph_' + start_str + '_' + end_str + '.json'),'w')
json.dump(mat_graph,fid, indent = 4)
fid.close()
