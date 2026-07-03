
# create tracks with FSEQ GT
import copy
import math
import random
import numpy as np
import sim_anneal
from matplotlib import pyplot as plt
import os
import json

def Distance(c1,c2):
    d = math.sqrt( (c1[0] - c2[0])*(c1[0] - c2[0]) + (c1[1] - c2[1])*(c1[1] - c2[1]) + (c1[2] - c2[2])*(c1[2] - c2[2]))
    return d

def OutputGraph(tracks,start_frame):
    node_list = []
    edge_list = []
    nframes = len(tracks.frames)
    for i in range(nframes):  # initialize daughter centroids and volumes
        for ln in tracks.frames[i].nuclei:
            node_list.append((i + start_frame,ln.label))
            if (ln.mother is not None):
                mother_label = ln.mother.label
                edge_list.append([ [i + start_frame,ln.label],[i + start_frame - 1,mother_label] ] )
            if (ln.daughters[0] is not None):
                daughter_label = (ln.daughters[0]).label
                edge_list.append([[i + start_frame,ln.label], [i + start_frame + 1,daughter_label] ] )
            if (ln.daughters[1] is not None):
                daughter_label = (ln.daughters[1]).label
                edge_list.append([[i + start_frame, ln.label], [i + start_frame + 1, daughter_label]])
    mat_graph = {}
    mat_graph['Nodes'] = node_list
    mat_graph['Edges'] = edge_list
    return mat_graph

# start_frame: real frame of start of tracks
# start_info: frame where registration began
# list_nuclei_labels: dict, index by real frame
def tracksInit(tracks, data,  start_frame, list_nuclei_labels,start_info):
    nframes = len(tracks.frames)
    print(f'Tracks start at frame {start_frame}')
    print(f'Tracks end at frame   {start_frame + nframes - 1}')

    for i in range(nframes):  # initialize daughter centroids and volumes
        #print('frame, nNuclei, nLabels ',i+start_frame, len(tracks.frames[i].nuclei),len(list_nuclei_labels[i+start_frame]))
        #print(list_nuclei_labels[i + start_frame])
        ilabel = 0
        for ln,curr_label in zip(tracks.frames[i].nuclei,list_nuclei_labels[i + start_frame]):
                    for j in range(3):
                        ln.centroid[j] = data['centroids'][start_frame + i - start_info ][curr_label][j]
                    #print('frame, curr_label, centroid ',start_frame + i -start_info, curr_label, ln.centroid)
                    ln.volume = data['volumes'][start_frame + i - start_info][curr_label]
                    if (ln.volume < 1e-5):
                       print('bad vol frame ', i + start_frame)
                       print('label ', ilabel, 'curr_label ', curr_label)
                       print(data['volumes'][start_frame + i - start_info])
                       exit()

                    ln.solidity = data['solidities'][start_frame + i - start_info][curr_label]
                    ln.sphericity = data['sphericity'][start_frame + i - start_info][curr_label]
                    ln.aspectratio = data['aspectratios'][start_frame + i - start_info ][curr_label]
                    ln.meanI = data['mean_intensity20'][start_frame + i - start_info][curr_label]
                    ln.stdI = data['std_intensity'][start_frame + i - start_info][curr_label]
                    ln.frame = i
                    ln.label = int(curr_label + 1) # note the label in the matlab graph is one more than indexing into data
                    ilabel = ilabel + 1

    return None



def tracksInitFromGT (tracks, data, data1, edges, start_frame, list_nuclei_labels, exclude_labels, start_info):
    nframes = len(tracks.frames)
    print(f'Tracks start at frame {start_frame}')
    print(f'Tracks end at frame   {start_frame + nframes - 1}')

    for  i in range (nframes):   # re-initialize so all daughters are none
        for  ln in tracks.frames[i].nuclei:
            ln.daughters = [None,None]
    
    for  i in range (nframes):   # initialize daughter centroids and volumes
        #print('frame ',i+start_frame, len(tracks.frames[i].nuclei))
        ilabel = 0
        #print('number of nuclei in this frame ',len(tracks.frames[i].nuclei))
        #print('Frame ',i+start_frame,' nlabels ',len(list_nuclei_labels[i + start_frame]),'GT labels ',list_nuclei_labels[i + start_frame])
        #print('Number nuc ', len(tracks.frames[i].nuclei))
        #print(exclude_labels[i + start_frame ])
        #if (len(exclude_labels) > 0):
        #    if ((i+start_frame) in exclude_labels.keys()):
        #        print('removing label ',exclude_labels[i + start_frame],' from frame ',i+start_frame)
        #list_nuclei_labels[i + start_frame].remove(exclude_labels[i + start_frame ])
        if (len(exclude_labels) > (i + start_frame)):
            list_nuclei_labels[i + start_frame ].remove(exclude_labels[i + start_frame  ])


        #print('nuclei this frame ',list_nuclei_labels[i + start_frame])
        for  ln in tracks.frames[i].nuclei:
            if ((i + start_frame) < (start_frame + nframes)):
                if (ilabel < len(list_nuclei_labels[i + start_frame])):
                    curr_label = list_nuclei_labels[i + start_frame ][ilabel]
                else:
                    print('bad ilabel at frame ',ilabel,i+start_frame)
            else:
                print('bad frame at ',i + start_frame) # bad frame at 16
            #print('frame, label ', start_frame + i - start_info, curr_label-1)
            for  j in range (3):
                ln.centroid[j] = data['centroids'][start_frame + i - start_info ][curr_label - 1][j]
                # good ln.centroid[j] = data['centroids'][start_frame + i - start_info][curr_label-1][j]
            #print('frame, curr_label, centroid ',start_frame + i - start_info, curr_label, ln.centroid)
            ln.volume = data['volumes'][start_frame + i - start_info ][curr_label - 1]
            if (ln.volume < 1e-5):
                print('frame ',i+start_frame -1)
                print('label ', ilabel, 'curr_label ',curr_label)

            ln.solidity = data['solidities'][start_frame + i - start_info ][curr_label - 1]
            ln.sphericity = data['sphericity'][start_frame + i - start_info ][curr_label - 1]
            ln.aspectratio = data['aspectratios'][start_frame + i - start_info ][curr_label - 1]
            ln.meanI = data1['mean_intensity20'][start_frame + i - start_info ][curr_label - 1]
            ln.stdI = data1['std_intensity'][start_frame + i - start_info ][curr_label - 1]
            ln.frame = i
            ln.label = curr_label #  note the label in the matlab graph is one more than indexing into data
            # note: mother/daughter indices are not labels
            # get mother and daughter(s) from lineage
            # mother: find edge with frame one less
            if i == 0:
                #print('frame ', start_frame + i, ' label ', ln.label)
                ln.mother = None   ### should already be None
                #print('First Frame index, labels ',ln.index, ln.label) # looks good (0 to 15) --> (1 to 16)
            elif (i > 0):
                test_frame = start_frame + i
                #print('index/label for ', test_frame, ': ',
                #      [(ln.index, ln.label) for ln in tracks.frames[test_frame - start_frame].nuclei])

                #print('about to set this mother ',tracks.frames[0].nuclei[0])
                #print('frame ',start_frame + i,' label ',ln.label)
                mother_ptr = GetMotherFromNode(start_frame + i, ln.label, edges, start_frame, tracks)
                ln.mother = mother_ptr
                #print('mother label ',mother_ptr.label, test_frame)
                #print('mother,daughter labels & link to daughter from mother', mother_ptr.label, ln.label, mother_ptr)
                if (mother_ptr.daughters[0] == None):
                    #print("FIRST DAUGHTER",ln.label)
                    mother_ptr.daughters[0] = ln
                elif (mother_ptr.daughters[1] == None):
                    mother_ptr.daughters[1] = ln
                else:
                    print('this link already has two daughters ',ln)
                    exit()
                #print('mother,daughter labels', mother_ptr.label, ln.label, mother_ptr.daughters[0].label)
            if i == (nframes -1):
                ln.daughters = [None,None]
            ilabel = ilabel + 1
        # each frame should set every ln
        #print('frame ',i + start_frame,' has n nodes ', len(tracks.frames[i].nuclei))
        # are they all set ?
        igood = 0
        for  ln in tracks.frames[i].nuclei:
            if ln.label <= 0:
                print('found bad nuclei ',igood)
                exit()
            else:
                igood = igood + 1


    return None

def tracksInitFromSimGraph(tracks, data, data1, sg_nodes, sg_edges, start_frame, start_info):
    """Initialize tracks from a Python-format sim_graph (OutputGraph output).

    sg_nodes: list of [frame, label]
    sg_edges: list of [[frame1, label1], [frame2, label2]]  (may be bidirectional)
    """
    nframes = len(tracks.frames)

    # Frame -> sorted label list
    frame_labels = {}
    for frame, label in sg_nodes:
        frame_labels.setdefault(frame, []).append(label)
    for f in frame_labels:
        frame_labels[f].sort()

    # Reset all links
    for i in range(nframes):
        for ln in tracks.frames[i].nuclei:
            ln.daughters = [None, None]
            ln.mother = None

    # Assign features; build (frame, label) -> nucleus lookup
    nucleus_map = {}
    for i in range(nframes):
        frame = start_frame + i
        labels = frame_labels.get(frame, [])
        feat_idx = frame - start_info
        for ln, label in zip(tracks.frames[i].nuclei, labels):
            fi = label - 1
            ln.label = label
            ln.frame = i
            for j in range(3):
                ln.centroid[j] = data['centroids'][feat_idx][fi][j]
            ln.volume    = data['volumes'][feat_idx][fi]
            ln.solidity  = data['solidities'][feat_idx][fi]
            ln.sphericity = data['sphericity'][feat_idx][fi]
            ln.aspectratio = data['aspectratios'][feat_idx][fi]
            ln.meanI = data1['mean_intensity20'][feat_idx][fi]
            ln.stdI  = data1['std_intensity'][feat_idx][fi]
            nucleus_map[(frame, label)] = ln

    # Set mother/daughter links — keep only forward edges (f2 == f1 + 1)
    for edge in sg_edges:
        (f1, l1), (f2, l2) = edge[0], edge[1]
        if f2 != f1 + 1:
            continue
        parent = nucleus_map.get((f1, l1))
        child  = nucleus_map.get((f2, l2))
        if parent is None or child is None:
            continue
        child.mother = parent
        if parent.daughters[0] is None:
            parent.daughters[0] = child
        elif parent.daughters[1] is None:
            parent.daughters[1] = child
        else:
            print(f'Warning: >2 daughters for frame {f1} label {l1}')

    return None




# use lineage to find mother, get mother's label, use label and previous frame to get index of mother
def GetMotherFromNode(frame, ilabel, edges, start_frame,tracks):
    frame_str = '%03d' % frame
    label_str = '%03d' % ilabel
    node_str = frame_str + '_' + label_str
    for iedge in edges:
        edge1 = iedge['EndNodes'][0]
        edge2 = iedge['EndNodes'][1]
        if node_str == edge1:
            test_frame = int(edge2[:3])
            if test_frame == (frame - 1):
                test_label = int(edge2[-3:])
                # find ln node with this frame,label - use this index
                index = GetIndexFromLabel(test_frame, test_label, start_frame, tracks)
                if (index == -1):
                    # print for test_frame (mother frame)
                    # each index and label
                    print('index/label for ',test_frame,': ',[(ln.index,ln.label) for ln in tracks.frames[test_frame - start_frame].nuclei])
                    print('no mother ptr ',test_frame,test_label,start_frame)
                return index
        elif node_str == edge2:
            test_frame = int(edge1[:3])
            if test_frame == (frame - 1):
                test_label = int(edge1[-3:])
                index = GetIndexFromLabel(test_frame, test_label, start_frame, tracks)
                if (index == -1):
                    print('no mother ptr ',test_frame,test_label,start_frame)
                    print('index/label for ',test_frame,': ',[(ln.index,ln.label) for ln in tracks.frames[test_frame - start_frame].nuclei])
                return index
    return None



def GetIndexFromLabel(frame,label,start_frame,tracks):
    for ln in tracks.frames[frame - start_frame].nuclei:
        if (ln.label == label):
            if (label == 0):
                print('found bad label 0')
                print('frame,label ',frame,label)
                print(ln)
                exit()
            return ln
    return -1



def centroidDist (ca, cb):   # euclidean distance between two centroids (lists of three numbers)
    dist = 0.0
    for  i in range (3):
        dist += (ca[i] - cb[i])**2
    dist = dist**0.5
    return  dist

def GetNucleiPerFrame(min_frame,max_frame,nodes,number_of_excludes_per_frame):
    list_nuclei_labels = {}
    nucCounts = []
    for iframe in range(min_frame,max_frame + 1):
        list_nuclei_labels[iframe] = []
        for inode in nodes:
            # get frame and label id
            node_str = inode['Name']
            frame = int(node_str[:3])
            label = int(node_str[-3:])
            if (frame == iframe):
                list_nuclei_labels[iframe].append(label)
        # how many unique in list
        nNuclei = len(np.unique(list_nuclei_labels[iframe]))
        #print('frame ',iframe,' nNuclei ',nNuclei,' labels ',list_nuclei_labels[iframe])
        nucCounts.append(nNuclei - number_of_excludes_per_frame) # subtract one because one polar body not part of balanced forest
    return [nucCounts, list_nuclei_labels]


def Eval(nframes, gt_tracks, tracks, start_frame):
    total_good_splits = 0
    total_good_1to1 = 0
    total_splits = 0
    total_1to1 = 0
    for iframe in range(nframes-1):
        mother_split = 0
        one_to_one_match = 0
        nNuc = len(tracks.frames[iframe].nuclei)
        nNuc_next_frame = len(tracks.frames[iframe + 1].nuclei)
        nsplits = nNuc_next_frame - nNuc
        n1to1 = nNuc - nsplits
        total_splits = total_splits + nsplits
        total_1to1 = total_1to1 + n1to1
        for ln1 in tracks.frames[iframe].nuclei:
            ann_label = ln1.label
            # if split (2 daughters) is there a match in gt ?
            if (ln1.daughters[0] != None) and (ln1.daughters[1] != None):
                # is mother in gt as mother of two?
                d1 = ln1.daughters[0].label
                d2 = ln1.daughters[1].label
                for ln2 in gt_tracks.frames[iframe].nuclei:
                    if (ln2.daughters[0] != None) and (ln2.daughters[1] != None):
                        g1 = ln2.daughters[0].label
                        g2 = ln2.daughters[1].label
                        if (ln2.label == ann_label) and ((d1 == g1) or (d1 == g2)) and ((d2 == g1) or (d2 == g2)):
                            mother_split = mother_split + 1

            # if one-to-one, is there a match in gt ?
            if (ln1.daughters[0] != None) or (ln1.daughters[1] != None):
                if (ln1.daughters[0] != None):
                    ann_label_d = ln1.daughters[0].label
                elif (ln1.daughters[1] != None):
                    ann_label_d = ln1.daughters[1].label
                for ln2 in gt_tracks.frames[iframe].nuclei:
                    if (ln2.label == ann_label):
                        # does gt have one daughter?
                        if (ln2.daughters[1] == None):
                            if (ln2.daughters[0].label == ann_label_d):
                                one_to_one_match = one_to_one_match + 1
                        elif (ln2.daughters[0] == None):
                            if (ln2.daughters[1].label == ann_label_d):
                                one_to_one_match = one_to_one_match + 1
 
                                

        if (mother_split != nsplits):
            for ln1 in tracks.frames[iframe].nuclei:
                ann_label = ln1.label
                if (ln1.daughters[0] != None) and (ln1.daughters[1] != None):
                    # is mother in gt as mother of two?
                    d1 = ln1.daughters[0].label
                    d2 = ln1.daughters[1].label
                    for ln2 in gt_tracks.frames[iframe].nuclei:
                        if (ln2.daughters[0] != None) and (ln2.daughters[1] != None):
                            g1 = ln2.daughters[0].label
                            g2 = ln2.daughters[1].label
                            print('******', iframe + start_frame, ' sim split ', ann_label, d1, d2, ' not the same as gt split ', ln2.label, g1, g2)
                            # print out the angle, symmetry and three distance in triangle
                            md1 = Distance(ln2.centroid, ln2.daughters[0].centroid)
                            md2 = Distance(ln2.centroid, ln2.daughters[1].centroid)
                            aveDist = (md1 + md2) / 2
                            if (aveDist < 1e-5):
                                aveDist = 1e-4
                            # difference between (m,d1) and (m,d2) should be small
                            symmetry = math.fabs(md1 - md2) / aveDist
                            v1 = np.asarray(ln2.centroid) - np.asarray(ln2.daughters[0].centroid)
                            v2 = np.asarray(ln2.centroid) - np.asarray(ln2.daughters[1].centroid)
                            v1_mag = math.sqrt(sum(pow(element, 2) for element in v1))
                            v2_mag = math.sqrt(sum(pow(element, 2) for element in v2))
                            val = v1_mag * v2_mag
                            if (val < 1e-5):
                                val = 1e-4
                            val = np.dot(v1, v2) / val
                            if (val > 1):
                                val = 1
                            elif (val < -1):
                                val = -1
                            angle = (180 / math.pi) * math.acos(val)
                            # volume ratio
                            if (ln2.volume > 100) and (ln2.daughters[0].volume > 100) and (ln2.daughters[1].volume > 100):
                                vol_ratio = (ln2.daughters[0].volume + ln2.daughters[1].volume) / ln2.volume
                            else:
                                vol_ratio = 1.0
                            print('GT Angle ',angle)
                            print('GT Distances ',md1,md2)
                            print('GT Symmetry ',symmetry)
                            print('GT Vol Ratio ', vol_ratio)
                            # now for sim
                            md1 = Distance(ln1.centroid, ln1.daughters[0].centroid)
                            md2 = Distance(ln1.centroid, ln1.daughters[1].centroid)
                            aveDist = (md1 + md2) / 2
                            if (aveDist < 1e-5):
                                aveDist = 1e-4
                            # difference between (m,d1) and (m,d2) should be small
                            symmetry = math.fabs(md1 - md2) / aveDist
                            v1 = np.asarray(ln1.centroid) - np.asarray(ln1.daughters[0].centroid)
                            v2 = np.asarray(ln1.centroid) - np.asarray(ln1.daughters[1].centroid)
                            v1_mag = math.sqrt(sum(pow(element, 2) for element in v1))
                            v2_mag = math.sqrt(sum(pow(element, 2) for element in v2))
                            val = v1_mag * v2_mag
                            if (val < 1e-5):
                                val = 1e-4
                            val = np.dot(v1, v2) / val
                            if (val > 1):
                                val = 1
                            elif (val < -1):
                                val = -1
                            angle = (180 / math.pi) * math.acos(val)
                            # volume ratio
                            if (ln1.volume > 100) and (ln1.daughters[0].volume > 100) and (ln1.daughters[1].volume > 100):
                                vol_ratio = (ln1.daughters[0].volume + ln1.daughters[1].volume) / ln1.volume
                            else:
                                vol_ratio = 1.0
                            print('Sim Angle ', angle)
                            print('Sim Distances ', md1, md2)
                            print('Sim Symmetry ', symmetry)
                            print('Sim Vol Ratio ',vol_ratio)


        total_good_1to1 = total_good_1to1 + one_to_one_match
        total_good_splits = total_good_splits + mother_split

        print('Frame ',iframe + start_frame ,' has 1-1 matches ',one_to_one_match, ' out of ',n1to1)
        print(' mothers good in splits ',mother_split,' out of ',nsplits)


    print('Total one-to-one ',total_good_1to1, ' out of ',total_1to1)
    print('Total splits ',total_good_splits, ' out of ',total_splits)
    return [total_good_1to1,total_1to1, total_good_splits,total_splits]
