

# function to show all components of cost function
# given set of parameters, tracks
# for every non-zero paramter, get cost of this component by itself

from sim_anneal import tracksCost
import numpy as np
import math
import random
from NewCostFunction import DisplayAllSplits, MyCostParams, GeneralCostFunc


def EvaluateCost(tracks, gt_tracks, MyCP, splitWt):
    eval_cost = {}
    # for each parameters in MyCP
    # mdDist,sym,ang,asp
    if MyCP.mdDistWt > 0:
        TestCP = MyCostParams(MyCP.mdDistWt,0,0,0,0,0,  0,0,0,0,0,0,  0,0,0,0,0,  0)
        eval_cost['mdDistSim'] = tracksCost(tracks, daughterCostFunc = GeneralCostFunc(TestCP), splitWt = 0)
        eval_cost['mdDistGT'] = tracksCost(gt_tracks, daughterCostFunc = GeneralCostFunc(TestCP), splitWt = 0)
        msg = 'D(m,d)  %8d   %8d' % (eval_cost['mdDistSim'], eval_cost['mdDistGT'])
        print(msg)
    if MyCP.symWt > 0:
        TestCP = MyCostParams(0,MyCP.symWt,0,0,0,0,  0,0,0,0,0,0,  0,0,0,0,0,  0)
        eval_cost['symSim'] = tracksCost(tracks, daughterCostFunc = GeneralCostFunc(TestCP), splitWt = 0)
        eval_cost['symGT'] = tracksCost(gt_tracks, daughterCostFunc = GeneralCostFunc(TestCP), splitWt = 0)
        msg = 'Symmetry  %8d   %8d' % (eval_cost['symSim'], eval_cost['symGT'])
        print(msg)
    if MyCP.angWt > 0:
        TestCP = MyCostParams(0,0,MyCP.angWt,0,0,0,  0,0,0,0,0,0,  0,0,0,0,0,  0)
        eval_cost['angSim'] = tracksCost(tracks, daughterCostFunc = GeneralCostFunc(TestCP), splitWt = 0)
        eval_cost['angGT'] = tracksCost(gt_tracks, daughterCostFunc = GeneralCostFunc(TestCP), splitWt = 0)
        msg = 'Angle       %8d   %8d' % (eval_cost['angSim'], eval_cost['angGT'])
        print(msg)
    if MyCP.aspWt > 0:
        TestCP = MyCostParams(0,0,0,MyCP.aspWt,0,0,  0,0,0,0,0,0,  0,0,0,0,0,  0)
        eval_cost['aspSim'] = tracksCost(tracks, daughterCostFunc = GeneralCostFunc(TestCP), splitWt = 0)
        eval_cost['aspGT'] = tracksCost(gt_tracks, daughterCostFunc=GeneralCostFunc(TestCP), splitWt=0)
        msg = 'AspectRatio   %8d   %8d' % (eval_cost['aspSim'], eval_cost['aspGT'])
        print(msg)

    if MyCP.meanIWt > 0:
        TestCP = MyCostParams(0,0,0,0,MyCP.meanIWt,0,  0,0,0,0,0,0,  0,0,0,0,0,  0)
        eval_cost['meanISim'] = tracksCost(tracks, daughterCostFunc = GeneralCostFunc(TestCP), splitWt = 0)
        eval_cost['meanIGT'] = tracksCost(gt_tracks, daughterCostFunc=GeneralCostFunc(TestCP), splitWt=0)
        msg = 'MeanI   %8d   %8d' % (eval_cost['meanISim'], eval_cost['meanIGT'])
        print(msg)

    # dist weights: Split NoSplit
    if MyCP.centWt > 0:
        if MyCP.centNoSplitWt > 0:
            TestCP = MyCostParams(0,0,0,0,0,0, MyCP.centWt,MyCP.centNoSplitWt,0,0,0,0,   0,0,0,0,0,  0)
            eval_cost['centNoSplitSim'] = tracksCost(tracks, daughterCostFunc=GeneralCostFunc(TestCP), splitWt=0)
            eval_cost['centNoSplitGT'] = tracksCost(gt_tracks, daughterCostFunc=GeneralCostFunc(TestCP), splitWt=0)
            msg = 'centNoSplit %8d   %8d' % (eval_cost['centNoSplitSim'], eval_cost['centNoSplitGT'])
            print(msg)
        if MyCP.centSplitMDWt > 0:
            TestCP = MyCostParams(0,0,0,0,0,0,  MyCP.centWt,0,MyCP.centSplitMDWt,MyCP.centSplitDDWt,17,34,  0,0,0,0,0,  0)
            eval_cost['centSplitSim'] = tracksCost(tracks, daughterCostFunc=GeneralCostFunc(TestCP), splitWt=0)
            eval_cost['centSplitGT'] = tracksCost(gt_tracks, daughterCostFunc=GeneralCostFunc(TestCP), splitWt=0)
            msg = 'CentSplit   %8d   %8d' % (eval_cost['centSplitSim'], eval_cost['centSplitGT'])
            print(msg)
    # vol weights: Split NoSplit
    # volWt = 1; volNoSplitWt = 0.0005; volSplitWt = 0.002; volNoSplitMult = 0.9; volSplitMult = 0.8
    if MyCP.volWt > 0:
        if MyCP.volNoSplitWt > 0:
            TestCP = MyCostParams(0,0,0,0,0,0,  0,0,0,0,0,0,  MyCP.volWt,MyCP.volNoSplitWt,0,0.9,0,  0)
            eval_cost['volNoSplitSim'] = tracksCost(tracks, daughterCostFunc=GeneralCostFunc(TestCP), splitWt=0)
            eval_cost['volNoSplitGT'] = tracksCost(gt_tracks, daughterCostFunc=GeneralCostFunc(TestCP), splitWt=0)
            msg = 'VolNoSplit  %8d   %8d' % (eval_cost['volNoSplitSim'], eval_cost['volNoSplitGT'])
            print(msg)
        if MyCP.volSplitWt > 0:
            TestCP = MyCostParams(0,0,0,0,0,0,  0,0,0,0,0,0,  MyCP.volWt,0,MyCP.volSplitWt,0,0.8,  0)
            eval_cost['volSplitSim'] = tracksCost(tracks, daughterCostFunc=GeneralCostFunc(TestCP), splitWt=0)
            eval_cost['volSplitGT'] = tracksCost(gt_tracks, daughterCostFunc=GeneralCostFunc(TestCP), splitWt=0)
            msg = 'VolSplit    %8d   %8d' % (eval_cost['volSplitSim'], eval_cost['volSplitGT'])
            print(msg)

    if MyCP.splitWt > 0:
        TestCP = MyCostParams(0,0,0,0,0,0,   0,0,0,0,0,0,   0,0,0,0,0,  splitWt)
        eval_cost['splitSim'] = tracksCost (tracks, daughterCostFunc = GeneralCostFunc(TestCP), splitWt = splitWt)
        eval_cost['splitGT'] = tracksCost(gt_tracks, daughterCostFunc=GeneralCostFunc(TestCP), splitWt=splitWt)
        msg = 'Split       %8d   %8d' % (eval_cost['splitSim'], eval_cost['splitGT'])
        print(msg)

    # full cost
    eval_cost['FullSim'] = tracksCost(tracks, daughterCostFunc=GeneralCostFunc(MyCP), splitWt=splitWt)
    eval_cost['FullGT'] = tracksCost(gt_tracks, daughterCostFunc=GeneralCostFunc(MyCP), splitWt=splitWt)
    msg = 'Full        %8d   %8d' % (eval_cost['FullSim'], eval_cost['FullGT'])
    print(msg)

    print ('tracksCost:',eval_cost)
    # ang, sym, centNoSplit, centSplit, volNoSplit, volSplit, splitWt

    # Which ones are Sim <= GT


    #Output_info['GT_split_volume_cost'] = cost
    return eval_cost


def EvaluateSimCost(tracks,  MyCP, splitWt):
    eval_cost = {}
    # for each parameters in MyCP
    # mdDist,sym,ang,asp
    if MyCP.mdDistWt > 0:
        TestCP = MyCostParams(MyCP.mdDistWt,0,0,0,0,0,  0,0,0,0,0,0,  0,0,0,0,0,  0)
        eval_cost['mdDistSim'] = tracksCost(tracks, daughterCostFunc = GeneralCostFunc(TestCP), splitWt = 0)
        msg = 'D(m,d)  %8d' % (eval_cost['mdDistSim'])
        print(msg)
    if MyCP.symWt > 0:
        TestCP = MyCostParams(0,MyCP.symWt,0,0,0,0,  0,0,0,0,0,0,  0,0,0,0,0,  0)
        eval_cost['symSim'] = tracksCost(tracks, daughterCostFunc = GeneralCostFunc(TestCP), splitWt = 0)
        msg = 'Symmetry  %8d' % (eval_cost['symSim'])
        print(msg)
    if MyCP.angWt > 0:
        TestCP = MyCostParams(0,0,MyCP.angWt,0,0,0,  0,0,0,0,0,0,  0,0,0,0,0,  0)
        eval_cost['angSim'] = tracksCost(tracks, daughterCostFunc = GeneralCostFunc(TestCP), splitWt = 0)
        msg = 'Angle       %8d' % (eval_cost['angSim'])
        print(msg)
    if MyCP.aspWt > 0:
        TestCP = MyCostParams(0,0,0,MyCP.aspWt,0,0,  0,0,0,0,0,0,  0,0,0,0,0,  0)
        eval_cost['aspSim'] = tracksCost(tracks, daughterCostFunc = GeneralCostFunc(TestCP), splitWt = 0)
        msg = 'AspectRatio   %8d' % (eval_cost['aspSim'])
        print(msg)

    if MyCP.meanIWt > 0:
        TestCP = MyCostParams(0,0,0,0,MyCP.meanIWt,0,  0,0,0,0,0,0,  0,0,0,0,0,  0)
        eval_cost['meanISim'] = tracksCost(tracks, daughterCostFunc = GeneralCostFunc(TestCP), splitWt = 0)
        msg = 'MeanI   %8d' % (eval_cost['meanISim'])
        print(msg)

    # dist weights: Split NoSplit
    if MyCP.centWt > 0:
        if MyCP.centNoSplitWt > 0:
            TestCP = MyCostParams(0,0,0,0,0,0, MyCP.centWt,MyCP.centNoSplitWt,0,0,0,0,   0,0,0,0,0,  0)
            eval_cost['centNoSplitSim'] = tracksCost(tracks, daughterCostFunc=GeneralCostFunc(TestCP), splitWt=0)
            msg = 'centNoSplit %8d' % (eval_cost['centNoSplitSim'])
            print(msg)
        if MyCP.centSplitMDWt > 0:
            TestCP = MyCostParams(0,0,0,0,0,0,  MyCP.centWt,0,MyCP.centSplitMDWt,MyCP.centSplitDDWt,17,34,  0,0,0,0,0,  0)
            eval_cost['centSplitSim'] = tracksCost(tracks, daughterCostFunc=GeneralCostFunc(TestCP), splitWt=0)
            msg = 'CentSplit   %8d' % (eval_cost['centSplitSim'])
            print(msg)
    # vol weights: Split NoSplit
    # volWt = 1; volNoSplitWt = 0.0005; volSplitWt = 0.002; volNoSplitMult = 0.9; volSplitMult = 0.8
    if MyCP.volWt > 0:
        if MyCP.volNoSplitWt > 0:
            TestCP = MyCostParams(0,0,0,0,0,0,  0,0,0,0,0,0,  MyCP.volWt,MyCP.volNoSplitWt,0,0.9,0,  0)
            eval_cost['volNoSplitSim'] = tracksCost(tracks, daughterCostFunc=GeneralCostFunc(TestCP), splitWt=0)
            msg = 'VolNoSplit  %8d' % (eval_cost['volNoSplitSim'])
            print(msg)
        if MyCP.volSplitWt > 0:
            TestCP = MyCostParams(0,0,0,0,0,0,  0,0,0,0,0,0,  MyCP.volWt,0,MyCP.volSplitWt,0,0.8,  0)
            eval_cost['volSplitSim'] = tracksCost(tracks, daughterCostFunc=GeneralCostFunc(TestCP), splitWt=0)
            msg = 'VolSplit    %8d' % (eval_cost['volSplitSim'])
            print(msg)

    if MyCP.splitWt > 0:
        TestCP = MyCostParams(0,0,0,0,0,0,   0,0,0,0,0,0,   0,0,0,0,0,  splitWt)
        eval_cost['splitSim'] = tracksCost (tracks, daughterCostFunc = GeneralCostFunc(TestCP), splitWt = splitWt)
        msg = 'Split       %8d' % (eval_cost['splitSim'])
        print(msg)

    # full cost
    eval_cost['FullSim'] = tracksCost(tracks, daughterCostFunc=GeneralCostFunc(MyCP), splitWt=splitWt)
    msg = 'Full        %8d' % (eval_cost['FullSim'])
    print(msg)

    print ('tracksCost:',eval_cost)

    return eval_cost

