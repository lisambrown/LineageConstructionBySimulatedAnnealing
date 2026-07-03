from sim_anneal import *
import numpy as np
import math

def centroidDist (ca, cb):   # euclidean distance between two centroids (lists of three numbers)
    dist = 0.0
    for  i in range (3):
        dist += (ca[i] - cb[i])**2
    dist = dist**0.5
    return  dist

def Distance(c1,c2):
    d = math.sqrt( (c1[0] - c2[0])*(c1[0] - c2[0]) + (c1[1] - c2[1])*(c1[1] - c2[1]) + (c1[2] - c2[2])*(c1[2] - c2[2]))
    return d

def DisplayTree(tracks,nframes, start_frame):
    print('Tree Centroids ')
    for iframe in range(nframes):
        for ln in tracks.frames[iframe].nuclei:
            if (ln.daughters[0] != None) and (ln.daughters[1] != None):
                print('split at frame ',iframe,ln.centroid)
            else:
                print('no split at frame ',iframe,ln.centroid,ln.frame,ln.label)



def DisplayAllSplits(gt_tracks,nframes, start_frame):
    nsplits = 0
    ave_md_dist = 0
    ave_dd_dist = 0
    ave_vol_ratio = 0
    ave_solidity = 0
    ave_angle = 0
    ave_symmetry = 0
    sd_md_dist = 0
    sd_dd_dist = 0
    sd_vol_ratio = 0
    sd_solidity = 0
    sd_symmetry = 0
    sd_angle = 0
    ave_solidity_nosplit = 0
    ave_vol_nosplit = 0
    sd_vol_nosplit = 0
    ave_sphericity = 0
    sd_sphericity = 0
    ave_sphericity_nosplit = 0
    ave_aspectratio = 0
    sd_aspectratio = 0
    ave_aspectratio_nosplit = 0
    ave_meanI = 0
    sd_meanI = 0
    ave_meanI_nosplit = 0
    ave_stdI = 0
    sd_stdI = 0
    ave_stdI_nosplit = 0
    count = 0


    for iframe in range(nframes):
        for ln in gt_tracks.frames[iframe].nuclei:
            if (ln.daughters[0] != None) and (ln.daughters[1] != None):
                nsplits = nsplits + 1
                if (ln.solidity == None):
                    solidity = 1.0
                else:
                    solidity = ln.solidity
                str = '%5.2f' % solidity
                #print(nsplits, iframe+start_frame+50,str)
                # angle
                v1 = np.asarray(ln.centroid) - np.asarray(ln.daughters[0].centroid)
                v2 = np.asarray(ln.centroid) - np.asarray(ln.daughters[1].centroid)
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
                angle = (180/math.pi)*math.acos(val)
                str = '%5.2f' % angle
                #print(nsplits, iframe+start_frame+50,str)
                # Symmetry
                md1 = Distance(ln.centroid, ln.daughters[0].centroid)
                md2 = Distance(ln.centroid, ln.daughters[1].centroid)
                aveDist = (md1 + md2) / 2
                if (aveDist < 1e-5):
                    aveDist = 1e-4
                # difference between (m,d1) and (m,d2) should be small
                symmetry = math.fabs(md1 - md2) / aveDist
                #print(nsplits,iframe+start_frame,md1,md2,aveDist,symmetry)
                # compute (d1 + d2)/m volume ratio
                if (ln.volume > 100) and (ln.daughters[0].volume > 100) and (ln.daughters[1].volume > 100):
                    vol_ratio = (ln.daughters[0].volume + ln.daughters[1].volume)/ln.volume
                else:
                    vol_ratio = 1.0
                str = '%5.2f' % vol_ratio
                #print(str)
                # compute distances
                dist_md1 = centroidDist(ln.centroid,ln.daughters[0].centroid)
                dist_md2 = centroidDist(ln.centroid,ln.daughters[1].centroid)
                dist_d1d2 = centroidDist(ln.daughters[0].centroid,ln.daughters[1].centroid)
                ave_md_dist = ave_md_dist + dist_md1 + dist_md2
                ave_dd_dist = ave_dd_dist + dist_d1d2
                ave_vol_ratio = ave_vol_ratio + vol_ratio
                ave_angle = ave_angle + angle
                ave_symmetry = ave_symmetry + symmetry
                ave_solidity = ave_solidity + solidity
                ave_sphericity = ave_sphericity + ln.sphericity 
                ave_aspectratio = ave_aspectratio + ln.aspectratio
                ave_meanI = ave_meanI + ln.meanI
                #print('meanI ',ln.meanI,ln.stdI)
                ave_stdI = ave_stdI + ln.stdI
                sd_md_dist = sd_md_dist + dist_md1*dist_md1 + dist_md2*dist_md2
                sd_dd_dist = sd_dd_dist + (dist_d1d2*dist_d1d2)
                sd_vol_ratio = sd_vol_ratio + vol_ratio*vol_ratio
                sd_solidity = sd_solidity + solidity * solidity
                sd_angle = sd_angle + angle * angle
                sd_sphericity = sd_sphericity + ln.sphericity * ln.sphericity
                sd_aspectratio = sd_aspectratio + ln.aspectratio * ln.aspectratio
                sd_meanI = sd_meanI + ln.meanI * ln.meanI
                sd_stdI = sd_stdI + ln.stdI * ln.stdI
                sd_symmetry = sd_symmetry + symmetry * symmetry
                #print('frame m, d1,d2 d(m,d1), d(m,d2), d(d1,d2)',iframe+start_frame,ln.label,ln.daughters[0].label,ln.daughters[1].label,round(dist_md1),round(dist_md2),round(dist_d1d2))
            else:
                #if (ln.solidity != None):
                ave_solidity_nosplit = ave_solidity_nosplit + ln.solidity
                count = count + 1
                #else: # WHY??
                #    print('solidity none')
                ave_meanI_nosplit = ave_meanI_nosplit + ln.meanI
                ave_stdI_nosplit = ave_stdI_nosplit + ln.stdI
                ave_aspectratio_nosplit = ave_aspectratio_nosplit + ln.aspectratio
                ave_sphericity_nosplit = ave_sphericity_nosplit + ln.sphericity
                if (ln.daughters[0] != None) and (ln.daughters[0].volume > 1e-5):
                    ave_vol_nosplit = ave_vol_nosplit + ln.volume/ln.daughters[0].volume
            #if (iframe + start_frame + 50 == 136):
            #    print('frame m, d1,d2 ', iframe + start_frame + 50, ln.label, ln.daughters)

    print('number of splits ',nsplits)
    if (nsplits > 0):
        ave_md_dist = (ave_md_dist/(2*nsplits))
        print('ave/sd md dist ', ave_md_dist, ((sd_md_dist/(2*nsplits)) - (ave_md_dist*ave_md_dist))**.5)
        ave_dd_dist = (ave_dd_dist/nsplits)
        print('ave/sd dd dist ', ave_dd_dist, ((sd_dd_dist/nsplits) - (ave_dd_dist*ave_dd_dist))**.5)
        ave_vol_ratio = ave_vol_ratio/nsplits
        ave_vol_nosplit = ave_vol_nosplit/count
        print('ave/sd vol ratio', ave_vol_ratio, ((sd_vol_ratio/nsplits) - (ave_vol_ratio*ave_vol_ratio))**.5, ave_vol_nosplit)
        ave_solidity = ave_solidity/nsplits
        print('ave mother solidity ',ave_solidity,((sd_solidity/nsplits) - (ave_solidity*ave_solidity))**.5)
        print('ave solidity no split', ave_solidity_nosplit / count)
        ave_angle = ave_angle/nsplits
        print('ave angle ',ave_angle,((sd_angle/nsplits) - (ave_angle*ave_angle))**.5)
        ave_symmetry = ave_symmetry/nsplits
        print('ave symmetry ',ave_symmetry,((sd_symmetry/nsplits) - (ave_symmetry*ave_symmetry))**.5)
        ave_sphericity = ave_sphericity/nsplits
        ave_sphericity_nosplit = ave_sphericity_nosplit/count
        print('ave sphericity ',ave_sphericity,((sd_sphericity/nsplits) - (ave_sphericity*ave_sphericity))**.5, ave_sphericity_nosplit)
        ave_aspectratio = ave_aspectratio / nsplits
        ave_aspectratio_nosplit = ave_aspectratio_nosplit/count
        print('ave aspect ratio ', ave_aspectratio, ((sd_aspectratio / nsplits) - (ave_aspectratio * ave_aspectratio)) ** .5, ave_aspectratio_nosplit)
        ave_meanI = ave_meanI / nsplits
        ave_meanI_nosplit = ave_meanI_nosplit/count
        print('ave meanI ', ave_meanI, ((sd_meanI / nsplits) - (ave_meanI * ave_meanI)) ** .5, ave_meanI_nosplit)
        ave_stdI = ave_stdI / nsplits
        ave_stdI_nosplit = ave_stdI_nosplit/count
        print('ave stdI ', ave_stdI, ((sd_stdI / nsplits) - (ave_stdI * ave_stdI)) ** .5, ave_stdI_nosplit)
    if (nsplits == 0):
        ave_vol_nosplit = ave_vol_nosplit/count
        print('ave/sd vol ratio', ave_vol_nosplit)
        ave_sphericity_nosplit = ave_sphericity_nosplit / count
        print('ave sphericity ',  ave_sphericity_nosplit)
        ave_aspectratio_nosplit = ave_aspectratio_nosplit / count
        print('ave aspect ratio ', ave_aspectratio_nosplit)
        ave_meanI_nosplit = ave_meanI_nosplit/count
        print('ave meanI ',ave_meanI_nosplit)
        ave_stdI_nosplit = ave_stdI_nosplit/count
        print('ave stdI_nosplit ',ave_stdI_nosplit)
    return

class MyCostParams:   # parameters for cost function
    def __init__ (
        self,
        # default values for cost parameters
        mdDistWt = 0,
        symWt = 0,
        angWt = 0,
        aspWt = 0,
        meanIWt = 0,
        stdIWt = 0,
        # centroid-mismatch parameters
        centWt           =  1.0,   # overall weight of centroid costs
        centNoSplitWt    =  1.0,   # weight of mother-single-daughter squared centroid distance
        centSplitMDWt    =  1.0,   # weight of mother-split-daughter squared deviation of centroid distance
        centSplitDDWt    =  1.0,   # weight of split-daughter-daughter squared deviation of centroid distance
        centSplitMDDist  =  5.0,   # best (zero-cost) distance between mother and split daughter
        centSplitDDDist  =  5.0,   # best (zero-cost) distance between the two split daughters
        #
        # volume-mismatch parameters
        volWt            =  1.0,   # overall weight volume costs
        volNoSplitWt     =  0.5,   # weight of mother-single-daughter squared volume mismatch
        volSplitWt       =  2.0,   # weight of mother-daughter-pair squared volume mismatch
        volNoSplitMult   =  1.1,   # single-daughter volumes as multiple of mother volume
        volSplitMult     =  1.2,   # sum of daughter volumes as multiple of mother volume
        #
        # unbalanced-forest penalty
        splitWt          =  1.0,   # weight of square of descendants in excess of two
        #
        # mdDist formula control
        # False (default): mdDistCost uses Euclidean distance, abs(dist)       when dist   > 20
        # True:            mdDistCost uses squared  distance, abs(sqDist-ref)  when sqDist > 20
        mdDistSquared    = False,
        mdDistRef        = 0.0,    # reference subtracted from sq-distance (only used when mdDistSquared=True)
        # angle cost target: angCost = |angle_rad - angTarget|
        # 16to32 default: 156*pi/180 ≈ 2.72 rad (penalises deviation from 156°)
        # From8 value:    156         (angle in rad vs 156 "degrees" → pushes toward 180°)
        angTarget        = 156 * math.pi / 180,
        # aspect ratio target: aspCost = (aspectratio - aspTarget)^2
        aspTarget        = 1.8,
    ):
        # initialize cost parameters
        self.mdDistWt = mdDistWt
        self.symWt = symWt
        self.angWt = angWt
        self.aspWt = aspWt
        self.meanIWt = meanIWt
        self.stdIWt = stdIWt
        self.mdDistSquared = mdDistSquared
        self.mdDistRef     = mdDistRef
        self.angTarget     = angTarget
        self.aspTarget     = aspTarget
        # centroid-mismatch parameters
        self.centWt           =  centWt
        self.centNoSplitWt    =  centNoSplitWt
        self.centSplitMDWt    =  centSplitMDWt
        self.centSplitDDWt    =  centSplitDDWt
        self.centSplitMDDist  =  centSplitMDDist
        self.centSplitDDDist  =  centSplitDDDist

        #
        # volume-mismatch parameters
        self.volWt            =  volWt
        self.volNoSplitWt     =  volNoSplitWt
        self.volSplitWt       =  volSplitWt
        self.volNoSplitMult   =  volNoSplitMult
        self.volSplitMult     =  volSplitMult
        #
        # unbalanced-forest penalty
        self.splitWt          =  splitWt

class GeneralCostFunc:
    def __init__ (self, MyParams ):
        self.CP = MyParams
    def __call__ (self, mt, lna, lnb):
        subParams = CostParams(self.CP.centWt,self.CP.centNoSplitWt,self.CP.centSplitMDWt,self.CP.centSplitDDWt,self.CP.centSplitMDDist,self.CP.centSplitDDDist,
                    self.CP.volWt,self.CP.volNoSplitWt,self.CP.volSplitWt,self.CP.volNoSplitMult,self.CP.volSplitMult,self.CP.splitWt)
        dc = DefaultDaughterCostFunc(costParams=subParams)
        defCost = dc(mt, lna, lnb)
        # solidity cost - mother expected to be <<1 (closer to .68)
        # better the smaller it is ..
        # best current cost 8K
        # solCost = 100*mt.solidity
        # when two daughters - the two distance (m,d1) and (m,d2) should be similar
        symCost = 0
        angCost = 0
        solCost = 0
        aspectCost = 0
        meanICost = 0
        stdICost = 0
        # mdDist cost: two formulas selected by mdDistSquared flag
        #   False (16to32): Euclidean distance, penalise abs(dist)           when dist   > 20
        #   True  (8to16):  squared  distance, penalise abs(sqDist - ref)   when sqDist > 20
        mdDistCost = 0
        # SPLIT COSTS
        if (lna is not None) and (lnb is not None):
            # mother should have high meanI and high stdI
            meanICost = -(mt.meanI)
            # mother should have low solidity, sphericity and high aspect ratio (1.8)
            aspectCost = (mt.aspectratio - self.CP.aspTarget) ** 2
            solCost = (mt.sphericity - 0.88)*(mt.sphericity - 0.88)
            if getattr(self.CP, 'mdDistSquared', False):
                # From8 formula: squared distance, threshold 20, reference mdDistRef
                md1 = centroidCost(mt.centroid, lna.centroid)   # squared distance
                md2 = centroidCost(mt.centroid, lnb.centroid)
                ref = self.CP.mdDistRef
                if (md1 > 20):
                    mdDistCost = mdDistCost + abs(md1 - ref)
                if (md2 > 20):
                    mdDistCost = mdDistCost + abs(md2 - ref)
            else:
                # 16to32 formula: Euclidean distance, threshold 20
                md1 = Distance(mt.centroid, lna.centroid)
                md2 = Distance(mt.centroid, lnb.centroid)
                if (md1 > 20):
                    mdDistCost = mdDistCost + abs(md1)
                if (md2 > 20):
                    mdDistCost = mdDistCost + abs(md2)
            aveDist = (md1 + md2) / 2
            if (aveDist > 1e-5):
                symCost =  math.fabs(md1 - md2) / aveDist
            else:
                symCost =  math.fabs(md1 - md2)
            # angle cost
            v1 = np.asarray(mt.centroid) - np.asarray(lna.centroid)
            v2 = np.asarray(mt.centroid) - np.asarray(lnb.centroid)
            v1_mag = math.sqrt(sum(pow(element, 2) for element in v1))
            v2_mag = math.sqrt(sum(pow(element, 2) for element in v2))
            if ((v1_mag * v2_mag) > 1e-5):
                test = np.dot(v1, v2) / (v1_mag * v2_mag)
                if (test <= 1) and (test >= -1):
                    angle = math.acos(np.dot(v1, v2) / (v1_mag * v2_mag))
                else:
                    angle = 0
            else:
                angle = 0
            angCost = math.fabs(angle - self.CP.angTarget)

        # 23K weight for solCost  14/16,  24.5K 13/16
        totalCost = self.CP.mdDistWt * mdDistCost + self.CP.angWt * angCost + defCost + self.CP.symWt * symCost + self.CP.aspWt * aspectCost + self.CP.meanIWt * meanICost
        # totalCost = defCost
        return totalCost

