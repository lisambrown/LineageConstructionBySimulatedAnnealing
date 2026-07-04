# sim_anneal.py          18.7.22
# version 0.6            6.11.22

# apply simulated annealing to nucleus tracking

# version 0.1     24.5.22
#   initial version
#
# version 0.2      9.6.22
#   improve data structures
#   add balanced-forest penalty
#      impose balanced-forest condition on each 2^n doubling step individually
#
# version 0.3     18.7.22
#   support for nuclei volumes and balanced forests
#      but not yet in metropolis step
#
# version 0.4     11.8.22
#   full support for centroids, nuclei volumes, and balanced forests
#   various structural improvements
#   documentation
#
# version 0.5     25.8.22
#   finer-grained CostParams
#   factor out computeKLeaf()
#   add convenience label property of LinkedNucleus
#
# version 0.6     6.11.22
#   flag to disable split cost
#   encapsulate non-split cost as user-provided mother-dauughter-cost function



# typical usage:
#
#     nucCounts = [2, 3, 4, 4, 6, 8]   # numbers of nuclei in each frame from external data source
#     tracks = Tracks (nucCounts)      # instantiate Tracks data structure
#     for  nd in nuclei_data:          # nuclei centroids and volumes from external data source
#         <initialize tracks>          # code to initialize links, centroids and volumes in Tracks data structure
#     tracks.computeKLeaf()            # compute internal kLeaf values to match links
#     anneal (tracks)                  # apply simulated-annealing algorithm to tracks
#
# tracks now contains optimized tracks in the form of linked nuclei


# simulated-annealing optimiztion of cell-nucleus lineage forests to perform
# frame-to-frame tracking of dividing cell nuclei

# the lineage forest is represented by the following data structure:
#
# the forest is an instance of the Tracks class.  Tracks contains a
# sequence of frames (the Frame class) that correspond to time-slices
# (frames) of the 3d-plus-time microscopic images of the developing
# embryo.  each frame consists of a list of nuclei (class LinkedNucleus)
# that are annotated with their 3d centroids (locations) and their volumes.
# a LinkedNucleus can be annotated with additional properties after
# instantiation to provide further information for use by a custom cost
# function.  a given nucleus in one frame is linked to its daughter or
# daughters in the subsequent frame (as well as its mother in the preceding
# frame).
#
# the purpose of the simulated annealing algorithm is to choose these
# mother-daughter links "optimally" so that the resulting forest best
# tracks how the nuclei move and split over time.  it is a forest because
# each nucleus in the first frame gives rise to -- and is the root of -- an
# individual tree in the forest.
#
# the Tracks data structure requires certain properties to hold.  each Frame
# contains a fixed nmber of nuclei.  the nuclei-counts of the first and last
# Frames must be exact powers of two.  the nuclei-counts must not decrease
# as one moves through the Frames from first to last -- that is, nuclei may
# divide, but not disappear. last, every intervening power of two between
# those of the first and last Frames must appear in Tracks.  that is, for
# example, [4, 4, 8], [4, 5, 8], and [4, 8, 16] would be legal sets of nuclei
# counts, while [4, 5, 16] would not.  these constraints on the nuclei counts
# of a Tracks data structure are validated and enforced when Tracks is constructed.
# the simulated-annealing algorithm assumes and relies on these constraints during
# its operation.  the power-of-two conditions are part of the "balanced-forest"
# infrastructure (see below) and are not required when balanced-forest support
# is turned off (by instantiating Tracks with supportSplit = False).
#
# additionally, each LinkedNucleus is required to have one or two daughter
# nuclei (except for those in the last Frame) and each LinkedNucleus is
# required to have exactly one mother nucleus (except for those in the first
# frame).  Tracks is constructed to satisfy these constraints, and the
# simulated-annealing algorithm maintains these constraints as it relinks
# the LinkedNuclei.
#
# the constructor of Tracks takes a list of nuclei counts that is validated
# as described above.  it initializes the nuclei centroids and volumes to
# zero and one, respectively.  the expected usage is that an instance of
# Tracks will be constructed with the list of nuclei counts from the data
# set whose nuclei one wishes to track, and then one will iterate over the
# LinkedNucleus objects in Tracks, setting their centroids and volumes to
# values from the data set.  Tracks also takes a supportSplit constructor
# argument (defaults to True) that enables or disables balanced-forest support.
#
# the simulated-annealing algorithm is implemented in the anneal() function:
#
# anneal (tracks, epochs, startTemp, stopTemp, daughterCostFunc, splitWt, history)
#
# tracks is the Tracks data structure that anneal() seeks to optimize by
# relinking the nuclei, updating tracks in place.
#
# epochs is the number of "epochs" to run (see below).
#
# startTemp and stopTemp are the initial and final "temperatures" used in
# the annealing algorithm.  the temperature is reduced exponentially from
# startTemp to stopTemp over the course of the annelaing process (on a
# per-epoch basis).
#
# the cost to be minimzed is determined by daughterCostFunc and splitWt.
# daughterCostFunc returns the cost contributed by a mother and its daughter(s).
# splitWt is the weight in the total cost of the unbalanced-forest penalty.
# daughterCostFunc defaults to DefaultDaughterCostFunc instatiated with
# costParams = CostParams() with its default values.  (daughterCostFunction
# may be a function or an instance of a callable class.  the callable-class
# vearsion is illustrated by DefaultDaughterCostFunc.)
#
# costParams is the set of parameters (in a CostParams instance) that
# control the objective function (see below).  (although CostParams
# contains a splitWt property, splitWt must be passed to anneal() and
# the various cost functions separately from daughterCostFunction.)
#
# history specifies whether the evolution of the annealing process should
# be recorded (for testing and visualization purposes).
#
# discussion of the simulated-annealing algorithm:
#
# simulated annealing is an algorithm for minimizing an objective function
# (cost function) over a configuration space.  in our case, a point in
# configuration space is a proposed set of tracks, that is, the set of links
# in a Tracks data structure that link mother and daughter nuclei (subject
# to the linking constraints discussed above).  the configuration space is
# therefore the set of all (legal) candidate tracks.
#
# the cost function should be chosen to take on its minimum for the correct
# tracking, that is, for the set of links that maps each mother nucleus to its
# correct daughter nuclei in the next Frame.  if the algorithm successfully
# minimizes such a cost function, it will, by definintion, have constructed
# the correct set of tracks.
#
# our default cost function consists of three terms:  the first is the sum of
# squared deviations of the euclidean distances between the mother nuclei and their
# daughters from a target distance plus the sum of squared deviations of the
# euclidean distances between the two daughters from a target (when the mother
# splits).  these target distances are a hard-wired zero for the unsplit-mother-daughter
# distance and the CostParams properties centSplitMDDist and centSplitDDDist for
# split-mother-daughter and daughter-daughter distances, respectively.  the second
# is the squared mismatch between the volume of the mother nuclei and their daughters.
# if a nucleus splits into two daughters, the sum of the daughter volumes is matched
# against CostParams.volSplitMult times the mother volume, otherwise the volume of
# the single daughter is matched against CostParams.volNoSplitMult times the mother
# volume.  these first two terms depend only on a mother and its daughters and are
# computed by daughterCostFunc. the third term is the "split" or "unbalanced forest"
# penalty, enabled if supportSplit = True (the default), and is computed separately
# from the mother-daughter cost terms in tracksCost() and Tracks.deltaCost().

# specifically, the unbalanced-forest penalty is computed as follow:  the ancestor
# nuclei in one Frame with 2^n nuclei are each expected to have exactly two descendants
# in the subsequent "power-of-two" Frame that has 2^(n+1) nuclei.  the split penalty
# is then the square of the excess over two descendants that each power-of-two-Frame
# nucleus has in the subsequent power-of-two Frame, summed over all (except for the
# last frame) power-of-two-Frame nuclei.  (the frames in which the split penalty is
# calculated are flagged with splitRoot = True in the Frame class.)
#
# these three terms are combined togheter into a composite cost function with the
# weights CostParams.centWt and CostParams.volWt (when using DefaultDaughterCostFunc)
# and the splitWt passed to tracksCost() and Tracks.deltaCost().  (splitWt must be set
# to None when supportSplit = False.)  sub-terms of centroid and volume cost terms are
# further weighted with the finer-grained weights in CostParams.
#
# in typical usage, simulated annealing proceeds by randomly choosing a candidate
# move in configuration space from the current point, point A, to some new point,
# point B.  it is not necessary to condsider as a candidate moving from some point
# A to any other point, say point Z, in configuration space, but it must be possible
# to move from point A to point Z with a sequence of candidate moves.
#
# the probability of choosing A->B as a candidate is required to be the same
# as choosing B->A.  if the cost function at point B is lower than that at A,
# the candiate move is accepted.  if the cost function is higher, the candidate
# move is accepted with probability exp (-delta_cost / temperature), where
# delta_cost = cost (B) - cost (A).  if the candidate move is accepted, the
# point in configuration space is set to point B (otherwise it remains point A)
# and the process of randomly selecting and accepting / rejecting candidate moves
# is repeated.  this update process is biased to reduce the cost function -- all
# moves that decrease the cost are accepted -- but permits the algorithm to explore,
# in principle, all of configuration space, as cost-increasing moves can be accepted,
# but with exponentially decreasing probability, as the annealing temperature is
# decreased toward zero.
#
# simulated annealing is generally applied to problems where a point in configuration
# space is itself a complicated object -- not something as simple as, say, a single
# three-dimensional point.  this is true for our use case where a point in configuration
# space is the entire forest of linked nuclei.  because points in configuration space
# are complicated, it can be expensive to compute the cost function from scratch for
# some given point.  for this reason, proposed candidate moves are chosen to be in some
# sense "local."  specifically, point B should be obtained from point A by making a
# "small" change, and delta_cost should be able to be computed from that change, rather
# than by incurring the expense of computing the entire cost directly for point B.
#
# our candidate moves consist of swapping the mother links of two nuclei in a given
# frame.  that is, nucleus A is relinked to the mother of nucleus B and vice versa.
# we also include "half-swaps" as candidate moves where a daughter of a two-daughter
# mother is relinked to a one-daughter mother.  it is possible to move from any point
# in configuration space to any other with a sequence of such swap and half-swap moves.
#
# it is cheap to compute the centroid and volume contributions to delta_cost for such
# swap and half-swap moves -- it is only necessary to look at the two nuclei being
# swapped and their mothers.  computing the change in the "split" penalty (unbalanced
# forest penalty), however, is more involved, as the change to mothers' numbers of
# descendants must be propagated up to the preceding power-of-two Frame (where the
# split penalty is ultimately computed).  this is made more manageable by storing in
# each LinkedNucleus a kLeaf property that is the number of that nucleus's descendants
# in the subsequent power-of-two Frame.  kLeaf is initialized when Tracks is constructed,
# and updated (with the changes propagated up to the preceding power-of-two Frame) when
# a candidate move is accepted.  (kLeaf is also recomputed by tracksShuffle(), see below.)
#
# the balanced-forest condition -- that each power-of-two-Frame nucleus has exactly two
# descendants in the subsequent power-of-two Frame -- is not enforced as a constraint
# because our swap and half-swap moves would not be able move to all points in configuration
# space in the presence of such a constraint.  instead the balanced-forest condition is
# achieved via a sufficiently-heavily-weighted split penatly in the cost function.
#
# candidate moves are selected by randonly choosing a pair of same-frame nuclei to
# potentially swap -- half-swap candidates are indicated by a randomly chosen boolean
# value that only takes effect if one of the two mothers has one daughter and the
# other, two.  ignoring the half-swap modifier, there is a given number of possible
# candidate moves, namely, the number of same-frame pairs.  an epoch is defined (rather
# arbitrarily) as proposing (and then accepting or rejecting) a number of randomly
# chosen candidates equal to the number of same-frame pairs.  there is no implication
# that all same-frame pairs will be proposed in a epoch -- only that they all have
# the same random chance of being chosen.
#
# some utility and testing functions are:
#
# Tracks.computeKLeaf(): kLeaf is an internal property of LinkedNucleus that
# is used to keep track of the balanced-forest condition.  it depends on the
# mother-daughter links.  if you initialize the mother-daughter links in Tracks
# based on an external dataset (and are using supportSplit = True), you should
# call Tracks.computeKLeaf() to initialize the kLeaf values consitently with the
# links.
#
# Tracks.validate() verifies that Tracks satisfies the constraints described
# above.  because the simulated-annealing algorithm relinks the nuclei, it
# is used in testing to verify that the algorithm maintains the mother-daughter
# linking constraint.  (as a convenience, Tracks.validate (computeKLeaf = True) can
# be used to intialize kLeaf properly at validation time.)
#
# tracksCost (tracks, daughterCostFunc, splitWt) returns the value of the objective
# function for the current configuration of tracks.
#
# tracksInitRandom (tracks, ...) initializes the centroids and volumes in the
# LinkedNuclei in tracks with random values that follow (with random deviations)
# the mother-daughter relationships.  this is to generate random data sets with
# trackable structure for testing.
#
# tracksShuffle (tracks, ...) shuffles the links in tracks (respecting the
# mother-daughter linking constraints).  after running tracksInitRandom(), tracks
# is linked (approximately) optimally -- this provides (approximate) ground-truth
# tracks for the (randomly-generated) tracking problem.  tracksShuffle() then
# scrambles the links, providing an initialized but not-yet-tracked Tracks data
# structure for testing the tracking algorithm



import copy
import math
import random


class LinkedNucleus:
    def __init__ (self, ind = None):
        self.index = ind                # index of self in current frame -- for convenience
        self.mother = None              # link to mother in preceding frame
        self.daughters = [None, None]   # links to daughters in subsequent frame
        self.kLeaf = 1                  # number of leaf descendants, down to next splitRoot
        self.centroid = [0., 0., 0.]    # location of the nucleus
        self.volume = 1.0               # volume of the nucleus
        self.label = '(none)'           # convenience label not used algorithmically
        self.solidity = 1.0
        self.sphericity = 1.0
        self.aspectratio = 1.0
        self.meanI = 118
        self.stdI = 0
        return None

    def __repr__ (self):
        sMother = str (self.mother.index) if self.mother else 'None'
        sDaughter0 = str (self.daughters[0].index) if self.daughters[0] else 'None'
        sDaughter1 = str (self.daughters[1].index) if self.daughters[1] else 'None'
        sDaughters = '[' + sDaughter0 + ', ' + sDaughter1 + ']'
        retval  = 'index:         ' + str (self.index)           + '\n'
        retval += '   mother:     ' + sMother                    + '\n'
        retval += '   daughters:  ' + sDaughters                 + '\n'
        retval += '   kLeaf:      ' + str (self.kLeaf)           + '\n'
        retval += '   centroid:   ' + str (self.centroid)        + '\n'
        retval += '   volume:     ' + str (self.volume)          + '\n'
        retval += '   label:      ' + str (self.label)           + '\n'
        return retval

class Frame:
    def __init__(self, index, nucCount, nextFrame = None):   # nucCount is number of nuclei in Frame
        self.index = index
        self.nuclei = []
        self.splitRoot = False
        for  i in range (nucCount):
            self.nuclei.append (LinkedNucleus (i))
        self.nextFrame = nextFrame   # Tracks initialization is from bottom up
        if  self.nextFrame:
            self.nextFrame.prevFrame = self
        self.prevFrame = None   # can't initialize link to prevFrame until it exists
    
    def __repr__ (self):
        sNextFrame = str (self.nextFrame.index) if self.nextFrame  else 'None'
        sPrevFrame = str (self.prevFrame.index) if self.prevFrame  else 'None'
        retval  = 'index:         ' + str (self.index)           + '\n'
        retval += '   nucCount:   ' + str (len (self.nuclei))    + '\n'
        retval += '   nextFrame:  ' + sNextFrame                 + '\n'
        retval += '   prevFrame:  ' + sPrevFrame                 + '\n'
        retval += '   splitRoot:  ' + str (self.splitRoot)       + '\n'
        return retval

def intLogTwo (x):      # returns log2 (x) if it's an integer
    y = math.log2 (x)   # assumes that log2 returns an exact integer for smallish powers of 2
    if  y.is_integer():
        return int (y)
    else:
        return None

def validateNucCounts (nucCounts, supportSplit = True):   # helper function
    if  not isinstance (nucCounts, list)  or  len (nucCounts) < 2:   # not a list or too short
        return  False
    if  not isinstance (nucCounts[0], int):                          # for consistency
        return  False
    if  nucCounts[0] < 1:                                            # not positive
        return  False
    if  supportSplit:                                                # validate power-of-two structure
        if  intLogTwo (nucCounts[0]) is None:                        # not an integer power of two
            return  False
        prevCount = nucCounts[0]
        nextPow = 2 * nucCounts[0]
        for  nc in nucCounts[1:]:
            if  not isinstance (nc, int)  or  nc < prevCount  or  nc > nextPow:   # not in sequence 
                return  False
            precCount = nc
            if  nc == nextPow:
                nextPow *= 2
        return  True
    else:                                                            # just validate n-to-2n constraint
        for  i in range (1, len (nucCounts)):
            if  nucCounts[i] < nucCounts[i - 1]  or  nucCounts[i] > 2 * nucCounts[i - 1]:
                return  False
        return  True

def centroidCost (ca, cb, targ = 0.0):   # squared deviation of distance between two centroids from target
    ssq = 0.0
    for  i in range (3):
        ssq += (ca[i] - cb[i])**2
    if  targ == 0.0:
        return  ssq
    else:
        return  (math.sqrt (ssq) - targ)**2

class CostParams:   # parameters for cost function
    def __init__ (
        self,
        # default values for cost parameters
        #
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
        splitWt          =  1.0    # weight of square of descendants in excess of two
    ):
        # initialize cost parameters
        #
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

class DefaultDaughterCostFunc:
    def __init__ (self, costParams = CostParams()):
        self.costParams = costParams
    def __call__ (self, mt, lna, lnb):
        if  lna is None:
            raise ValueError ('lna is None')
        cp = self.costParams
        if  lnb is None:   # single daughter
            cCost = cp.centNoSplitWt * centroidCost (mt.centroid, lna.centroid)
            vCost = cp.volNoSplitWt * (cp.volNoSplitMult * mt.volume - lna.volume)**2
        else:   # two daughters

            cCost  = cp.centSplitMDWt * centroidCost (mt.centroid, lna.centroid, cp.centSplitMDDist)
            cCost += cp.centSplitMDWt * centroidCost (mt.centroid, lnb.centroid, cp.centSplitMDDist)
            cCost += cp.centSplitDDWt * centroidCost (lna.centroid, lnb.centroid, cp.centSplitDDDist)
            vCost  = cp.volSplitWt * (cp.volSplitMult * mt.volume - (lna.volume + lnb.volume))**2
        return  cp.centWt * cCost + cp.volWt * vCost

class Tracks:
    def __init__(self, nucCounts, supportSplit = True):   # nucCounts is list of number of nuclei in each frame
        if  not validateNucCounts (nucCounts, supportSplit):
            raise  ValueError ('invalid nucCounts (supportSplit = %s)' % supportSplit)
        self.supportSplit = supportSplit
        self.frames = []       # list of frames (time-slices); a frame contains a list of nuclei
        # initialize from bottom up with balanced divisions
        subsN = None
        fr = None   # "next" Frame
        for  i, n in enumerate (reversed (nucCounts)):
            self.frames[:0] = [Frame (len (nucCounts) - i - 1, n, fr)]
            fr = self.frames[0]   # current Frame, becomes "next" Frame for following iteration
            if  not subsN is None:   # link subsequent frame to this frame
                if  supportSplit  and  intLogTwo (n)  and  n != subsN:
                    fr.splitRoot = True
                nSplit = subsN - n
                j = 0   # working index of nucleus in this frame
                for  k, ln in enumerate (self.frames[1].nuclei):   ###  what does k do?
                    if  not self.supportSplit  or  self.frames[1].splitRoot  or  ln.kLeaf == 1:
                        if  not fr.nuclei[j].daughters[0]:
                            ln.mother = fr.nuclei[j]
                            fr.nuclei[j].daughters[0] = ln
                            fr.nuclei[j].kLeaf = 1
                            if  nSplit == 0:
                                j += 1
                        elif  nSplit > 0:
                            ln.mother = fr.nuclei[j]
                            fr.nuclei[j].daughters[1] = ln
                            fr.nuclei[j].kLeaf += 1
                            nSplit -= 1
                            j += 1
                    else:
                        while  fr.nuclei[j].daughters[0]:   # find unlinked mother
                            j += 1
                        ln.mother = fr.nuclei[j]
                        fr.nuclei[j].daughters[0] = ln
                        if  self.frames[1].splitRoot:
                            fr.nuclei[j].kLeaf = 1
                        else:
                            fr.nuclei[j].kLeaf = ln.kLeaf
                        j += 1
            subsN = n
        return None
    
    def swap (self, pair, halfSwap):   # pair is [i, [j, k]], where the nuclei to swap, j, k, are in frame i
        i = pair[0]
        if  i < 0  or  i >= len (self.frames):
            raise ValueError ('bad frames index: i = %d' % i)
        if  i == 0:   # swapping first-frame nuclei is a no-op
            return
        fr = self.frames[i]
        for  j in pair[1]:
            if  j < 0  or  j >= len (fr.nuclei):
                raise ValueError ('bad nucleus index: j = %d for frames index i = %d' % (j, i))
        j, k = pair[1]
        lnj = fr.nuclei[j]
        lnk = fr.nuclei[k]
        mj = lnj.mother
        mk = lnk.mother
        if  mj == mk:   # swapping daughters of the same mother is a no-op
            return
        # halfSwap -- move daughter from two-daughter mother to single-daughter mother
        halfSwap  =  halfSwap  and  (mj.daughters[1] is None) != (mk.daughters[1] is None)
        # djk is increase (decrease) in kLeaf for mj (mk)  (if supportSplit == True)
        if  halfSwap:   # move daughter from two-daughter mother to single-daughter mother
            if  (mj.daughters[1] is None) == (mk.daughters[1] is None):
                raise  ValueError ('invalid halfSwap, i, j, k = %d, %d, %d' % (i, j, k))
            if  mj.daughters[1] is None:
                m1 = mj
                m2 = mk
                l2 = lnk
                if  self.supportSplit:
                    djk = l2.kLeaf
                    if  fr.splitRoot:
                        djk = 1
            else:
                m1 = mk
                m2 = mj
                l2 = lnj
                if  self.supportSplit:
                    djk = -l2.kLeaf
                    if  fr.splitRoot:
                        djk = -1
            m1.daughters[1] = l2
            l2.mother = m1
            if  m2.daughters[0] == l2:
                m2.daughters[0] = m2.daughters[1]
            m2.daughters[1] = None
        else:
            if  self.supportSplit:
                djk = lnk.kLeaf - lnj.kLeaf
                if  fr.splitRoot:
                    djk = 0
            if  mj.daughters[0] == lnj:
                mj.daughters[0] = lnk
            else:
                mj.daughters[1] = lnk
            if  mk.daughters[0] == lnk:
                mk.daughters[0] = lnj
            else:
                mk.daughters[1] = lnj
            lnj.mother, lnk.mother = lnk.mother, lnj.mother
        if  not self.supportSplit  or  djk == 0:
            return
        # update kLeaf up to splitRoot
        gmj = mj
        gmk = mk
        gfr = fr.prevFrame
        while  gmj != gmk:
             gmj.kLeaf += djk
             gmk.kLeaf -= djk
             if  gfr.splitRoot:
                 break
             gmj = gmj.mother
             gmk = gmk.mother
             gfr = gfr.prevFrame
             if  gmj is None  or  gmk is None  or  gfr is None:
                 print ('i, j, k =', i, j, k)
                 print ('gmj =', gmj)
                 print ('gmk =', gmk)
                 print ('gfr =', gfr)
                 raise  ValueError ('None gmj, gmk, or gfr before splitRoot')
        return
    
    # pair[0] it the index of the frame of the daughters to swap
    # pair[1][0] and pair[1][1] are the indices of the two daughters to swap
    # halfSwap = True indicates only moving one daughter
    # daughterCostFunction returns the cost contribution of a mother-daughter(s) triple
    # splitWt = None indicates not to compute the unbalanced-forest penalty
    # otherwise weight the unbalanced-forest penalty with splitWt
    def deltaCost (self, pair, halfSwap, daughterCostFunc = DefaultDaughterCostFunc(), splitWt = 1.0):
        if  not splitWt is None  and  not self.supportSplit:
            raise  ValueError ('this Tracks not instantiated with splitWt support')
        lna = self.frames[pair[0]].nuclei[pair[1][0]]
        lnb = self.frames[pair[0]].nuclei[pair[1][1]]
        if  lna.mother == lnb.mother:   # swapping daughters of the same mother is a no-op
            return 0.0
        mta = lna.mother
        mtb = lnb.mother
        origCost = daughterCostFunc (mta, mta.daughters[0], mta.daughters[1]) + \
                   daughterCostFunc (mtb, mtb.daughters[0], mtb.daughters[1])
        # halfSwap -- move daughter from two-daughter mother to single-daughter mother
        halfSwap  =  halfSwap  and  (mta.daughters[1] is None) != (mtb.daughters[1] is None)
        if  halfSwap:   # compute daughter-cost after half-swap and mother kLeaf change
            if  mta.daughters[1] is None:
                mt1 = mta   # initial single-daughter mother (two-daughter after half-swap)
                mt2 = mtb   # initial two-daughter mother (single-daughter after half-swap)
                lnx = lnb   # daughter to half-swap
                ln1 = lna   # mt1 (non-swap) daughter
            else:
                mt1 = mtb   # initial single-daughter mother
                mt2 = mta   # initial two-daughter mother
                lnx = lna   # daughter to half-swap
                ln1 = lnb   # mt1 (non-swap) daughter
            ln2 = mt2.daughters[0]   # ln2 will be daughter of mt2 that is not being swapped
            if  ln2 is lnx:
                ln2 = mt2.daughters[1]
            newCost = daughterCostFunc (mt1, ln1, lnx) + daughterCostFunc (mt2, ln2, None)
            # half-swap version of  split-cost computation if activated
            if  not splitWt is None:
                # dab is increase (decrease) in kLeaf for mta (mtb)
                dab = lnx.kLeaf
                if self.frames[pair[0]].splitRoot:
                    dab = 1
                if  not mta.daughters[1] is None:   # mta.kLeaf is reduced
                    dab = -dab
        else:   # full swap
            # lnc and lnd are mta and mtb second (non-swap) daughters (or None)
            lnc  =  mta.daughters[1]  if  lna is mta.daughters[0]  else  mta.daughters[0]
            lnd  =  mtb.daughters[1]  if  lnb is mtb.daughters[0]  else  mtb.daughters[0]
            newCost = daughterCostFunc (mta, lnb, lnc) + daughterCostFunc (mtb, lna, lnd)
            # full-swap version of split-cost computation if activated
            if  not splitWt is None:
                # dab is increase (decrease) in kLeaf for mta (mtb)
                if self.frames[pair[0]].splitRoot:
                    dab = 0
                else:
                    dab = lnb.kLeaf - lna.kLeaf
        dCost = newCost - origCost
        if  not splitWt is None:   # add dSplit to dCost
            dSplit = 0.0
            if  dab != 0:   # propagate dab up to splitRoot
                gma = mta
                gmb = mtb
                gfr = self.frames[pair[0]].prevFrame
                while  not gfr.splitRoot:
                    gma = gma.mother
                    gmb = gmb.mother
                    gfr = gfr.prevFrame
                    if  gma == gmb:
                        break
                if  gma != gmb:
                    dSplit  = max (0.0, (gma.kLeaf + dab) - 2.0)**2 + max (0.0, (gmb.kLeaf - dab) - 2.0)**2
                    dSplit -= max (0.0, gma.kLeaf - 2.0)**2 + max (0.0, gmb.kLeaf - 2.0)**2
            dCost += splitWt * dSplit
        return  dCost
    
    
    def computeKLeaf (self):   # compute kLeaf for nuclei in otherwise valid Tracks
        if  not self.supportSplit:
            raise ValueError ('supportSplit is False')
        for  fr in reversed (self.frames):
            if  not fr.nextFrame:   # last-frame kLeaf is 1
                for  ln in fr.nuclei:
                    ln.kLeaf = 1
            else:
                for  ln in fr.nuclei:
                    if  fr.nextFrame.splitRoot:
                        ln.kLeaf = 1
                        if  not ln.daughters[1] is None:
                            ln.kLeaf += 1
                    else:
                        ln.kLeaf = ln.daughters[0].kLeaf
                        if  not ln.daughters[1] is None:
                            ln.kLeaf += ln.daughters[1].kLeaf
        return  None
    
    
    def validate (self, computeKLeaf = False):   # verify validity of nuclei tracks and optionally compute kLeaf
        if  self.frames is None  or  len (self.frames) <= 0:
            print ('invalid Tracks: frames empty or None')
            return  False
        for  i, fr in enumerate (self.frames):
            if  fr is None:
                print ('invalid Tracks: frames[%d] is None' % i)
                return False
            if  fr.index is None  or  fr.index != i:
                print ('invalid Tracks: frames[%d].index inconsistent or None' % i)
                return False
            if  fr.nuclei is None  or  len (fr.nuclei) <= 0:
                print ('invalid Tracks: frames[%d] is empty' % i)
                return False
            if  self.supportSplit:   # validate splitRoot
                nc = len (fr.nuclei)   # validate splitRoot
                if  i == len (self.frames) - 1:
                    if  fr.splitRoot == True:
                        print ('invalid Tracks: frames[%d].splitRoot = %s for last frame' % (i, fr.splitRoot))
                        return False
                else:
                    if  intLogTwo (nc):
                        if  fr.splitRoot != (nc != len (fr.nextFrame.nuclei)):
                            print ('invalid Tracks: bad frames[%d].splitRoot = %s for power-of-two frame' % (i, fr.splitRoot))
                            return False
                    elif  fr.splitRoot == True:
                        print ('invalid Tracks: True frames[%d].splitRoot for non-power-of-two frame' % i)
                        return False
            for  j, ln in enumerate (fr.nuclei):
                if  ln is None  or  ln.index is None  or  ln.index != j:
                    print ('invalid Tracks: ln is None or ln.index is inconsistent or None, i, j = %d, %d' % (i, j))
                    return False
                if  i > 0:   # validate mother links
                    if  ln.mother is None:
                        print ('invalid Tracks: ln.mother is None, i, j = %d, %d' % (i, j))
                        return False
                    ds = ln.mother.daughters
                    if  ds is None  or  len (ds) != 2:
                        print ('invalid Tracks: ln.mother.daughters is None or invalid, i, j = %d, %d' % (i, j))
                        return False
                    if  ds[0] is None:
                        print ('invalid Tracks: ln.mother.daughters[0] is None, i, j = %d, %d' % (i, j))
                        return False
                    if  ds[1] is None  and  not ds[0] is ln:
                        print ('inconsistent single-daughter link, i, j = %d, %d' % (i, j))
                        return False
                    if  ds[0] is ds[1]:
                        print ('duplicate single-daughter link, i, j = %d, %d' % (i, j))
                        return False
                    if  not ds[0] is ln  and  not ds[1] is ln:
                        print ('inconsistent two-daughter link, i, j = %d, %d' % (i, j))
                        return False
                if  i < len (self.frames) - 1:   # validate daughter links
                    ds = ln.daughters
                    if  ds is None  or  len (ds) != 2  or  ds[0] is None:
                        print ('invalid Tracks: frames[%d][%d].daughters invalid or None' % (i, j))
                        return False
                    if  not ds[0].mother is ln:
                        print ('invalid Tracks: frames[%d][%d] inconsistent first-daughter link' % (i, j))
                        return False
                    if  ds[1]  and  not ds[1].mother is ln:
                        print ('invalid Tracks: frames[%d][%d] inconsistent second-daughter link' % (i, j))
                        return False
                    if  ds[0] is ds[1]:
                        print ('invalid Tracks: frames[%d][%d] duplicate two-daughter link' % (i, j))
                        return False
                    if  self.supportSplit  and  not computeKLeaf:   # validate existing kLeaf
                        if  fr.nextFrame.splitRoot:
                            kl = 1
                            if  not ds[1] is None:
                                kl += 1
                        else:
                            kl = ds[0].kLeaf
                            if  not ds[1] is None:
                                kl += ds[1].kLeaf
                        if  ln.kLeaf != kl:
                            print ('invalid Tracks: frames[%d][%d] bad kLeaf' % (i, j))
                            return False
                if  self.supportSplit  and  not computeKLeaf:   # validate existing kLeaf
                    if  i == len (self.frames) - 1:
                        if  ln.kLeaf != 1:
                            print ('invalid Tracks: frames[%d][%d] bad kLeaf for last frame' % (i, j))
                            return False
        if  computeKLeaf:   # compute valid kLeaf for otherwise valid Tracks
            self.computeKLeaf()
        return True


def tracksCounts (tracks):   # returns nFrames, nNuclei, nLinks
    nFrames = len (tracks.frames)
    nNuclei = 0
    nLinks = 0
    for  fr in tracks.frames:
        nNuclei += len (fr.nuclei)
        for  ln in fr.nuclei:
            for  dg in ln.daughters:
                nLinks  +=  not dg is None
    return  nFrames, nNuclei, nLinks


def tracksCost (tracks, daughterCostFunc = DefaultDaughterCostFunc(), splitWt = 1.0):
    cCost = 0.0
    vCost = 0.0
    sCost = 0.0
    cost = 0.0
    for  fr in tracks.frames:
        if  fr.nextFrame is None:   # last frame has no daughters or splits
            continue
        for  ln in fr.nuclei:
            cost += daughterCostFunc (ln, ln.daughters[0], ln.daughters[1])
            if  fr.splitRoot  and  not splitWt is None:
                cost += splitWt * max (0.0, ln.kLeaf - 2.0)**2
    return  cost


def tracksInitRandom (tracks, locScale = 100.0, offsetScale = 5.0, splitScale = 10.0, initVol = 10.0, volFrac = 1.2, volFracScale = 0.10):
    for  ln in tracks.frames[0].nuclei:   # initialize first frame with random centroids and volumes
        for  i in range (3):
            ln.centroid[i] = random.normalvariate (0.0, locScale)
        vol = initVol
        ln.volume = random.normalvariate (vol, volFracScale * vol)
    for  i in range (len (tracks.frames) - 1):   # initialize daughter centroids and volumes
        for  ln in tracks.frames[i].nuclei:
            ds = ln.daughters
            if  ds[1] is None:   # single daughter
                scale = offsetScale
                vF = 1.0
            else:                   # split
                scale = splitScale
                vF = volFrac / 2
            for  dg in ds:
                if  dg is None:
                    continue
                for  j in range (3):
                    dg.centroid[j] = ln.centroid[j] + random.normalvariate (0.0, scale)
                vol = volFrac * ln.volume
                dg.volume = random.normalvariate (vol, volFracScale * vol)
    return None


def tracksShuffle (tracks, shuffleFirst = False):   # shuffle the linked-nuclei within each frame of tracks
    for  i, fr in enumerate (tracks.frames):
        if  shuffleFirst  or  i > 0:   # optionally shuffle first frame
            random.shuffle (fr.nuclei)
            for  j, ln in enumerate (fr.nuclei):   # update indices to self
                ln.index = j
        if  fr.nextFrame:   # randomly relink daughters
            mNuc = fr.nuclei
            dNuc = fr.nextFrame.nuclei
            mN = len (mNuc)
            dN = len (dNuc)
            if  dN < mN:
                raise ValueError (
                    f'Cell count decreases from frame {i} ({mN} cells) to frame {i+1} ({dN} cells). '
                    f'This is likely a segmentation artifact. '
                    f'tracksShuffle requires non-decreasing cell counts (cells can split but not disappear).')
            mInds = random.sample (range (mN), dN - mN) + list (range (dN))
            mInds.sort()
            dInds = list (range (dN))
            random.shuffle (dInds)
            for  ln in mNuc:
                ln.daughters[0] = ln.daughters[1] = None
            for  mI, dI in zip (mInds, dInds):   # link mothers to shuffled daughters
                if  mNuc[mI].daughters[0] is None:
                    mNuc[mI].daughters[0] = dNuc[dI]
                else:
                    mNuc[mI].daughters[1] = dNuc[dI]
                dNuc[dI].mother = mNuc[mI]
    if  tracks.supportSplit:   # recompute kLeaf ...
        tracks.computeKLeaf()
    return None


def makeNode (ln):   # package history node as dict for json
    lnCopy = copy.deepcopy (ln)   # because LinkedNucleus changes from epoch to epoch
    dInds = [-1, -1]
    for  i, dg in enumerate (lnCopy.daughters):
        if  dg:
            dInds[i] = dg.index
    return {'index' : lnCopy.index, 'daughters' : dInds, 'centroid' : lnCopy.centroid}

def makeHistRecord (epoch, cost, tracks):   # package history record as dict / list for json
    histRecord = {'epoch' : epoch, 'cost' : cost, 'frames' : []}
    for  fr in tracks.frames:
        nodes = []
        for  ln in fr.nuclei:
            nodes.append (makeNode (ln))
        histRecord['frames'].append (nodes)
    return histRecord


def anneal (tracks, epochs = 20, startTemp = 1.e5, stopTemp = 1.e1, daughterCostFunc = DefaultDaughterCostFunc(), splitWt = 1.0, history = False, earlyStopIter = 500):
    if  history:   # record optimization path
        historyList = []
    pairsList = []   # to facilitate randomly selecting pair to swap
    for  i, fr in enumerate(tracks.frames):
        if  i == 0:  # don't swap pairs in first frame
            continue
        for  j in range (0, len (fr.nuclei)):
            for  k in range (j + 1, len (fr.nuclei)):
                pairsList.append ([i, [j, k]])
    #
    tempFac = (stopTemp / startTemp)**(1 / max (epochs - 1, 1))   # temperature reduction factor
    temp = startTemp
    cost = tracksCost (tracks, daughterCostFunc, splitWt)
    bestEpoch = -1
    bestEpochCost = cost
    bestUpdate = [-1, -1]
    bestUpdateCost = cost
    iCountBestUpdateCost = 0
    lastBestUpdateCost = 0
    gt_tracks = copy.deepcopy(tracks)
    for  i in range (epochs):   # one epoch is len (pairsList) randomly-chosen update trials
        if  history:
            historyList.append (makeHistRecord (i, cost, tracks))
        for  j in range (len (pairsList)):
            pair = random.choice (pairsList)
            halfSwap = random.choice ([True, False])
            delta = tracks.deltaCost (pair, halfSwap, daughterCostFunc, splitWt)
            if  delta <= 0.0  or  random.random() < math.exp (-delta / temp):   # metropolis step
                tracks.swap (pair, halfSwap)
                cost += delta
                if  cost < bestUpdateCost:
                    bestUpdate = [i, j]
                    bestUpdateCost = cost
        cost = tracksCost (tracks, daughterCostFunc, splitWt)
        if  cost < bestEpochCost:
            bestEpoch = i
            bestEpochCost = cost
            gt_tracks = copy.deepcopy(tracks)
        temp *= tempFac
        if  (i % 10) == 0:
            print ('bestUpdate = [%d, %d], bestUpdateCost = %f' % (bestUpdate[0], i, bestUpdateCost))
        if  abs(bestUpdateCost - lastBestUpdateCost) < 1e-6:
            iCountBestUpdateCost += 1
        else:
            iCountBestUpdateCost = 0
        lastBestUpdateCost = bestUpdateCost
        if  iCountBestUpdateCost > earlyStopIter:
            if splitWt is not None:
                balance_cost = sum(
                    splitWt * max(0.0, ln.kLeaf - 2.0) ** 2
                    for fr in gt_tracks.frames
                    for ln in fr.nuclei
                    if fr.splitRoot
                )
                if balance_cost > 0:
                    iCountBestUpdateCost = 0
                    continue
            cost = tracksCost(gt_tracks, daughterCostFunc, splitWt)
            print ('End Condition Met', cost)
            return gt_tracks
    #
    print ('bestUpdate = [%d, %d], bestUpdateCost = %f' % (bestUpdate[0], bestUpdate[1], bestUpdateCost))
    if  history:
        historyList.append (makeHistRecord (epochs, cost, tracks))
    return gt_tracks

