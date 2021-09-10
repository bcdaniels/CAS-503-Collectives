# informationDecomposition.py
#
# Bryan Daniels
# 2021/6/18
#
# Implement Williams and Beer pairwise mutual information decomposition.
#
# Branched from mutualInfo.py (4.2.2012)

import numpy as np
import warnings

def arrayFlatten(arr):
    return np.ndarray.flatten(arr)

# replacement for pylab.find
def find(condition):
    res, = np.nonzero(np.ravel(condition))
    return res

# 1.6.2012
# from EntropyEstimates.py
def naiveEntropy(dist):
    """
    In bits.
    """
    eps = 1.e-6
    if abs(1. - sum(dist)) > eps:
        raise Exception("Distribution is not normalized.")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore") # ignore "invalid value" warnings
        ent = -np.sum( np.nan_to_num(dist*np.log2(dist)) )
    return ent

# see EntropyEstimates.py for implementation
def meanAndStdevEntropyNem():
    raise Exception("NSB entropy estimates are not yet implemented.")

# 7.20.2012
class infoContainer():
    
    def __init__(self,):
        raise notImplementedError
    
    def _calculateNvec(self,possibleValues=None):
        if possibleValues is not None:
            # ensure each state is counted once, then remove one from each at the end
            trialVals = list(self.trialValues) + list(possibleValues)
        else:
            trialVals = self.trialValues
            
        # taken from EntropyEstimates.fights2kxnx "efficient version"
        tvSorted = np.sort(trialVals)
        diffJumpLocs = find( (tvSorted[1:]-tvSorted[:-1])>0. )
        nList = list( diffJumpLocs[1:]-diffJumpLocs[:-1] )
        if len(diffJumpLocs) > 0:
            # add for beginning and end
            nList.insert(0,diffJumpLocs[0]+1)
            nList.append(len(trialVals)-diffJumpLocs[-1]-1)
        else: # all values are equal
            nList.append(len(trialVals))
            
        if possibleValues is not None:
            nList = np.array(nList) - 1
        self.nVec = np.array(nList)
    
    def _setupEmpty(self,):
        self.trialValues = []
        self.numTrials = 0
        self.numDimensions = 0
        self.maxVal = 0
        self._calculateNvec()
    
    def calculateEntropy(self,naive=True,save=True):
        """
        naive (True)        : True uses naive entropy estimation,
                              fast but potentially biased (and
                              gives no estimate of its error).
                              False uses NSB entropy estimation.
                              None uses
                              the naive method when there are at
                              least 10 samples for every possibility.
        """
        if hasattr(self,'savedEntropy'):
            return self.savedEntropy
        if self.numTrials == 0:
            return (np.nan,np.nan)
        if naive is None:
            naive = np.all(self.nVec > 10)
        if naive:
            entropy = (naiveEntropy(self.nVec/float(sum(self.nVec))),0.)
        else:
            #entropy = meanAndStdevEntropyNem(self.nVec,K=self.maxVal)
            raise Exception("NSB entropy estimates are not yet implemented.")
        if save: self.savedEntropy = entropy
        return entropy
  

# 7.20.2012
class binaryInfo(infoContainer):
    def __init__(self,binaryData,maxVal=None):
        """
        binaryData          : (#trials)x(#dimensions)
        maxVal (None)       : Maximum representable number.
                              Defaults to 2^(#dimensions).
        """
        if np.prod(np.shape(binaryData)) == 0:
            self._setupEmpty()
            return
        b = np.array(binaryData,dtype=int)
        if     (max(arrayFlatten(b)) > 1)              \
            or (min(arrayFlatten(b)) < 0):
            raise Exception("binaryData is not in a recognized binary format.")
        if len(np.shape(b)) == 1:
            N = len(b)
            b = b.reshape((N,1))
        if len(np.shape(b)) != 2:
            raise Exception("binaryData should have 1 or 2 dimensions, not "\
                +str(len(np.shape(b))))
        self.numDimensions = len(b[0])
        self.numTrials = len(b)
        if maxVal is None: self.maxVal = 2**self.numDimensions
        else: self.maxVal = maxVal
        self.trialValues = binaryToDecimal(b)
        self._calculateNvec()
    
def binaryToDecimal(binaryData):
    # taken from SparsenessTools.fight2number
    return np.sum(                                       \
        2**np.arange(float(len(binaryData[0])))[::-1]    \
        * np.array(binaryData), axis=1 )

# 7.20.2012
class discreteInfo(infoContainer):
    def __init__(self,discreteData,maxVal=None):
        """
        discreteData        : length = #trials
        maxVal (None)       : Defaults to
                              len(np.unique(discreteData))
        """
        if np.prod(np.shape(discreteData)) == 0:
            self._setupEmpty()
            return
        if len(np.shape(discreteData)) > 1:
            raise Exception("discreteData should have 1 dimension, not " \
                +str(len(np.shape(discreteData))))
        self.discreteValues,self.trialValues =                  \
            np.unique(discreteData,return_inverse=True)
        if maxVal is None: self.maxVal = len(self.discreteValues)
        else: self.maxVal = maxVal
        self.numTrials = len(discreteData)
        self._calculateNvec()

# 9.7.2012
class continuousInfo(infoContainer):
    def __init__(self,continuousData,numBins):
        """
        continuousData      : length = #trials
        numBins             : Data is binned into numBins bins of
                              equal width.  The first bin has left
                              edge at the minimum value in the data;
                              the last bin has right edge at the
                              maximum value in the data.
        """
        if np.prod(np.shape(continuousData)) == 0:
            self._setupEmpty()
            return
        if len(np.shape(continuousData)) > 1:
            raise Exception("continuousData should have 1 dimension, not "\
                +str(len(np.shape(continuousData))))
        mn,mx = min(continuousData),max(continuousData)
        binEdges = np.linspace(mn,mx,numBins+1)
        d = np.digitize(continuousData,binEdges)
        # digitize maps max to next bin; fix:
        d[find(d==numBins+1)] = numBins
        self.trialValues = d - 1
        self.maxVal = numBins
        self.numTrials = len(continuousData)
        self._calculateNvec()

# 7.20.2012
class jointInfo(infoContainer):
    def __init__(self,infoContainer1,infoContainer2):
        IC1, IC2 = infoContainer1, infoContainer2
        IC1vals, IC2vals = IC1.trialValues, IC2.trialValues
        if len(IC1vals) != len(IC2vals):
            raise Exception("infoContainers must have equal numbers of trials")
        self.maxVal = IC1.maxVal * IC2.maxVal
        
        self.trialValues =                                      \
            IC1.maxVal*IC2.trialValues + IC1.trialValues
        self.numTrials = len(self.trialValues)
        self._calculateNvec()
        
class conditionalInfo(infoContainer):
    def __init__(self,infoContainerX,infoContainerY,stateIndexY):
        """
        Conditional distribution over X given Y = the state referred to
        by stateIndexY.
        
        (Has currently only been tested with discreteValues infoContainers.)
        """
        ICX, ICY = infoContainerX, infoContainerY
        ICXvals, ICYvals = ICX.trialValues, ICY.trialValues
        if len(ICXvals) != len(ICYvals):
            raise Exception("infoContainers must have equal numbers of trials")
        stateYtrials = find(ICYvals == stateIndexY)
        
        self.maxVal = ICX.maxVal
        self.trialValues = ICX.trialValues[stateYtrials]
        
        self.numTrials = len(self.trialValues)
        self._calculateNvec(possibleValues=range(ICX.maxVal))
        
        
# 7.20.2012
def mutualInfo(infoContainer1,infoContainer2,verbose=False,
    returnStds=False,**kwargs):
    """
    returnStds (False)      : Also return estimates of the
                              standard deviations for
                              S1, S2, and S12.
                              (requires NSB entropy estimation)
    """
    ICboth = jointInfo(infoContainer1,infoContainer2)
    S1 = infoContainer1.calculateEntropy(**kwargs)
    if verbose: print("S1,stdS1 =",S1)
    S2 = infoContainer2.calculateEntropy(**kwargs)
    if verbose: print("S2,stdS2 =",S2)
    S12 = ICboth.calculateEntropy(**kwargs)
    if verbose: print("S12,stdS12 =",S12)
    if returnStds:
        return S1[0] + S2[0] - S12[0], (S1[1],S2[1],S12[1])
    else:
        return S1[0] + S2[0] - S12[0]

def discreteMutualInfo(data1,data2,maxVal1=None,maxVal2=None,**kwargs):
    """
    Using data sampled simultaneously from two discrete distributions,
    return an estimate in bits of the mutual information between the two
    distributions.
    
    data1 and data2 should have the same length.
    
    maxVals are passed to the discreteInfo function, and other kwargs
    are passed to the mutualInfo function.
    """
    assert(len(data1)==len(data2))
    info1 = discreteInfo(data1,maxVal=maxVal1)
    info2 = discreteInfo(data2,maxVal=maxVal2)
    return mutualInfo(info1,info2,**kwargs)
    
def discreteJointInfo(data1,data2,data3,maxVal1=None,maxVal2=None,
    maxVal3=None,**kwargs):
    """
    Using data sampled simultaneously from three discrete distributions,
    return an estimate in bits of the mutual information between the
    (single-valued) distribution of the first variable and the
    joint distribution of the last two variables.  That is:
    
    mutualInfo( data1 | data2 , data3 )
    
    data1, data2, and data3 should have the same length.
    
    maxVals are passed to the discreteInfo function, and other kwargs
    are passed to the mutualInfo function.
    """
    assert(len(data1)==len(data2))
    assert(len(data1)==len(data3))
    
    info1 = discreteInfo(data1,maxVal=maxVal1)
    info2 = discreteInfo(data2,maxVal=maxVal2)
    info3 = discreteInfo(data3,maxVal=maxVal3)
    
    return mutualInfo(info1,jointInfo(info2,info3),**kwargs)

def specificInfo(infoContainerY,infoContainerX,stateIndexY):
    """
    As defined in Timme et al. 2014, equation (29).
    
    Uses naive frequencies as probability estimates.
    
    Currently only works with discreteInfo infoContainers.
    """
    pY = infoContainerY.nVec/float(sum(infoContainerY.nVec))
    pStateY = pY[stateIndexY]
    
    # pXgivenStateY is indexed by X states
    infoXgivenStateY = conditionalInfo(infoContainerX,infoContainerY,stateIndexY)
    pXgivenStateY = infoXgivenStateY.nVec/float(sum(infoXgivenStateY.nVec))
    
    # pStateYgivenX is also indexed by X states
    pStateYgivenX = []
    possibleStateIndicesX = range(infoContainerX.maxVal)
    for stateIndexX in possibleStateIndicesX:
        infoYgivenStateX = conditionalInfo(infoContainerY,infoContainerX,stateIndexX)
        pYgivenStateX = infoYgivenStateX.nVec/float(sum(infoYgivenStateX.nVec))
        pStateYgivenStateX = pYgivenStateX[stateIndexY]
        pStateYgivenX.append(pStateYgivenStateX)
    
    with warnings.catch_warnings():
        # ignore "divide by zero" and "invalid value" warnings
        warnings.simplefilter("ignore")
        si = np.sum( np.nan_to_num(
            pXgivenStateY * ( - np.log2(pStateY) + np.log2(pStateYgivenX) ) ) )
    
    return si
   
def redundancy(dataY,dataX1,dataX2):
    """
    Calculate the redundant info. as given in Timme et al. 2014, equation (31).
    
    Uses naive frequencies as probability estimates.
    
    Currently only works with discrete datasets.
    """
    assert(len(dataY)==len(dataX1))
    assert(len(dataY)==len(dataX2))
    
    infoContainerY = discreteInfo(dataY)
    infoContainerX1 = discreteInfo(dataX1)
    infoContainerX2 = discreteInfo(dataX2)
    
    return redundancyContainer(infoContainerY,infoContainerX1,infoContainerX2)

def redundancyContainer(infoContainerY,infoContainerX1,infoContainerX2):
    """
    Calculate the redundant info. as given in Timme et al. 2014, equation (31).
    
    Uses naive frequencies as probability estimates.
    
    Currently only works with discrete infoContainers.
    """
    pY = infoContainerY.nVec/float(sum(infoContainerY.nVec))
    Imin = 0
    possibleStateIndicesY = range(infoContainerY.maxVal)
    for stateIndexY in possibleStateIndicesY:
        specificInfoX1 = specificInfo(infoContainerY,infoContainerX1,stateIndexY)
        specificInfoX2 = specificInfo(infoContainerY,infoContainerX2,stateIndexY)
        Imin += pY[stateIndexY] * min(specificInfoX1,specificInfoX2)
    return Imin

def unique(dataY,dataX1,dataX2):
    """
    Calculate the unique info. given by X1 and X2, as given in
    Timme et al. 2014, equation (33-34).
    
    Uses naive frequencies as probability estimates.
    
    Currently only works with discrete datasets.
    """
    assert(len(dataY)==len(dataX1))
    assert(len(dataY)==len(dataX2))
    
    infoContainerY = discreteInfo(dataY)
    infoContainerX1 = discreteInfo(dataX1)
    infoContainerX2 = discreteInfo(dataX2)
    
    R = redundancyContainer(infoContainerY,infoContainerX1,infoContainerX2)
    U1 = mutualInfo(infoContainerY,infoContainerX1) - R
    U2 = mutualInfo(infoContainerY,infoContainerX2) - R
    return U1,U2

def synergy(dataY,dataX1,dataX2):
    """
    Calculate the synergistic info. as given in
    Timme et al. 2014, equation (32).
    
    Uses naive frequencies as probability estimates.
    
    Currently only works with discrete datasets.
    """
    assert(len(dataY)==len(dataX1))
    assert(len(dataY)==len(dataX2))
    
    infoContainerY = discreteInfo(dataY)
    infoContainerX1 = discreteInfo(dataX1)
    infoContainerX2 = discreteInfo(dataX2)
    
    joint = mutualInfo(infoContainerY,jointInfo(infoContainerX1,infoContainerX2))
    MI1 = mutualInfo(infoContainerY,infoContainerX1)
    MI2 = mutualInfo(infoContainerY,infoContainerX2)
    R = redundancyContainer(infoContainerY,infoContainerX1,infoContainerX2)
    return joint - MI1 - MI2 + R
