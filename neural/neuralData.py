# neuralData.py
#
# Bryan Daniels
# 2021/06/17 branched from loadRoozbehData.py
#
# Loads and manipulates neural spike time data.
#

import numpy as np
import pandas as pd
import scipy.io

def loadBinnedSpikingData(matFilename,alignName='go_cue',timeWindow=100,
                          binRange=[-10,10],relativeMidTimes=None):
    """
    matFilename             : Name of MATLAB file to read in
    alignName ('go_cue')    : Name of time data with which to align trials.
                              Choose from 'go_cue', 'dots_onset', 'dots_offset',
                              or 'sacc_onset'.
    timeWindow (100)        : length (in ms) of time window
    binRange ([-10,10])     : Indices of equally-spaced start and
                              stop bins, each of width timeWindow.
                              Bin 0 is centered on the given zero time.
    relativeMidTimes (None) : List of times (in ms) that overrides
                              the binRange option.  Zero
                              corresponds to the align time.  Bins
                              are of width timeWindow centered
                              on each midTime.
    """
    matlabdata = scipy.io.loadmat(matFilename)
    
    spikes = matlabdata['spike_times']
    spikes_flat = spikeTimesArray(spikes)
    
    alignTimes = np.ndarray.flatten(matlabdata['event_times'][alignName][0][0])
    
    return binnedSpikingData(spikes_flat,timeWindow,alignTimes,
                             binRange=binRange,relativeMidTimes=relativeMidTimes)

def loadBehaviorData(matFilename,name='chosen_target'):
    matlabdata = scipy.io.loadmat(matFilename)
    npdata = np.ndarray.flatten(matlabdata['trial_info'][name][0,0])
    numTrials = len(npdata)
    return pd.Series(npdata,index=trialNames(numTrials))

def trialNames(numTrials):
    return ['Trial {}'.format(i) for i in range(numTrials)]

def spikeTimesArray(spikeTimeLists):
    """
    Convert spike time lists as given in MATLAB files to a non-ragged
    numpy array of shape (#trials)x(#neural units)x(max # spikes).
    
    Each (trial)x(neural unit) combination lists all spike times and
    is otherwise padded with nans.
    """
    max_len = np.max([[len(trialNeuronSpikes) for trialNeuronSpikes in trialSpikes ] \
        for trialSpikes in spikeTimeLists ])
    spikes_flat = np.nan*np.ones(
                    [len(spikeTimeLists),len(spikeTimeLists[0]),max_len],
                    dtype=float)
    for i,trialSpikes in enumerate(spikeTimeLists):
        for j,trialNeuronSpikes in enumerate(trialSpikes):
            spikes_flat[i,j,:len(trialNeuronSpikes)] = \
                np.ndarray.flatten(trialNeuronSpikes)
    return spikes_flat

def binnedSpikingData(allSpikeTimesArray,timeWindow,alignTimes,
                      binRange=[-10,10],relativeMidTimes=None):
    """
    Returns integer data representing how many times
    each neuron spiked within each window of time.
    
    timeWindow              : length (in ms) of time window
                              (SchBerSeg06 uses 20 ms)
                              (I've been using 100 ms or 250 ms or 200 ms)
    alignTimes              : list of times, one for each trial,
                              indicating the zero time for that trial.
                              (e.g. behavioralDataDict['GoCueTime'])
    binRange ([-10,10])     : Indices of equally-spaced start and
                              stop bins, each of width timeWindow.
                              Bin 0 is centered on the given zero time.
    relativeMidTimes (None) : List of times (in ms) that overrides
                              the binRange option.  Zero
                              corresponds to the align time.  Bins
                              are of width timeWindow centered
                              on each midTime.
    """
    
    if relativeMidTimes is None:
        binNumbers = range(binRange[0],binRange[1]+1)
        binRelativeTimes = np.array(binNumbers)*timeWindow
        relativeMidTimes = binRelativeTimes
    binnedData = []

    for trialSpikeTimes,alignTime in                        \
        zip(allSpikeTimesArray,alignTimes):
            binnedTrialData = []
            #timeBins = binRelativeTimes + alignTime
            midTimes = relativeMidTimes + alignTime
            startTimes = midTimes - timeWindow/2.
            
            # Messing around with arrays to make faster.
            # Could be even better if you care to try...
            for s in startTimes:
                h = np.sum(np.logical_and(np.less(s,trialSpikeTimes),np.less(trialSpikeTimes,s+timeWindow)),axis=1)
                binnedTrialData.append(h)
            binnedData.append(np.transpose(binnedTrialData))
    
    binnedData = np.asarray(binnedData)
    
    # make pandas dataframe
    numTrials,numNeurons,numTimes = binnedData.shape
    idx = pd.MultiIndex.from_product([np.arange(numNeurons),relativeMidTimes],
        names=['neural unit','time (ms)'])
    df = pd.DataFrame(binnedData.reshape(numTrials,numNeurons*numTimes).T,
                      index=idx,
                      columns=trialNames(numTrials))
    
    return df

