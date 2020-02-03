def subspec(F):
    import re, os 
    
    # Specify SEEG folders and file locations
    #-------------------------------------------------------------------------------
    names = os.listdir(F['seeg'])
    r     = re.compile('^[A-Z].*[a-z]$')
    folds = list(filter(r.match, names))
    folds.sort()

    Sub = []
    for f in folds:
        edfs = os.listdir(F['seeg'] +os.sep+ f)
        r    = re.compile('^[A-Z].*Baseline.*[EDF|edf]') 
        bl   = list(filter(r.match, edfs))
        for k in range(len(bl)): bl[k] =  F['seeg'] +os.sep+ f +os.sep+ bl[k] 
        bl.sort()

        r    = re.compile('^[A-Z].*Seizure.*[EDF|edf]') 
        sz = list(filter(r.match, edfs))
        for k in range(len(sz)): sz[k] =  F['seeg'] +os.sep+ f +os.sep+ sz[k] 
        sz.sort()

        Sub.append({'Base': bl, 'Seiz': sz})
    
    return Sub

def load_bin_sub(sub, PP, BN):
    if len(sub['Base']) != len(sub['Seiz']): raise ValueError('Mismatch in the number of Baseline and Seizure segments')
    # Find out how many channels there are for this subject 
    #-------------------------------------------------------------------------------
    bl     = mne.io.read_raw_edf(sub['Base'][0])
    Nch    = len(bl.ch_names)
    Bl_bin = np.ndarray((Nch,0)) 
    bl_id  = np.ndarray(0)
        

def binarise(sub, PP, BN, which='both'):
    import mne
    import numpy as np
    import os, re, sys
    import scipy 
    
    if len(sub['Base']) != len(sub['Seiz']): raise ValueError('Mismatch in the number of Baseline and Seizure segments')

    # Find out how many channels there are for this subject 
    #-------------------------------------------------------------------------------
    bl     = mne.io.read_raw_edf(sub['Base'][0])
    Nch    = len(bl.ch_names)
    Bl_bin = np.ndarray((Nch,0)) 
    bl_id  = np.ndarray(0)
    
    if which == 'both':
        Sz_bin = np.ndarray((Nch,0))
        sz_id  = np.ndarray(0)

    for k in range(len(sub['Base'])):

        # Load data, filter and rereference
        #---------------------------------------------------------------------------
        bl = mne.io.read_raw_edf(sub['Base'][k], preload=True)
        bl.filter(PP['Fbp'][0], PP['Fbp'][1])
        bl.set_eeg_reference(ref_channels='average')
        
        if which == 'both':
            sz = mne.io.read_raw_edf(sub['Seiz'][k], preload=True)
            sz.filter(PP['Fbp'][0], PP['Fbp'][1])
            sz.set_eeg_reference(ref_channels='average')

        # Binarise matrices
        #---------------------------------------------------------------------------
        chanmean = bl._data.mean(1)
        chanstdv = bl._data.std(1)

        zbl = np.divide(np.subtract(bl._data, chanmean[:,np.newaxis]), chanstdv[:,np.newaxis])
        tbl = np.zeros(zbl.shape)
        
        if which == 'both':
            zsz = np.divide(np.subtract(sz._data, chanmean[:,np.newaxis]), chanstdv[:,np.newaxis]) 
            tsz = np.zeros(zsz.shape)
            
        for chan in range(zbl.shape[0]):
            pbl = scipy.signal.find_peaks(abs(zbl[chan,:]).astype('float64'), 
                                          height=BN['peak_height'], width=BN['separation_win'])
            for p in pbl[0]: tbl[chan,p] = 1
            
            if which == 'both':
                psz = scipy.signal.find_peaks(abs(zsz[chan,:]).astype('float64'), 
                                              height=BN['peak_height'], width=BN['separation_win'])
                for p in psz[0]: tsz[chan,p] = 1

        # Calculate ranges to avoid edges of the segment 
        #---------------------------------------------------------------------------
        blrange = (1000, min(tbl.shape[1]-1000, 31000))
        if which == 'both': szrange = (1000, min(tsz.shape[1]-1000, 31000))

        # Segment out the ranges above and keep track of segment boundaries
        #---------------------------------------------------------------------------
        Bl_bin = np.append(Bl_bin, tbl[:,blrange[0]:blrange[1]], axis=1)
        bl_id  = np.append(bl_id, np.ones((blrange[1]-blrange[0]))*k, axis=0)
        if which == 'both':
            Sz_bin = np.append(Sz_bin, tsz[:,szrange[0]:szrange[1]], axis=1)
            sz_id  = np.append(sz_id, np.ones((szrange[1]-szrange[0]))*k, axis=0)

    sub.update({'bin_base':Bl_bin, 'bl_seg_id':bl_id})
    if which == 'both': sub.update({'bin_seiz':Sz_bin, 'sz_seg_id':sz_id})
    return sub


def avcount(bintrace, AS):
    import numpy as np
    
    # Calculate sums of events within each firing window 
    #-------------------------------------------------------------------------------
    tsums = np.sum(bintrace,0)  
    sums  = np.ndarray(0)
    for s in range(0,len(tsums),AS['dt']):
        sums = np.append(sums, np.sum(tsums[s: (s+AS['dt']-1)]))

    # Concatenate ongoing runs of events
    #-------------------------------------------------------------------------------
    thisid   = 0
    lastseen = -1
    allids   = np.zeros(sums.shape)

    for s in range(len(sums)):
        if sums[s] > 0:
            allids[s] = thisid
            lastseen  = s
        else: 
            if lastseen == s-1: thisid = thisid+1
            allids[s] = 0
    #     print(allids[s])

    # Count number of total events within each cascade
    #-------------------------------------------------------------------------------
    avcounts = np.ndarray(0)
    avids    = np.unique(allids)
    avids    = avids[np.where(avids != 0)]
    for a in avids:
        avcounts = np.append(avcounts, np.sum(sums[np.where(allids==a)[0]]))
    
    return avcounts

def plot_ccdf(avc, No_bins=50, ax=None, color=1):
    import numpy as np
    import matplotlib.pyplot as plt 
    import matplotlib
    
    cmap       = matplotlib.cm.get_cmap('Set1')
    tc         = cmap(color)
    
    # Calcualte logarithmically binned c-cdf 
    #-------------------------------------------------------------------------------
    bins       = np.linspace(np.log(np.min(avc)),np.log(np.max(avc)), No_bins)
    avc_binned = np.histogram(avc, bins=np.exp(bins))
    avc_cdf    = len(avc) - np.cumsum(avc_binned[0])
    
    # Calcualte logarithmically binned c-cdf 
    #-------------------------------------------------------------------------------
    if not ax:
        plt.scatter(np.log(avc_binned[1][1:]), np.log(avc_cdf), color=tc)
        ax = plt.gca()
    else:
        ax.scatter(np.log(avc_binned[1][1:]), np.log(avc_cdf), color=tc)