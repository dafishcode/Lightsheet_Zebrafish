#=======================================================================
def load(Fcmn): # Load imaging datasets from caiman output to custom object
#=======================================================================    
    import os 
    import re
    from scipy import io
    import numpy as np

    # These things ideally need to be defined in the Fish files
    #-----------------------------------------------------------------------------
    spacing = [1.6, 1.6, 8]
    xy_size = [451, 298]

    # Arrange optional prefix
    #-----------------------------------------------------------------------------
    prefx = ''
    if prefx: prefx = prefx + '_'

    # Find all fish folders
    #-----------------------------------------------------------------------------
    dirlist = os.listdir(Fcmn)
    r       = re.compile(prefx + '[A-Z].*')
    fishies = list(filter(r.match, dirlist))
    print('I found ' + str(len(fishies)) + ' datasets')

    # Go through through folder tree Fish > Condition > Plane.mat 
    #-----------------------------------------------------------------------------
    Fish = []
    for f in fishies:
        print('Loading dataset ' + f)
        condlist   = os.listdir(Fcmn + os.sep + f)
        r          = re.compile('^[A-Z]')
        condlist   = list(filter(r.match, condlist))
        Condition  = []

        for c in condlist:
            files = os.listdir(Fcmn + os.sep + f + os.sep + c)
            r     = re.compile('^[A-Z]')
            files = list(filter(r.match, files))
            files.sort()
            testp = io.loadmat(Fcmn + os.sep + f + os.sep + c + os.sep + files[0])

            pid = 0
            xyz = np.empty((0,3))
            dff = np.empty((0,testp["Temporal"].shape[1]))
            for p in files:

                tp  = io.loadmat(Fcmn + os.sep + f + os.sep + c + os.sep + p)

                # Unpack coordinates
                #---------------------------------------------------------------
                for s in range(tp["Spatial"].shape[1]):
                    xy    = np.reshape(tp["Spatial"][:,s].toarray(), xy_size, order='C')
                    x,y   = np.where(xy)
                    XYZ   = np.array([np.mean(x) * spacing[0], np.mean(y) * spacing[1], pid * spacing[2]])
                    xyz   = np.vstack((xyz, XYZ))

                DFF = tp["Temporal"]
                dff = np.vstack((dff, DFF))
                pid = pid + 1

            Condition.append({"Name":c, "Data":dff, "Coordinates":xyz})

        Fish.append({"Name":f, "Cond":Condition, "Path":Fcmn, "xy_size":xy_size, "spacing":spacing})

    return Fish


#=======================================================================
def nneigh(cs, rng = None, dim = [1,1,1], cnt=5, fullmat = False): # xyz (or xy) coordinates of nodes
#======================================================================= 
    import numpy as np
    
    # Set up nearest neighbour graph
    #---------------------------------------------------------------------------
    mcs  = np.multiply(cs, dim)     # metrically scaled coordinates (in microns)
           
    # Initialise full distance matrix and nearest neighbour graph (binary) matrix
    #---------------------------------------------------------------------------
    nnb    = np.zeros((cs.shape[0],cs.shape[0]))
    dismat = np.zeros((cs.shape[0], cs.shape[0]))
    if rng == None: rng = cs.shape[0] * 2 + 1
    
    # Loop through all cells to fill in distances
    #---------------------------------------------------------------------------
    for r in range(cs.shape[0]):
        dis    = np.ones((1,cs.shape[0]))*10000
        if r % round((10*cs.shape[0]/100)) == 0: print("Doing row " + str(r) + " of " + str(cs.shape[0])) 
            
        for rr in range(max([r-int(rng/2),0]), min([r+int(rng/2),dis.shape[1]])):  # moving window around r
            if fullmat: dismat[r,rr] = np.linalg.norm(mcs[r,:] - mcs[rr,:])
            
            if r == rr: dis[0,rr] = 10000 
            else:       dis[0,rr] = np.linalg.norm(mcs[r,:]-mcs[rr,:])
                
        mini = np.where(dis[0,:] < np.nanpercentile(dis[0,:],cnt))[0]
        nnb[r,mini] = 1
        
    print('Done')
    
    if fullmat: return nnb, dismat
    else:       return nnb

#======================================================================= 
def peaks(dat, cnt = 95, typ = 'std', stdlim = 3):
#======================================================================= 
    import numpy as np
    from scipy import stats
    from scipy import signal
    
    # Find activity peaks
    #---------------------------------------------------------------------------
    pks = np.zeros(dat.shape)
    for i in range(dat.shape[0]):
        d = dat[i,:]                                                            
        if typ == 'peaks':
            p, prop = signal.find_peaks(d,threshold=np.percentile(d,cnt))
        elif typ == 'std':
            sem = np.std(d)
            p   = np.where(d > stdlim*sem)[0]
        else: print('Don''t know what type of binarisation to use')
        
        pks[i,p] = 1
            
    return pks


#======================================================================= 
def avalanche(pks, nnb): 
#======================================================================= 
# pkg - returns peaks grouped into contiguous clusters (coded by numbers)
# avsz - returns the size of all avalanches (in integers)

    import numpy as np
    
    pkg    = np.zeros(pks.shape)                       # peak groups
    act_t  = np.where(np.sum(pks, axis=0) > 3)[0]      # Time points with at least 3 cells active

    i = 0 
    for t in act_t:
        
        tlen = act_t.shape[0]
        if i % round(10*tlen/100) == 0: print("Doing time point " + str(i) + " of " + str(tlen)) 
        i   = i + 1
        
        gr  = 1
        cid = np.where(pks[:,t] > 0)[0]
        
        for c in cid:
            
            # If currently unlabelled, label with gr + 1
            #-------------------------------------------------------------------
            if pkg[c,t] == 0:        
                gr = gr + 1
                pkg[c,t] = gr

            # Find all neighbours
            #-------------------------------------------------------------------
            nb   = np.where(nnb[c,:] > 0)[0]
            tgr  = np.intersect1d(cid,nb)


            # Label those that are active the same as 
            #-------------------------------------------------------------------
            pkg[tgr,t] = pkg[c,t]
            
    # For each time point count number of coactivations
    #----------------------------------------------------------------------------
    print('Now calculating avalanche size')
    avsz = np.array([])
    for t in range(pkg.shape[1]):
        comms = np.unique(pkg[:,t][pkg[:,t] > 0])
        for c in comms:
            avsz = np.append(avsz, pkg[:,t][pkg[:,t] == c].shape[0])    

    return pkg, avsz


#=======================================================================
def spacek(cs, mcc = 10):    # K-means clustering on cell coordinates 
#======================================================================= 
# This function takes the x-y dimensions from identified cells in a numpy array 
# It then performs clustering to pull out spatially contiguous groups of cells
# (This is borrowed from Rick Betzel's zebrafish paper: 
#  https://www.biorxiv.org/content/early/2018/12/15/496414 )    
# 
# mcc = mean cells per cluster
# returns updated Fish object

    import numpy as np
    from sklearn.cluster import KMeans
    
    n_clust = int(cs.shape[0] / mcc)    
    kmeans  = KMeans(n_clusters = n_clust, random_state = 0).fit(cs)

    return kmeans.labels_

#===============================================================================
def revreg(fish, zbb, F): 
#===============================================================================
    # This function takes the (single) fish (and single condition) defined in
    # 'fish' and applies the reverse registration between that fish and the 
    # atlas image supplied
    
    import ants 
    import os
    import numpy as np
    
    Ftemps = F["Ftemps"]
    Ftrans = F["Ftrans"]
    Freg   = F["Freg"]
    
    c = 0  # << This still needs editing
    
    # Load raw image and apply initial transform from registration
    #------------------------------------------------------------------------------
    rw = ants.image_read(Freg +os.sep+ fish["Name"] +os.sep+ fish["Cond"][c]["Name"] +os.sep+
                        'Raw' +os.sep+ fish["Cond"][c]["Name"] + '.tif')
    rw.set_direction(np.array([[-1.,0.,0.],[0.,1.,0.],[0.,0.,-1.]]))
    rw.set_spacing([.3,.3,6.])
    
    # Locate relevant transforms
    #==============================================================================
    # Confocal image to reference (ZBB) image
    #------------------------------------------------------------------------------
    aff1 = Ftrans + os.sep + 'ref2cf_R.mat'
    aff2 = Ftrans + os.sep + 'ref2cf_S.mat'
    syn  = Ftrans + os.sep + 'ref2cf_S.nii.gz'
    
    # Functional image to confocal image
    #------------------------------------------------------------------------------
    c = 0
    ftrans = Freg + os.sep + fish["Name"] + os.sep + fish["Cond"][c]["Name"] + os.sep + 'FUN2CF'
    AFF1   = ftrans + os.sep + 'cf2fun_R.mat'
    AFF2   = ftrans + os.sep + 'cf2fun_S.mat'
    SYN    = ftrans + os.sep + 'cf2fun_S.nii.gz'
    
    # Load templates and intermediates
    #-----------------------------------------------------------------------------
    cfc   = ants.image_read(Ftemps +os.sep+ 'Confoc.tif')
    cfc.set_spacing([.3,.3,1.])        # Same as used in registration 
    cfc.set_direction(np.array([[-1., 0., 0.],[0., 1., 0.],[0., 0., -1]]))
    
    if zbb.shape[0] == 515: zbb.set_spacing([.6,.6,2.])
    else: zbb.set_spacing([.3,.3,1.]) 

    zbb_t = ants.apply_transforms(cfc, zbb, [aff1, aff2, syn], whichtoinvert = [True, True, False])
    zbb_t = ants.apply_transforms(rw, zbb_t, [AFF1, AFF2, SYN], whichtoinvert = [True, True, False])
    
    return(zbb_t)
    
    
#===============================================================================
def fishplot(img, overl = '', orient = 'axial', sliceno = 20, al = .5, col = 'magma'):
#===============================================================================
    # N.B This breaks frequently - I have no idea what is wrong with the implementation
    
    import ants
    import numpy as np
    r = img.shape
    
    if   orient == 'coronal':     axs = 0; ri = 0
    elif orient == 'sagittal':    axs = 1; ri = 1
    elif orient == 'axial':       axs = 2; ri = 2 
    
    # ALERT ALERT I HAVE NO IDEA WHAT THE SLICE INDEXING WANTS FROM ME
    # Does it live in the physical space?
    # Does it live in the voxel space?
    # I DON'T KNOW - so I'm fudging it
    
    sliceno = min([sliceno,r[ri]])
    
    if orient == 'axial': mx_dim = r[ri] - 1
    else: mx_dim = r[ri] * img.spacing[ri]-1
    sli     = list(map(int, np.ndarray.tolist(np.linspace(0, mx_dim, sliceno))))
        
    if not overl: 
        ants.plot(img, axis = axs, slices = sli, figsize = 6)
    else:         
        ants.plot(img, overl, overlay_cmap = col, overlay_alpha= al, axis = axs, slices = sli, figsize = 6) 
       
    return None

#===============================================================================
def modelfunctions(mtype = 'decay'):
#===============================================================================
    # This function is used for modelling single cell spatial correlation patterns. 
    # It supplies type of functions that require different sets of parameters for use
    # in optimisation procedures
    
    import numpy as np
    
    if mtype == 'decay':
        def fun(x, width):   
            height = 1
            lam = 1/width
            y   = height * np.exp(-lam * x)
        
            return y
    
    if mtype == 'mixed':
        def fun(x, h1, w1, h2, w2):
            lam = 1/w1
            y1  = h1 * np.exp(-lam * x)

            mu  = 0
            sig = np.sqrt(2*np.log(2)) / w2
            y2  = h2 * (np.exp( - np.square(x - mu) / 2 * np.square(sig)) / (sig * np.sqrt(2*np.pi)) / 1/(sig * np.sqrt(2*np.pi)) )
            
            return y1 + y2

    if mtype == 'norm':
        def fun(x, h, w):
            mu   = 0
            sig  = np.sqrt(2 * np.log(2)) / w
            y    = h * np.exp( - np.square(x - mu) / 2 * np.square(sig)) / \
                   (sig * np.sqrt(2*np.pi)) / 1/(sig * np.sqrt(2 * np.pi)) 
        
            return y

    return fun

#===============================================================================
def univars(mtype, d):
#===============================================================================
    # This function applies a certain calculation on data array(s) stored in the
    # list d. The function types can be:
    # 'p_firing' - expects the peak array as input
    
    from cde_net_functions import modelfunctions as mf    
    from scipy import optimize
    import numpy as np
    
    if mtype == 'p_firing': return (np.sum(d[0], axis = 1) / d[0].shape[1]) 
    if mtype == 'spatial':
        # requires d to be full data (for correlation estimation) in d[0]
        # requires dis to contain full distance matrix in d[1]
        cor    = np.corrcoef(d[0])
        cwidth = np.ndarray([])
        for cll in range(d[0].shape[0]):
            dvals       = d[1][cll,:].flatten()
            cvals       = np.nan_to_num(cor[cll,:].flatten(), 0)
            cvals[cvals < 0] = 0

            try: 
                popt, pcov = optimize.curve_fit(mf('decay'), dvals, cvals)
                cwidth     = np.append(cwidth, popt[0])
            except: 
                print('Did not optimise cell ' + str(cll))
                cwidth     = np.append(cwidth, np.nan)
            
        
        return cwidth
        

#===============================================================================
def winslide(fish, win = 60 * 4, stp = 1, mtype = 'p_firing'):
#===============================================================================
    # This function estimates different univariate measures using a sliding 
    # window approach
    
    import numpy as np
    import cde_net_functions as cn
    dat = fish["Data"]
    pks = fish["Peaks"]
    dis = fish["Distances"]
    
    starts = np.arange(0, dat.shape[1] - win - 1, win*stp)
    
    # N.B. The numpy stacking is the most confusing thing in the universe,
    # I have no idea why and when I am transposing anything here, so ERRORs
    # are likely 
    
    starts = np.arange(0, dat.shape[1] - win - 1, win*stp)
    wd = np.array([])
    for s in range(len(starts)-1):
        print('Working on time step ' + str(s+1) + ' of ' + str(len(starts)))
        d = dat[:,starts[s]:starts[s]+win]
        p = pks[:,starts[s]:starts[s]+win]
        if mtype == 'p_firing':  td = np.transpose(cn.univars(mtype, [p]))
        elif mtype == 'spatial': td = np.transpose(cn.univars(mtype, [d, dis]))
        wd = np.vstack((wd, td)) if wd.size else td
        
    wd = np.transpose(wd)
    
    return wd

#===============================================================================
def atlasmap(Fish, F):
#===============================================================================
    import os 
    import ants
    import cde_net_functions as cn
    
    def atlasgen(frompath, topath, Fish, F):
        atl = ants.image_read(frompath)
        return cn.revreg(Fish, atl, F)
    
    for c in range(len(Fish["Cond"])):
        print('Checking / Making atlases for ' + Fish["Cond"][c]["Name"])
        
        Flut = F["Freg"] +os.sep+ Fish["Name"] + os.sep + Fish["Cond"][c]["Name"] +os.sep+ 'LUTs'
        if not os.path.exists(Flut): os.makedirs(Flut)

        # Zbrain atlas
        #---------------------------------------------------------------------------
        to = Flut +os.sep+ 'Z-brain_Atlas.tif'
        fr = F["Ftemps"] +os.sep+ 'z-brain.tif'
        if not os.path.isfile(to): ants.image_write(atlasgen(fr, to, Fish, F), to)

        # Pajevic atlas
        #--------------------------------------------------------------------------
        to = Flut +os.sep+ 'Pajevic.tif'
        fr = F["Ftemps"] +os.sep+ 'pajevic1.tif'
        if not os.path.isfile(to): ants.image_write(atlasgen(fr, to, Fish, F), to)

#===============================================================================
def atlasload(Fish, F):
#===============================================================================
    import os
    import ants
    import numpy as np
    import pandas as pd
    
    # Original (voxel count) coordinates - for look up tables
    #------------------------------------------------------------------------------
    for c in range(len(Fish["Cond"])):
        rw = ants.image_read(F["Freg"] +os.sep+ Fish["Name"] +os.sep+ Fish["Cond"][c]["Name"] +os.sep+
                            'Raw' +os.sep+ Fish["Cond"][c]["Name"] + '.tif')

        # Set up dummy Ants image to check orientation
        #------------------------------------------------------------------------------
        sp  = Fish["spacing"]                       # spacing used for the current fish
        dm  = Fish["xy_size"]                       # dimensions of current fish's xy plane (in pixels? Don't know...)
        cs  = Fish["Cond"][c]["Coordinates"]        # coordinates of the point cloud derived from current fish
        ocs = np.divide(cs, sp)                          # Derive original (voxel count) coordinates

        Fish["Cond"][c]["OCoord"] = ocs
    
    # Find voxel labels from look up tables previously produced
    #------------------------------------------------------------------------------
    for c in range(len(Fish["Cond"])):
        Flut = F["Freg"] +os.sep + Fish["Name"] + os.sep + Fish["Cond"][c]["Name"] +os.sep+'LUTs'
        zbat = ants.image_read(Flut +os.sep+ 'Z-brain_Atlas.tif')
        pajt = ants.image_read(Flut +os.sep+ 'Pajevic.tif')
        ocs  = Fish["Cond"][c]["OCoord"]

        bm = np.array((0))
        za = np.array((0))
        pj = np.array((0))

        for o in ocs:
            pj = np.append(pj, pajt[int(o[0]), int(o[1]), int(o[2])])
            za = np.append(za, zbat[int(o[0]), int(o[1]), int(o[2])])
            bm = np.append(bm, pj[-1] > 0)

        Fish["Cond"][c]["Features"] = pd.DataFrame({'Brain Mask':bm[1:], 'Z-brain':za[1:], 'Coexpression':pj[1:]})
        
    return Fish

#===============================================================================
def fishdot(fish, cols, ax = None, cmap = 'Spectral_r', al = 1):
#===============================================================================
    # This function takes a fish (single fish, single condition) and a single 
    # vector of the same length as numbers of cells and returns a mapping of 
    # that vector onto the cells as a plot
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    if ax == None: ax = plt.gca()
        
    bmi    = fish["Features"]["Brain Mask"] == 1
    if cols.shape[0] == bmi.shape[0]: bmi = bmi
    else:                             
        print('There may be brain mask inconsistencies')
        bmi = np.ones(cols.shape)
    ocs    = fish["OCoord"]
    cs     = fish["Coordinates"]

    outplot = ax.scatter(ocs[bmi,0], ocs[bmi,1], 600, c = cols[bmi], 
                         cmap = cmap, alpha = al)
        
    return outplot

#===============================================================================
def fishplay(Fish, colmat):
#===============================================================================
    import numpy as np
    import matplotlib.pyplot as plt

    from matplotlib import animation, rc
    from IPython.display import HTML
    
    # First set up the figure, the axis, and the plot element we want to animate
    fig, ax = plt.subplots()

    ax.set_xlim(( 0, 2))
    ax.set_ylim((-2, 2))

    line, = ax.plot([], [], lw=2)
    
    # initialization function: plot the background of each frame
    def init():
        line.set_data([], [])
        return (line,)
    
    # animation function. This is called sequentially
    def animate(i):
        x = np.linspace(0, 2, 1000)
        y = np.sin(2 * np.pi * (x - 0.01 * i))
        line.set_data(x, y)
        return (line,)
    
    # call the animator. blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=100, interval=20, blit=True)
    
    HTML(anim.to_html5_video())