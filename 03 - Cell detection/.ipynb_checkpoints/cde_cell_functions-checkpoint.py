def cde_cell_fishspec(Fdata, prefx = ''):
    import os
    import re
    
    if prefx: prefx = '^' + prefx + '.*' 
    names = os.listdir(Fdata)
    r     = re.compile(prefx + 'ZFRR.*')
    folds = list(filter(r.match, names))
 
    Zfish = []
    for f in folds:
        cfld = next(os.walk(Fdata + os.sep + f))[1]
        Cond = []
        for c in cfld:
            Tpaths = []
            tifs = os.listdir(Fdata + os.sep + f + os.sep + c)
            r    = re.compile('^.*[tif|tiff|TIF|TIFF]$')
            tifs = list(filter(r.match, tifs))
            Tpaths = []
            for t in tifs:
                Tpaths.append(Fdata + os.sep + f + os.sep + c + os.sep + t)
                
            Cond.append({'Name':c, 
                         'Path':Fdata + os.sep + f + os.sep + c, 
                         'Tifs':tifs,
                         'Tpaths':Tpaths})
            
        Zfish.append({'Cond':Cond, 'Name':f[len(prefx)-2:]})
    
    return Zfish


def cde_cell_planesave(Fdata, Fish, cname = 'all', mxpf = 7500):

    from skimage import io
    import os
    import numpy as np

    # Make new folder for plane-divided data
    #--------------------------------------------------------------
    Fplanes = Fdata + os.sep + 'PL_' + Fish["Name"]
    if not os.path.isdir(Fplanes): os.mkdir(Fplanes)

    if cname.lower() == 'all':
        crange = list(range(0,len(Fish["Cond"])))
    else:
        crange = []
        for ci in range(0,len(Fish["Cond"])): 
            if Fish["Cond"][ci]["Name"].lower() == cname.lower():
                crange.append(ci) 

    print('I found ' + str(len(crange)) + ' Conditions')

    # Loop through conditions
    #--------------------------------------------------------------
    for c in crange:

            Fcond = Fplanes + os.sep + Fish["Cond"][c]["Name"]
            if not os.path.isdir(Fcond): os.mkdir(Fcond)

            # Get shape information on the zplanes
            #----------------------------------------------------------
            ttif = io.imread(Fish["Cond"][c]["Tpaths"][0])
            npln = ttif.shape[0]

            print('Condition ' + Fish["Cond"][c]["Name"])
            print('> There are ' + str(npln) + ' Planes')
            tifs = Fish["Cond"][c]["Tpaths"]

            # Loop through tifs and collate in single numpy array
            #---------------------------------------------------------
            for pl in range(0,npln):
                print('> Processing plane ' + str(pl+1))

                # Load individual tifs in batches
                #----------------------------------------------------
                btch = list(range(0,len(tifs), mxpf))
                if len(tifs) > btch[-1]: btch.append(len(tifs))            
                for bi in range(len(btch)-1):
                    pln = io.imread(tifs[0])[pl,:,:]
                    pln = pln.reshape(1, pln.shape[0], pln.shape[1])
                    for i in range(btch[bi],btch[bi+1]):
                        if i%1000 == 0: print('> > ' + str(i)) 
                        ldd = io.imread(tifs[i])[pl,:,:]
                        ldd = ldd.reshape(1, ldd.shape[0], ldd.shape[1])
                        pln = np.concatenate((pln, ldd), axis = 0)

                    # Save new tif
                    #---------------------------------------------------
                    ps = str(pl)
                    bs = str(bi)
                    pth  = Fcond + os.sep
                    fnm  = (Fish["Name"] + '_s' + bs.zfill(2) 
                            + '_' + Fish["Cond"][c]["Name"][0] 
                            + '_PL' + ps.zfill(2) + '.tif')
                    print(pth + fnm)
                        
                    io.imsave(pth + fnm, pln)
            