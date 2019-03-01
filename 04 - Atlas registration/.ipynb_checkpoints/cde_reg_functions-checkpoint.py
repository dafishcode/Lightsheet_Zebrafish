#===============================================================================
def fishspec(Fdata, prefx = ''):
#===============================================================================
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

#===============================================================================
def meancalc(imgs, Fimg, noimages = 100, delfirst = True, crop = False):
#===============================================================================
    import numpy as np
    import ants
    import os

    print('I found ' + str(len(imgs)) + ' images')

    # Load subsection of tifs
    #---------------------------------------------------------------------------
    maxno   = np.min([len(imgs),noimages])
    loadi   = np.linspace(0,len(imgs)-1,maxno)
    loadi   = loadi.astype(int)
    print('Of these I''m loading ' + str(maxno))
    if delfirst: 
        loadi = np.delete(loadi, 0)
        print('I''m ignoring the first volume')
        
    # Load initial image for dimensions
    #---------------------------------------------------------------------------
    if type(imgs[0]) == str:     
        templ = ants.image_read(Fimg + os.sep + imgs[0])
   
    elif type(imgs[0]) == ants.core.ants_image.ANTsImage:                        
        templ = imgs[0]

    if crop:    
        templ = ants.crop_indices(templ, [0,0,1], templ.shape)
    
    mean_arr    = np.multiply(templ.numpy(), 0);
    imglist     = []
    
    for i in loadi:
        
        if type(imgs[0]) == str:     
            img = ants.image_read(Fimg + os.sep + imgs[i])
        elif type(imgs[0]) == ants.core.ants_image.ANTsImage:                        
            img = imgs[i]  
        if crop:    img = ants.crop_indices(img, [0,0,1], img.shape)

        mean_arr    = mean_arr + img.numpy() / maxno
        imglist.append(img)

    mimg = ants.from_numpy(mean_arr)
    ants.plot(mimg, axis=2, slices = range(8), figsize=3)
    
    return mimg, imglist

#===============================================================================
def fishplot(img, overl = '', orient = 'axial', sliceno = 20, al = .5):
#===============================================================================
    # N.B This breaks frequently - I have no idea what is wrong with the implementation
    
    import ants
    import numpy as np
    r = img.shape
    
    if   orient == 'coronal':     axs = 0; ri = 0
    elif orient == 'sagittal':    axs = 1; ri = 2
    elif orient == 'axial':       axs = 2; ri = 2 
    
    sliceno = min([sliceno,r[ri]])
    sli     = list(map(int, np.ndarray.tolist(np.linspace(0, r[ri]*img.spacing[ri]-1, sliceno))))
    print(r)  
        
    if not overl: ants.plot(img, axis = axs, slices = sli, figsize = 6)
    
    else: ants.plot(img, overl, overlay_cmap = 'magma', overlay_alpha= al, axis = axs, slices = sli, figsize = 6) 
        
