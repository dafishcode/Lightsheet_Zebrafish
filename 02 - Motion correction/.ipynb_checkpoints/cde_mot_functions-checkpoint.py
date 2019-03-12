#===============================================================================
# Find relevant folders and subfolders
#===============================================================================
def cde_mot_fishdef(Fbase, prefix = ''):
    import os
    import re

    if prefix: prefix = prefix + '_'

    dirlist = os.listdir(Fbase)
    r       = re.compile('^' + prefix + 'Z')
    dirlist = list(filter(r.match, dirlist))
    Fish = []

    for d in dirlist:
        if os.path.isdir(Fbase + os.sep + d):
            confolds = os.listdir(Fbase + os.sep + d)

            # Find folders that contain conditions in the folder
            #-----------------------------------------------------------------------
            clist = []
            for c in confolds:
                if os.path.isdir(Fbase + os.sep + d + os.sep + c):
                    tiflist = os.listdir(Fbase + os.sep + d + os.sep + c)
                    r     = re.compile('^[0-9]+_.*[tif|tiff|TIF|TIFF]$')
                    tifs  = list(filter(r.match, tiflist))
                    pth   = Fbase + os.sep + d + os.sep + c
                    clist.append({"tifs":tifs, "name":c, "path":pth})

            Fish.append({"cond" : clist, "base" : Fbase, "id" : d})

    return Fish

#===============================================================================
# Make new folders with specified prefix
#===============================================================================
def cde_mot_makepath(Fish, prefix = '', verbose = False):
    # This function takes a fish specification object and makes directories
    # with a specified prefix. These are saved as 'savepath' in the Fishspec
    # object

    import os

    # Add underscore if string is not empty
    #---------------------------------------------------------------------------
    if prefix: prefix = prefix + '_'

    for f in range(len(Fish)):
        for c in range(len(Fish[f]["cond"])):
            newpath = Fish[f]["base"] + os.sep + prefix + Fish[f]["id"] + os.sep + Fish[f]["cond"][c]["name"]
            if not os.path.isdir(newpath):
                os.makedirs(newpath)
                print('Just created ' + newpath)
            else:
                if verbose: print(newpath + ' already exists')
            Fish[f]["cond"][c]["savepath"] = newpath
    return Fish

#===============================================================================
# Calculate mean image
#===============================================================================
def cde_mot_meancalc(imgs, Fimg, noimages = 100, delfirst = True, crop = False, plot = 'do'):
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
    if plot == 'do':    
        ants.plot(mimg, axis=2, slices = range(mimg.shape[2]), figsize=3)
        
    return mimg, imglist

#===============================================================================
# Perform rigid registration on a list of images
#===============================================================================
def cde_mot_rigidreg(fixed, images, Fimg = '..', spacing = list([1.6,1.6,8]), crop = False, saveprog = False, 
                     savesuff = '', savedir = '.'):
    import ants
    import os
    import datetime
    import time
    
    print('>> Starting rigid registration <<')
    
    if type(fixed) == str: fi = ants.image_read(Fimg + os.sep + fixed)
    else:                  fi = fixed
    if crop:               fi = ants.crop_indices(fi, [0,0,1], fi.shape)
        
    if savesuff:    savesuff = '_' + savesuff
    
    mvd = [];    cnt = 0;    pct1 = len(images) / 100;
    
    for i in images:
        cnt = cnt + 1;   
                
        if type(i) == str: img = ants.image_read(Fimg + os.sep + i)
        else:              img = i
            
        img.set_spacing(spacing)
        if crop:    img = ants.crop_indices(img, [0,0,1], img.shape)
        fi.set_spacing(spacing)
        
        # Actual ants registration step
        #-----------------------------------------------------------------------
        moved  = ants.registration(fi, img, type_of_transform = 'QuickRigid')
        
        if saveprog:
            savename = savedir + os.sep + str(cnt).zfill(4) + savesuff + '.tif'
            ants.image_write(moved["warpedmovout"], savename)
            mvd.append(savename)
            
        else:
            mvd.append(moved["warpedmovout"])
            
        if cnt/pct1 % 5 == 0:   # < this doesn't work robustly 
            ts = time.time()
            st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
            print('Completed ' + str((cnt)/pct1) + '% at ' + st)
        
        
    
    print('All done with rigid registration')
    if saveprog: print('The returned file contains tifs')

    return mvd