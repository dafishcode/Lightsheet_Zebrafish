def run(Fbase):
    print('yea')
    import os
    import cde_mot_functions as cf
    import ants

    Fish     = cf.cde_mot_fishdef(Fbase)
    print('yeah')
    # Produce mean image and run registration to the mean (rigid)
    #-------------------------------------------------------------------------------
    Fish  = cf.cde_mot_makepath(Fish, prefix = 'RM')

    for f in range(len(Fish)):
        for c in range(len(Fish[f]["cond"])):

            print('=========================================================')
            print('Working on fish ' + str(f+1) + ' of ' + str(len(Fish)) + ', condition: ' + Fish[f]["cond"][c]["name"] )
            print('=========================================================')
            tifs  = Fish[f]["cond"][c]["tifs"]
            Fimg  = Fish[f]["cond"][c]["path"]
            Fsave = Fish[f]["cond"][c]["savepath"]

            # Calculate mean image - iterative rigid registration to find good mean image
            #-------------------------------------------------------------------------------
            mimg, imglist     = cf.cde_mot_meancalc(tifs, Fimg, noimages = 50, plot = 'none')   # mimg is provisional mean image
            moved             = cf.cde_mot_rigidreg(mimg, imglist, Fimg)        # moved is provisionally registred images
            mimg2, imglist2   = cf.cde_mot_meancalc(moved, Fimg, crop = True, plot = 'none')  # mimg2 is the post-reg mean image

            # run registration to mean
            #-------------------------------------------------------------------------------
            moved             = cf.cde_mot_rigidreg(mimg2, tifs, Fimg, saveprog = True,
                                                    savesuff = Fish[f]["cond"][c]["name"][0], savedir = Fsave)