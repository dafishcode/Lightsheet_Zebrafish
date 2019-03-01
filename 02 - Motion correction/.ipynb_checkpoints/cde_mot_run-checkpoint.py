import os
import cde_mot_functions as cf
import ants
from importlib import reload

# Get fish specifications
#-------------------------------------------------------------------------------
Fbase    = '/Volumes/MARIANNE/1812 Critical Dynamics in Epilepsy'
Fish     = cf.cde_mot_fishdef(Fbase)

# produce mean image and run registration to the mean (rigid)
#-------------------------------------------------------------------------------

Fish = cf.cde_mot_makepath(Fish, prefix = 'MC')

f = 9;
c = 0;
tifs = Fish[f]["cond"][c]["tifs"]
Fimg = Fish[f]["cond"][c]["path"]
Fsave = Fish[f]["cond"][c]["savepath"]

# Calculate mean image - iterative rigid registration to find good mean image
#-------------------------------------------------------------------------------
mimg, imglist     = cf.cde_mot_meancalc(tifs, Fimg, noimages = 3)
moved             = cf.cde_mot_rigidreg(imglist[1], imglist, Fimg)
mimg2, imglist2   = cf.cde_mot_meancalc(moved, Fimg)


# run registration to mean

# cf.cde_mot_rigidreg(Fish[f]["cond"][c]["tifs"], Fish[f]["cond"][c]["path"])
