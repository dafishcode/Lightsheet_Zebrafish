function D = cde_tif_housekeeping

fs      = filesep;
[p f]   = fileparts(which('cde_tif_housekeeping'));
addpath(genpath([p fs '..']))


% Define 'Template Fish' object that contains experimental incl in folder name
%--------------------------------------------------------------------------
clear Fish
Fish.age    = 6;
Fish.type   = 'WT';
Fish.exp    = 'PTZ';
Fish.base   = '/Volumes/BRANDT/1807 CDE WT Data';
Fish        = cde_tif_fishfinder(Fish);

% Housekeeping
% ==========================================================================
fs          = filesep;
D.Fsave     = '/Volumes/MARIANNE/1812 Critical Dynamics in Epilepsy';
D.Fscript   = '/Users/roschkoenig/Dropbox/Research/1812 Critical Dynamics Epilepsy';
D.Fish      = Fish;