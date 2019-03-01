function [tomat, frommat] = cde_tif_tifmats(fish, c, nplanes)
% This function takes fish specifications (single fish), the condition
% number that is of interest, and the number of planes that constitute a
% single volume. It will return two matrices: 
%
% 'tomat' -- is a P x T matrix, where P is the number of planes, and T is  
% the number of time windows. Each entry in this matrix is the index of a 
% single acquired plane. 
%
% 'frommat' -- is a N x F matrix, where N is the number of planes in the
% largest tif file the data is stored in, and F is the number of original
% tif files. Each entry is the same index as in the matrix above for cross
% referencing


% Make a matrix of tif indices in planes x time windows/frames
    %--------------------------------------------------------------------------
    ntifs   = sum(fish.cond(c).tifn);
    rntifs  = ntifs - mod(ntifs, nplanes);
    tomat  = reshape(1:rntifs, nplanes, fix(ntifs/nplanes));

    if numel(tomat) < ntifs,   warning(['Ignoring ' num2str(mod(ntifs, nplanes)) ' tifs']);  end

    % Make a matrix of tif indices in planes x index of original tif 
    %--------------------------------------------------------------------------
    total = 0;
    tifn  = fish.cond(c).tifn;
    for d = 1:length(fish.cond(c).tifn)
        frommat(1:tifn(d),d) = (1:tifn(d)) + total;
        total = total + tifn(d);
    end
