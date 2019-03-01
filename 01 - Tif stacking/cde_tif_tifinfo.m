function F = cde_tif_tifinfo(Fpath)
% Housekeeping
%--------------------------------------------------------------------------
fs      = filesep;

% Index all single TIF slices (uses parallel processing if possible)
%==========================================================================
files   = dir(Fpath);    
id      = ~cellfun(@isempty, strfind({files.name}, 'tif'));
files   = files(id);

parfor f = 1:length(files) 
    info{f} = imfinfo([Fpath fs files(f).name]); 
end

% Sort all files into single structure
%==========================================================================
close all
allcount = 0;

for i = 1:length(info)   
for l = 1:length(info{i})
    allcount = allcount + 1;
    F(allcount).fname   = info{i}(l).Filename;
    F(allcount).ind     = l;
end
end
