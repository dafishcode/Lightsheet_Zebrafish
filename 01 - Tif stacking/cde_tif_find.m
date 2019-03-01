function fish = cde_tif_find(fish)
fs       = filesep;
allfolds = dir(fish.path);
allfolds = allfolds(3:end);

dirid = find([allfolds.isdir]);
if ~isempty(dirid)
    disp(['Found ' num2str(length(dirid)) ' folders'])
    
    for d = 1:length(dirid)
        
        % Find raw tif files (many planes cobbled together)
        %------------------------------------------------------------------
        fpath       = [fish.path fs allfolds(dirid(d)).name];
        allfiles    = dir(fpath);
        tifids      = find(~cellfun(@isempty, regexp({allfiles.name},  '^[a-zA-Z0-9].*\.tif$')));
        tifnames    = {allfiles(tifids).name};
        
        % Extract more tif information from this list
        %------------------------------------------------------------------
        running_tot = 0;
        for t = 1:length(tifnames)
            disp(['Processing tif ' num2str(t) ' of ' num2str(length(tifnames)) ' in current batch'])            
            inf = imfinfo([fpath fs tifnames{t}]);
            fish.cond(d).name           = allfolds(dirid(d)).name;
            fish.cond(d).tif{t}         = [fpath fs tifnames{t}];
            fish.cond(d).tifn(t)        = length(inf);
            fish.cond(d).tifinf(t).I    = inf; 
            running_tot = running_tot + length(inf);
        end
        
        disp(['+ ' allfolds(dirid(d)).name ' with ' num2str(running_tot) ' planes']); 
        
    end
else error('Richard you need to fix this - add something to look for files not just folders')

end