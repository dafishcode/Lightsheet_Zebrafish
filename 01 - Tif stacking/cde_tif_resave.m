function cde_tif_resave(Fish, Fsave)
fs = filesep;
for f = 1 : length(Fish)
    fish = Fish(f); 
    fishname = [fish.reg '_' num2str(fish.num, '%02.f')];
    disp('=======================================================');
    disp(['Currently working on fish ' fishname ', which is ' num2str(f) ' of ' num2str(length(Fish))])
    disp('=======================================================');
    conds = {'BL', 'PTZ'};
    
    for ci = 1:length(conds)
        disp(['Working on ' conds{ci} ' images']);
       
        fishfold = [Fsave fs fishname fs conds{ci}];
        if isdir(fishfold),     rmdir(fishfold, 's'); end
        mkdir(fishfold); 
        c = find(strcmp({fish.cond.name}, conds{ci})); 
        
        tomat       = fish.cond(c).newto;
        frommat     = fish.cond(c).frommat;
        
        for t = 1 : size(tomat,2)
        if mod(t,10) == 0
           fprintf('%s', '.')
        end
        if mod(t,100) == 0  
            disp([num2str(t) ' of ' num2str(size(tomat,2)) ' done for fish ' num2str(f) ' of ' num2str(length(Fish))]),  
        end  
        for p = 1:size(tomat,1)
            [plane fileid] = find(frommat == tomat(p,t));
            img            = imread(fish.cond(c).tif{fileid}, plane);
            imwrite(uint16(img), ...
                [fishfold fs num2str(t,'%05.f') '_' conds{ci}(1) '.tif'], ...
                'writemode', 'append', 'compression', 'none');
        end
        end
        disp('All files for this condition now done'); 
    end
end
