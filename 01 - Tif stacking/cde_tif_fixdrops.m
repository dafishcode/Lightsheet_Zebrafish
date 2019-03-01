function newto = cde_tif_fixdrops(fish, nplanes,cond,flipframe)
% Flipframe should be the last normal frame
if flipframe <= 1 
    disp('Need to start with higher frame number')
    flipframe = flipframe + 1;
end

fishname    = [fish.reg '-' num2str(fish.num)];
c           = find(strcmp(cond, {fish.cond.name}));
longids     = [1:10, 1:10, 1:10];
oldons      = 1;
newons      = 1;


disp(['Now working on ' fishname ' in condition ' fish.cond(c).name]);
if isfield(fish.cond(c), 'newto') 
if ~isempty(fish.cond(c).newto)
    disp('Using existing reshuffling');
    tomat = fish.cond(c).newto;
    frommat = fish.cond(c).frommat;
else,   [tomat, frommat] = cde_tif_tifmats(fish, c, nplanes);   end
else,   [tomat, frommat] = cde_tif_tifmats(fish, c, nplanes);
end

srt = flipframe - 1;   % Starting frame
stp = flipframe + 1;   % Stopping frame
seq = srt:stp;


figure(1), set(gcf, 'Position', [200,200, 600, 1200]);
stopit = 0;

while ~stopit
    for s = 1:length(seq)
        if s < length(seq)
            pl 	= longids(oldons : oldons+nplanes-1);
        else   
            pl  = longids(newons : newons + nplanes - 1); 
        end
        
        sid = seq(s);

        for p = 1:length(pl)
            pp              = pl(p);
            toplot          = tomat(pp, sid);
            [plane,fileid]  = find(frommat == toplot);
            figure(1)
            subplot(length(pl), length(seq), s + (p-1)*length(seq)), 
                imagesc(imread(fish.cond(c).tif{fileid}, plane))
                title(['Plane ' num2str(pp) ', Volume No ' num2str(sid)])
        end
    end
    
    % Get user input and save new order of planes
    %----------------------------------------------------------------------
    gd = input('All good? (0 - no; 1 = yes): ');
    if ~gd
        oldons  = input('What is the original first plane (99 to stop): '); 
        if oldons == 99,    break; end 
        newons  = input('What is the new first plane (99 to stop): '); 
        if newons == 99,    break; end 
    else 
        stopit  = 1;
        dlff = input('Delete flip frame (0 - no, 1 - yes): ');
        for t = 1:size(tomat,2)
            if t <= flipframe,  npl = longids(oldons : oldons+nplanes-1);
            else,               npl = longids(newons : newons+nplanes-1); 
            end
            
            newto(:,t) = tomat(npl, t); 
        end
        if dlff
            newto = horzcat(newto(:,1:flipframe-1), newto(:, flipframe+1:size(newto,2)));
        end
    end
end