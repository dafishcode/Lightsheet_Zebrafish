% cde_tif_run
%--------------------------------------------------------------------------
% This code runs the relevant steps to take raw TIF data from the light
% sheet microscope, and convert them to stacks that can be used for
% subsequent processing
%--------------------------------------------------------------------------
%% Housekeeping
%==========================================================================
D       = cde_tif_housekeeping;
Fish    = D.Fish;
Fsave   = D.Fsave;
Fscript = D.Fscript;
fs      = filesep;


%% Find locations and file information
%==========================================================================
Fishfile = [Fscript fs '01 - Tif stacking' fs 'Fish.mat'];
if isfile(Fishfile),    load(Fishfile);     
else 
    clear Tempfish
    for f = 1:length(Fish)
        % Specify file locations and numbers
        %--------------------------------------------------------------------------
        fish            = Fish(f);
        fish            = cde_tif_find(fish);	 % Can't find an efficient way of doing this 
        Tempfish(f)     = fish; 
    end
    Fish = Tempfish; 
    save(Fishfile, 'Fish');
end
    
%% Initial quality check - to identify skipped frames
%==========================================================================
f = 11;  c = 'PTZ';
fish = Fish(f); 
fish = cde_tif_qualcheck(fish, 10, 0);
cde_tif_finddrops(fish, 10, c)

%% Identify and fix dropped frames
%==========================================================================
f           = 11; 
c           = 'PTZ'; 
flipframe   = 6815;     % Flipframe should be the last frame that's in the original order

nplanes     = 10; 
vidplot     = 1;
fish = Fish(f);
newto = cde_tif_fixdrops(fish, nplanes, c, flipframe);

% Pack into fish structure
%--------------------------------------------------------------------------
ci = find(strcmp({fish.cond.name}, c));
[tomat, frommat] = cde_tif_tifmats(fish, ci, nplanes);
Fish(f).cond(ci).frommat = frommat; 
Fish(f).cond(ci).tomat   = tomat;
Fish(f).cond(ci).newto   = newto;

%% Review video with new order
%==========================================================================
fish = Fish(f); 
figure(1),  set(gcf, 'Position', [400, 400, 1200, 600]); 
cde_tif_qualcheck(fish, nplanes, vidplot); 

%% Save to fish folder
%--------------------------------------------------------------------------
save([fish.path fs fish.reg '_' num2str(fish.num, '%02.f') '_reshuffle'], 'fish'); 
disp(['Saved to ' fish.reg]); 

%% Save files into new tif order
%==========================================================================
cde_tif_resave(Fish, Fsave); 
