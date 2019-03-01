% cde_tif_volseg(fish

for c = 1
    ntifs   = sum(fish.cond(c).tifn);
    tim_min = ntifs / Fs / 60;
    disp('=============================================================')
    disp(['I found ' num2str(ntifs) ' tifs, covering ' num2str(tim_min) ...
            ' min at a sampling rate of ' num2str(Fs) ' Hz']) 
    if mod(ntifs, Fs) ~= 0,   warning('The number of planes does not break into volumes evenly at this sampling rate');   end
    
    % Set up folder structure
    %----------------------------------------------------------------------
    [s,m] = mkdir([Fsave fs fish.reg '_' num2str(fish.num, '%02.f')]);
    
    
end
