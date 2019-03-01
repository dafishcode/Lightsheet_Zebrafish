function cde_tif_finddrops(fish, nplanes,cond)

fishname = [fish.reg '-' num2str(fish.num)];
c = find(strcmp(cond, {fish.cond.name}));

disp(['Now working on ' fishname ' in condition ' fish.cond(c).name]);
% Identify total number of step sizes
[tomat, frommat] = cde_tif_tifmats(fish, c, nplanes);

ts  = 10;               % Time steps to plot in one go
pl  = [1 3 7 10];        % Plane ID to plot

srt = 1;                % Starting frame
stp = size(tomat,2); % size(tomat,2);    % Stopping frame
stopnow = 0; 

while ~stopnow
    seq = fix(linspace(srt, stp, ts)); 
    figure(1), set(gcf, 'Position', [400,800, 1800, 600]);
    for s = 1:length(seq)
        sid             = seq(s);

        for p = 1:length(pl)
            pp              = pl(p);
            toplot          = tomat(pp, sid);
            [plane,fileid]  = find(frommat == toplot);
            figure(1)
            subplot(length(pl), ts, s + (p-1)*ts), 
                imagesc(imread(fish.cond(c).tif{fileid}, plane))
                title(['Plane ' num2str(pp) ', Volume No ' num2str(sid)])
        end
    end
    figure(2), clf;  set(gcf, 'Position', [400,400,1800,200]);
    plot(linspace(1,size(tomat,2),length(fish.cond(c).corrval)), fish.cond(c).corrval); ylim([0 1]); xlim([-Inf Inf]); hold on
    for s = 1:length(seq)
        plot([seq(s), seq(s)], [0,1], 'k');
    end

    srtstr = input('New start frame: [x for exit]: ', 's');  if srtstr == 'x'; break, end
    stpstr = input('New stop frame: [x for exit]: ', 's');   if stpstr == 'x'; break, end
    srt = str2double(srtstr);
    stp = str2double(stpstr); 
    if stp - srt < ts, stopnow = 1; disp(['Increment smaller than the expected ' num2str(ts) ' frames']);    end 
    
end

