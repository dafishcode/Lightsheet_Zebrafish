function fish = cde_tif_qualcheck(fish, nplanes, doplot)
% This function visualises an initial quality check for the data specified
% in the 'fish' object. If doplot is set to 1, it will plot video sequences
% of 75 images from the sequence to see whether a frame is dropped
% somewhere along the way. 
% The function also plots the correlation of any frame with any other image
% frame in order to visualise when such potential drops occur. 

fishname = [fish.reg '-' num2str(fish.num)];
conds = {'BL', 'PTZ'};

for cname = 1:length(conds)
    c = find(strcmp({fish.cond.name}, conds{cname})); 
    if isempty(c),  warning(['Could not find any mention of condition ' conds{cname}]);
    elseif length(c) > 2,   warning(['Unsure about the naming of condition ' conds{cname}]);
    else
        if isfield(fish.cond(c), 'newto')
            disp('Using reshuffled sequence');
            tomat = fish.cond(c).newto;
            frommat = fish.cond(c).frommat;
            if isempty(tomat),  error('New sequence map is empty - have you run all conditions?'); end
        else
            [tomat, frommat] = cde_tif_tifmats(fish, c, nplanes);
        end
        maxframes = 75; 
        pplot = 5;       % plane to plot
        if size(tomat,2) > maxframes,  tplot = fix(linspace(1,size(tomat,2),maxframes));
        else                           tplot = 1:size(tomat,2);     end
        toplot = tomat(pplot, tplot);

        % Load initial image matrix for comparison
        %--------------------------------------------------------------------------
        [plane,fileid]  = find(frommat == toplot(1));
        img_01          = imread(fish.cond(c).tif{fileid}, plane);
        clear corrval

        for t = 1:length(toplot)
            tp = toplot(t);
            [plane,fileid]  = find(frommat == tp);
            img             = imread(fish.cond(c).tif{fileid}, plane);

            if doplot
                figure(1)
                subplot(10,1,[1:9]), cla, imagesc(img);
                title([fishname ', this is condition: ' fish.cond(c).name])
                subplot(10,1,10), 	box off; axis off;
                                    plot([t,t],[0,1], 'k', 'linewidth', 3), ylim([0,1]), xlim([1,length(toplot)]);
                pause(.01)
                
            end
            corrval(t) = corr(double(img(:)), double(img_01(:)));
        end
        fish.cond(c).corrval = corrval; 
    end
end

figure(2)
bl = find(strcmp({fish.cond.name}, 'BL'));
pt = find(strcmp({fish.cond.name}, 'PTZ'));
subplot(2,1,1),     plot(fish.cond(bl).corrval), ylim([0,1]);   title([fishname ', baseline']);
subplot(2,1,2),     plot(fish.cond(pt).corrval), ylim([0,1]);   title([fishname ', PTZ']);
