housekeep   = 1;    % Run housekeeping section
fishload    = 1;    % Load fish data
datasegm    = 1;    % segment/collate data according to relevant atlas
winslide    = 0;    % run sliding window functional connectivity estimation
moremats    = 1;    % run 
regressm    = 0;

% Housekeeping
%==========================================================================
if housekeep
fs          = filesep;
F.scripts   = '/Users/roschkoenig/Dropbox/Research/1812 Critical Dynamics Epilepsy';
F.data      = '/Users/roschkoenig/Dropbox/Research/1812 Critical Dynamics Epilepsy Data';
F.temps     = [F.data fs 'Templates'];
F.analysis  = [F.data fs 'Analysis'];
F.cmn       = [F.analysis fs 'CMN']; 
F.addpath   = {'/Users/roschkoenig/Dropbox/Research/0002 Tools/Tools/cbrewer',
               '/Users/roschkoenig/Dropbox/Research/0002 Tools/Tools/csv2cell'};
for a = 1:length(F.addpath),    addpath(F.addpath{a});  end
addpath(genpath(F.scripts)); 
end

% Loading relevant files and preparing data structures
%==========================================================================
if fishload
    
% Load fishies
%--------------------------------------------------------------------------
fishlist = dir(F.cmn);
fishlist = {fishlist(~cellfun(@isempty, regexp({fishlist.name}, '^Z'))).name};

for f = 1:length(fishlist), Fish(f) = load([F.cmn fs fishlist{f} fs 'dat.mat']); end

% Load all templates
%--------------------------------------------------------------------------
tiflist = dir([F.temps fs 'Transgenic' fs '*.tif']);
tiflist = {tiflist.name};

clear tif
for t = 1:length(tiflist)
    slc = length(imfinfo([F.temps fs 'Transgenic' fs tiflist{t}]));
    for s = 1:slc
        tif{t}(:,:,s) = imread([F.temps fs 'Transgenic' fs tiflist{t}], s);
    end
end
end

% Chunk data into Pajevic segments
%--------------------------------------------------------------------------
if datasegm
plab = intersect(unique(Fish(1).Cond{1}.Cell_labs.Pajevic), unique(Fish(1).Cond{2}.Cell_labs.Pajevic));
spac = [.3, .3, 1]; 
D    = [];

clear d Nd
ci = 0;
c  = 2;

for p = 1:length(plab)
    if plab(p) ~= 0
    ind = Fish(1).Cond{c}.Cell_labs.Pajevic == plab(p); 
 
    lft = Fish(1).Cond{c}.Cell_labs.Side == 0;
    lid = intersect(find(lft),find(ind)); 
    if ~isempty(lid)
        ci = ci + 1;
        Nd(ci).d = mean(Fish(1).Cond{c}.Data(lid,:),1); 
        Nd(ci).s = 0; 
        Nd(ci).p = plab(p);
        Nd(ci).k = mode(Fish(1).Cond{c}.Cell_labs.KunstID(lid));
        Nd(ci).l = mean(Fish(1).Cond{c}.ZBBCoord(lid,:),1); 
        Nd(ci).tp = Fish(1).Cond{c}.ZBBCoord(lid,:) ./ spac; 
    end
    
    rit = Fish(1).Cond{c}.Cell_labs.Side == 1; 
    rid = intersect(find(rit),find(ind)); 
    if ~isempty(rid)
        ci = ci + 1;
        Nd(ci).d = mean(Fish(1).Cond{c}.Data(rid,:),1); 
        Nd(ci).s = 1;
        Nd(ci).p = plab(p); 
        Nd(ci).k = mode(Fish(1).Cond{c}.Cell_labs.KunstID(rid));
        Nd(ci).l = mean(Fish(1).Cond{c}.ZBBCoord(rid,:),1);
        Nd(ci).tp = Fish(1).Cond{c}.ZBBCoord(rid,:) ./ spac; 
    end
end
end
end

% Calculating of sliding window connectivity matrices
%==========================================================================
%--------------------------------------------------------------------------
if winslide
win = 30 * 60 * 4; 
stp = fix(win/2);
dat = vertcat(Nd.d); 
srt = 1:stp:size(dat,2)-win;

clear Func
for s = 1:length(srt)-1
    d = dat(:,srt(s):srt(s+1)-1) + 0.0001; 
    Func{s} = corr(d'); 
    Func{s}(Func{s} < 0) = 0; 
end
end


% Calculating of relevant predictor matrices
%==========================================================================
%--------------------------------------------------------------------------
if moremats
    
% Calculate baseline connectivity matrix
%==========================================================================
% Find clusters that correspond to functional nodes in seizure data
%--------------------------------------------------------------------------
clear Bl
bas     = Fish(1).Cond{1}.Data; 
bpaj    = Fish(1).Cond{1}.Cell_labs.Pajevic; 
bsid     = Fish(1).Cond{1}.Cell_labs.Side;
for n = 1:length(Nd)
    bi = intersect(find(bpaj == Nd(n).p), find(bsid == Nd(n).s));
    Bl(n).d = mean(bas(bi,:),1); 
end

% Calculate the actual connectivity matrix
%--------------------------------------------------------------------------
covariance = 0;
if covariance 
    for r = 1:length(Bl)
    disp(['Working on row ' num2str(r) ' of ' num2str(length(Bl))]); 
    for c = 1:length(Bl)
        Base(r,c) = mean(mean(wcoherence(Bl(r).d, Bl(c).d))); 
    end
    end

else
    Base = corr(vertcat(Bl.d)); 
    Base(Base < 0) = 0; 
end


% Distance
%==========================================================================
% Estimate Pajevic coordinates in standard space - Load template
%--------------------------------------------------------------------------
Dist = dist(vertcat(Nd.l)'); 

%% Structural connectivity
%==========================================================================
load([F.temps fs 'Connectivity Matrices' fs 'Structural_connectivity.mat']);
load([F.temps fs 'Connectivity Matrices' fs 'Regions.mat']);
cnx = flip(connectivity_matrix');
reg = regions; 
ki  = [Nd.k]; 
kwh = find(ki); 

Stru = zeros(size(Nd,2)); 
Stru(kwh,kwh) = cnx(ki(kwh),ki(kwh));

% Add additional within-region components to structural connectivity matrix
%--------------------------------------------------------------------------
pajcode = csv2cell([F.temps fs 'Pajevic_annotations.csv'], 'fromfile');
clear pajid
for p = 1:size(pajcode,1)-1,  pajid(p) = str2double(pajcode{p+1});  end
gross = unique(pajcode(2:end, 3)); 

% Label nodes by gross anatomical region
%--------------------------------------------------------------------------
for n = 1:length(Nd)
    id          = find(pajid == Nd(n).p); 
    thisgross   = pajcode{id+1, 3}; 
    Nd(n).g     = find(strcmp(gross, thisgross)); 
end

% Add corresponding connectivity
%--------------------------------------------------------------------------
mcx = mode(Stru(Stru > 0));
stru = Stru; 
for g = [1 2 3 5]   % Diencephalon, Hindbrain, Midbrain and Telencephalon
    tid = find([Nd.g] == g);
    stru(tid,tid) = Stru(tid,tid) + mcx; 
end

% Path length
%--------------------------------------------------------------------------
L       = -log(stru+min(stru(stru>0))); 
[D,B]   = distance_wei(L); 
Path    = D; 

% Search information
%--------------------------------------------------------------------------
L       = -log(stru+min(stru(stru>0))); 
W       = stru;     W(W == 0) = min(stru(stru > 0));
Srch    = search_information(W, L); 

% Maximum flow
%--------------------------------------------------------------------------
Maxf    = zeros(size(stru)); 
G       = graph((stru + stru')./2); 
for r = 1:length(stru)
for c = 1:length(stru)
    if r ~= c, Maxf(r,c) = maxflow(G,r,c);  end
end
end

% Genetic covariance
%==========================================================================
% Associate each node with gene expression vectors
%--------------------------------------------------------------------------
clear val
for tf = 1:length(tif)
disp(['Working on tif ' num2str(tf) ' of ' num2str(length(tif))])
for ni = 1:length(Nd)
tlcs    = round(Nd(ni).tp); 
tt      = tif{tf}; 
for tl = 1:size(tlcs,1)
    val(tl) = tt(tlcs(tl,2), tlcs(tl,1), tlcs(tl,3));
end
Nd(ni).t(tf) = mean(val); 
end
end

% Generate actual covariance
%--------------------------------------------------------------------------
Gene = cov(vertcat(Nd.t)'); 
end

% Run regression
%==========================================================================
if regressm
dummy = ones(size(Func{1})); 
dummy = triu(dummy,1); 
id    = find(dummy); 

clear B
for s = 1:length(Func)
    Y       = Func{s}(id);
    X       = [dummy(id), Path(id), Srch(id), Maxf(id), Dist(id), Gene(id), Base(id)]; 
    B(s,:)  = regress(Y,X);
end
end


%% Reconstitute and compare
%--------------------------------------------------------------------------
for s = 1:length(Func)
    subplot(1,length(Func),s)
    Y   = Func{s}(find(dummy));
    cid = find(Y > 0); 
    prd = X * B(s,:)';
    prd = prd(cid); 
    obs = Y(cid);
    
    scatter(obs, prd, [], 'filled', 'markerfacealpha', .2)
    title(corr(obs, prd)); 
end




% %% Code graveyard RIP
% %==========================================================================
% %% Plot an example tif with cells of single node
% %--------------------------------------------------------------------------
% nid = 52; 
% clf
% figure(1)
% locs = vertcat(Nd.tp);      locs(:,3) = locs(:,3); 
% tlcs = round(Nd(nid).tp);   tlcs(:,3) = tlcs(:,3); 
% 
% tt = tif{1}; 
% colormap(flip(cbrewer('div', 'Spectral', 100)));
% subplot(2,1,1), imagesc(squeeze(max(tt, [], 3))),   set(gca, 'Ydir', 'normal'); hold on
% subplot(2,1,2), imagesc(squeeze(max(tt, [], 1))'),  hold on %%set(gca, 'Ydir', 'normal'); hold on
% 
% clear val
% for t = 1:length(tlcs)
%     val(t) = tt(tlcs(t,2), tlcs(t,1), tlcs(t,3));
% end
% 
% subplot(2,1,1)
% scatter(locs(:,1), locs(:,2), [], [.5 .5 .5], 'filled', 'MarkerFaceAlpha', .2); hold on,   
% ylim([1 616]); xlim([1 1030]);
% scatter(tlcs(:,1), tlcs(:,2), 100, val, 'filled');                       
% ylim([1 616]); xlim([1 1030]);
% 
% subplot(2,1,2)
% scatter(locs(:,1), locs(:,3), [], [.5 .5 .5], 'filled', 'MarkerFaceAlpha', .2); hold on,  
% ylim([1 420]); xlim([1 1030]);
% scatter(tlcs(:,1), tlcs(:,3), 100, val, 'filled');                       
% ylim([1 420]); xlim([1 1030]);
% 
% 
% %% Segment into parcels relevant to structural connectivity matrix
% %--------------------------------------------------------------------------
% % Load structural connectivity
% %--------------------------------------------------------------------------
% ST = load([F.temps fs 'Connectivity Matrices' fs 'Structural_connectivity.mat']);
% load([F.temps fs 'Connectivity Matrices' fs 'Regions.mat'], 'regions'); 
% 
% % Identify Kunst IDs and names from functional data
% %--------------------------------------------------------------------------
% clear ki
% for c = 1:2
% ki{c} = unique(Fish(1).Cond{c}.Cell_labs.KunstID); 
% ki{c} = ki{c}(ki{c} > 0); 
% end
% 
% nki     = intersect(ki{1}, ki{2});
% clear D kn
% for c = 1:2
% knames  = cellstr(Fish(1).Cond{c}.Cell_labs.Kunst);
% for k = 1:length(nki)
%     ci = find(Fish(1).Cond{c}.Cell_labs.KunstID == nki(k));
%     dt(k,:) = mean(Fish(1).Cond{c}.Data(ci,:),1); 
%     kn{k}   = knames{ci(1)};
% %     if isstrprop(kn{k}(1), 'upper'), kn{k} = upper(kn{k});  end
% end
% 
% D(c).dt = dt;
% D(c).ki = nki;
% D(c).kn = kn;
% clear dt
% end
% 
% %% Plot subsection of structural connectivity
% %--------------------------------------------------------------------------
% for n = 1:length(D(1).kn)
%     kn = D(1).kn{n};
%     id(n) = find(strcmp(regions,kn));
% end
% [std, stg] = sort(id);
% 
% subplot(1,2,1)
% stcx = flip(ST.connectivity_matrix); 
% imagesc(log(stcx(id(stg),id(stg))))
% axis square
% 
% subplot(1,2,2)
% % cr = corr(D(1).dt'); 
% cr = JMatNews{end};
% imagesc(cr(stg,stg), [0 4]);
% axis square
% 
% 
% %% Identifying network transitions
% %--------------------------------------------------------------------------
% plab = intersect(unique(Fish(1).Cond{1}.Cell_labs.Pajevic), unique(Fish(1).Cond{2}.Cell_labs.Pajevic));
% D    = [];
% for c = 1:2
% clear d
% for p = 1:length(plab)
%     ind = Fish(1).Cond{c}.Cell_labs.Pajevic == plab(p); 
%     d(p,:) = mean(Fish(1).Cond{c}.Data(ind,:),1);  
% end
% D = [D, d];
% end
% 
% %% 
% for d = 1:size(D,1)
%     plot(D(d,:) + 10*d); hold on
% end
%     
% %% 
% [scr cof] = pca(D); 
% scatter3(scr(:,1), scr(:,2), scr(:,3), [], [1:size(scr,1)], 'filled'); 
% 
% %%
% ranges = [1,7200; 7301, 14500; 14501, 21600];
% for k = 1:3
%     subplot(3,1,k)
%     r = ranges(k,1):ranges(k,2); 
%     scatter3(scr(r,1), scr(r,2), scr(r,3), [], [1:length(r)], 'filled'); 
% end
