% Modelled after Zhigalov et al J Neurosci 2015

% 10 min 1kHz; closest white matter referencing; 
% filtered 1-40Hz; baseline corrected, divided by SD (zscored)
% Identified peaks > threshold T ranging from 1.5 to 5.25 (0.25 steps)
% summing events across electrodes in different time bins (dt - 4:4:80ms)
% avalanche = cluster of events in successive time bins 

% Long range temporal correlation exponent beta 

% Housekeeping
%==========================================================================
F = cde_seeg_housekeeping;
S = cde_seeg_subs(F); 

% Load an example subject
%--------------------------------------------------------------------------
sub = 1;
trl = 1;

hb = ft_read_header(S(sub).baseline{trl});
db = ft_read_data(S(sub).baseline{trl}); 
hs = ft_read_header(S(sub).seizure{trl}); 
ds = ft_read_data(S(sub).seizure{trl}); 

% Preprocessing
%==========================================================================
% Bandpass filter [1 - 40Hz]
%--------------------------------------------------------------------------
db_f = ft_preproc_bandpassfilter(db, hb.Fs, [1 40], 1800, 'firws'); 
ds_f = ft_preproc_bandpassfilter(ds, hs.Fs, [1 40], 1800, 'firws');

% Z-score
%--------------------------------------------------------------------------
d_z = zscore([db_f ds_f]')';
db_z = d_z(:,1:size(db_f,2));
ds_z = d_z(:,size(db_f,2)+1:end);

%% Avalanche detection
%==========================================================================
% Binarise traces
%--------------------------------------------------------------------------
bb = zeros(size(db_z)); 
bs = zeros(size(ds_z)); 

for ch = 1:size(db_z,1)
    mph = 0.5;
    [vals locs] = findpeaks(db_z(ch,:), 'MinPeakHeight', mph);
    bb(ch,locs) = ones(1,length(locs)); 
    
	[vals locs] = findpeaks(ds_z(ch,:), 'MinPeakHeight', mph);
    bs(ch,locs) = ones(1,length(locs)); 
end

% Calculate total number of events per bin
%--------------------------------------------------------------------------
bin = fix(0.004 * hb.Fs);
k   = 0; 
clear avb
for b = 1:bin:length(bb)-bin
    k = k + 1;
    win = b:b+bin; 
    avb(k) = sum(sum(bb(:,win))); 
end

k   = 0; 
clear avs
for b = 1:bin:length(bs)-bin
    k = k + 1;
    win = b:b+bin; 
    avs(k) = sum(sum(bs(:,win))); 
end

% Identify individual avalanches (from zero to zero)
%--------------------------------------------------------------------------
zeropos = [0 find(avb == 0) length(avb)+1];
k       = 0;
clear ab
for z = 1:length(zeropos)-1
    avlength = zeropos(z+1) - zeropos(z);
    if avlength > 1
        k = k + 1;
        ab(k).length = avlength-1;
        srt     = max(zeropos(z),1);  % in case zero is leading
        stp     = min(zeropos(z+1),length(avb)); % in case avalanche lasts to end 
        ab(k).size   = sum(avb(zeropos(z):zeropos(z+1))); 
    end    
end

zeropos = [0 find(avs == 0) length(avs)+1];
k       = 0;
clear as
for z = 1:length(zeropos)-1
    avlength = zeropos(z+1) - zeropos(z);
    if avlength > 1
        k = k + 1;
        as(k).length = avlength-1;
        srt     = max(zeropos(z),1);  % in case zero is leading
        stp     = min(zeropos(z+1),length(avs)); % in case avalanche lasts to end 
        as(k).size   = sum(avs(srt:stp)); 
    end    
end

edges = exp([linspace(0, log(max([ab.size, as.size])), 10), log(max([ab.size, as.size]))]);
bnobs = histcounts([ab.size],edges);
snobs = histcounts([as.size],edges); 
loglog(edges(1:end-1), bnobs, 'k'); hold on
loglog(edges(1:end-1), snobs, 'r');



