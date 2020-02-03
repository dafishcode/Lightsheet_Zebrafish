% Housekeeping
%==========================================================================
F = cde_seeg_housekeeping;
S = cde_seeg_subs(F); 

%% Load an example subject
%--------------------------------------------------------------------------
sub = 1;
trl = 1;

hb = ft_read_header(S(sub).baseline{trl});
db = ft_read_data(S(sub).baseline{trl}); 
hs = ft_read_header(S(sub).seizure{trl}); 
ds = ft_read_data(S(sub).seizure{trl}); 

%% Calculate pairwise complex phase vectors and phase angle 
%--------------------------------------------------------------------------
PLI = cde_seeg_pli(hs,ds); 

%% In development
pmat = vertcat(PLI.pl); 
inv  = ~pmat;
seqs = []; 
for c = 1:size(inv,2)
    gaps = diff(find(inv(:,c)));
    gaps = gaps(gaps > 1)-1; 
    seqs = [seqs, gaps'];
end
    
hold on
[n edg] = histcounts(seqs,10);
loglog(edg(1:10), n)