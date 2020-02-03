function P = cde_seeg_pli(h,d)
% Scale dependent estimate of phase difference in EEG dataset

% Filter into 31-63 gamma Hz range %% This shoud be done in a wavelet dec
%--------------------------------------------------------------------------
Fs   = h.Fs;
Fbp  = [31 63];
ft   = fir1(1000, [Fbp(1)/(Fs/2), Fbp(2)/(Fs/2)]); 
df   = filtfilt(ft, 1, d'); 
dh   = hilbert(df);
 
% Chunk into relevant windows
%--------------------------------------------------------------------------
Lwin = ceil(8/Fbp(2) * Fs); 
srts = 1:Lwin:size(df,1); 
tim  = srts(1:end-1)/Fs; 

% Estimate compplex phase vector and local mean phase difference
%==========================================================================
% C_i,j(t) = [W_k*(F_i)+ * W_k(F_j)] / [|W_k(F_i)|*|W_k(F_j)|'] - averaged
% over 8 cycles highest frequency; 4 cycles lowest frequency (given wavelet
% width)
% W_k - kth scale of Hilbert wavelet transform; W+: complex conjugate

% Find indices for unique pairs 
%--------------------------------------------------------------------------
dummy = ones(size(df,2)); 
[r,c] = find(triu(dummy, 1));

clear P

for k = 1:length(r)
    
    if mod(k,250) == 0,  disp(['Channel pair ' num2str(k)]); end
    P(k).chans = sort([r(k),c(k)]);
    
    % For each time step calculate instantaneous complex phase vector
    %----------------------------------------------------------------------
    for s = 1:length(srts)-1
        twin        = srts(s):srts(s+1)-1;
        P(k).C(s)   = mean(conj(dh(twin,r(k))) .* dh(twin,c(k))) / ...
                      sqrt(mean(abs(dh(twin,r(k))).^2) * mean(abs(dh(twin(c(k))).^2))); 
    end
    
    % Convert into measures of phase difference and significance
    %----------------------------------------------------------------------
    P(k).dA = angle(P(k).C);
    P(k).sD = abs(P(k).C).^2;

    % Apply thresholding
    %----------------------------------------------------------------------
    crit1   = find(abs(P(k).dA) < pi/4);
    crit2   = find(P(k).sD > 0.5);
    P(k).pl = zeros(size(P(k).dA)); 
    P(k).pl(intersect(crit1,crit2)) = 1; 

end

% Synthetic data
%--------------------------------------------------------------------------
% x = linspace(0,50,1000);  
% tf = [ones(1,200), linspace(1,0,200), zeros(1,200), linspace(0,1,200), ones(1,200)]; 
% y1 = sin(x) .* tf + cos(x) .* (1-tf); 
% y2 = sin(x); 

% plcked = abs(can) < pi/4;
% sdev   = abs(C).^2;
% Plckd  = intersect(find(plcked > 0), find(sdev > .5)); 
% plot(abs(can)), hold on
% plot(Plcked)
% % plot(crl)

% 
% subplot(2,1,1)
% imagesc(vertcat(P.pl))
