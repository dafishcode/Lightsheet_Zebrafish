% function cde_windowslide(Fish, cfg)

c   = 2;
bm  = Fish.Cond{c}.Cell_labs.Brainmask;
dat = Fish.Cond{c}.Data(bm,:); 
win = 60 * 4; 
stp = win; 

srt = 1:stp:size(dat,2)-win;
for s = 1:length(srt)-1
    d = dat(:, srt(s):srt(s+1)-1);
    r = corr(d'); 
end


