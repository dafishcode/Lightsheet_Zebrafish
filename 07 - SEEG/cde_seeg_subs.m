function S = cde_seeg_subs(F)

fs      = filesep; 
names   = dir(F.data);
names   = {names.name};

for n = 1:length(names)
    S(n).name = names{n}; 
    S(n).fold = [F.data fs names{n}];
    
    % Find all baseline EDF files
    %----------------------------------------------------------------------
    bnames = dir([S(n).fold fs '*_Baseline*']); bnames = {bnames.name};
    for b = 1:length(bnames)
        S(n).baseline{b} = [S(n).fold fs bnames{b}]; 
    end
    if b ~= 3,  disp(['For subject ' num2str(n) ' I found ' num2str(b) ' seizure files']);  end
    
    % Find all seizure EDF files
    %----------------------------------------------------------------------
    snames = dir([S(n).fold fs '*_Seizure*']);  snames = {snames.name}; 
    for s = 1:length(snames)
        S(n).seizure{s} = [S(n).fold fs snames{s}];
    end
    if s ~= 3,  disp(['For subject ' num2str(n) ' I found ' num2str(s) ' seizure files']);  end
    
end

