function Fish = cde_tif_fishfinder(Tish)
fs  = filesep;

% Datafile descriptions
%==========================================================================
% Find and specify fish details with available datasets
%--------------------------------------------------------------------------
fishdir = dir(Tish.base);
fishdir = {fishdir.name};

% Regular expression has to match specified name
%--------------------------------------------------------------------------
disp('--------------------------------------------------')
disp('** Looking for files with specific format name: **')
disp('--------------------------------------------------')
disp('      Date (YYMMDD) _ # of dpf _ Subject ID _ Line descriptor _ Experiment _ # fish')
disp('      e.g.: 180705_6dpf_ZFRR002_GRIN2A_PTZ_01');
disp('      These folders must contain a ''BL'' and a ''PTZ'' folder')

nametemp = '^[0-9]+_[0-9]+dpf_[A-Z]+.*_[A-Z]+.*[A-Z]+.*_[0-9]+$';
fishid   = find(~cellfun(@isempty, regexp(fishdir, nametemp)));
fcnt     = 0;

% Go through all available folder and compare to specs
%--------------------------------------------------------------------------
for f = 1:length(fishid)
    fid             = fishid(f);
    dirname         = fishdir{fid};
    seppos          = find(dirname == '_');
    
    agestr = dirname(seppos(1)+1 : seppos(2)-1);    age = str2double(agestr(1:strfind(agestr, 'dpf')-1));
    regstr = dirname(seppos(2)+1 : seppos(3)-1);
    linstr = dirname(seppos(3)+1 : seppos(4)-1);
    expstr = dirname(seppos(4)+1 : seppos(5)-1);
    numstr = dirname(seppos(5)+1 : end);
    
    fish.fold    = {dirname};
    fish.reg     = regstr;
    fish.num     = str2double(numstr);
    fish.age     = age;
    fish.reg     = regstr;
    fish.lin     = linstr;
    fish.exp     = expstr;
    fish.num     = str2double(numstr);
    fish.path    = [Tish.base fs dirname];
    
    % Check compliance with template fish
    %-----------------------------------------------------------------------
    if  age == Tish.age && strcmp(linstr, Tish.type) && strcmp(expstr, Tish.exp) 
        fish_subfold = dir([Tish.base fs fish.fold{1}]);
        condlist = {fish_subfold.name};
        got_b = ~isempty(find(~cellfun(@isempty, regexp(condlist, '^BL$'))));
        got_p = ~isempty(find(~cellfun(@isempty, regexp(condlist, '^PTZ$'))));
        if got_b && got_p
            fcnt = fcnt + 1;
            Fish(fcnt) = fish;        
        end
    end
end

disp(' ');
disp(['----- I found ' num2str(fcnt) ' datasets that meet all conditions'])