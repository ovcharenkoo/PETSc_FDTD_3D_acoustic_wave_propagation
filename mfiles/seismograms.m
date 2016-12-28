%% Plot seismograms

clc;
close all;
clear all;

MANUAL_RUN=1;                    %Compare file by file by pressing Enter ...

c_file=mfilename('fullpath');       %current path to this running script
c_file=strrep(c_file,mfilename,''); %remove name of script from path to get path to folder
pathh=[c_file '../seism/'];        %redirect to the folder with data files
fileList=dir(pathh); %get list of files to compare => number must be EVEN (2*n=even)
% fileList = fileList(~[fileList.isdir]); %remove hidden directories (like . and ..)

DSStore = find(strcmp({fileList.name},'.DS_Store'));     % Remove MAC OS files
if DSStore > 0  
    fileList = fileList(DSStore+1:end);
%     folder_list = file_names(~cellfun('isempty',file_names));
else
    fileList = fileList(3:end);
end

names = {fileList.name};            %sort by names
[S,sortorder] = sort_nat(names);    %

%[junk, sortorder] = sort([fileList.datenum]); %sort by date of creation

fileList = fileList(sortorder);     %list is now in ascending date order
numfiles = numel(fileList);         %get number of files inside the ./datasets
if isempty(fileList)
   disp('ERROR. Folder ./datasets is empty. Check if your files are there');
%    break;
end
h1 = figure;
WinOnTop(h1);

minvecx = [];
minvecy = [];
maxvecx = [];
maxvecy = [];
for i = 1:numfiles
    clf;
   cur_file=[pathh fileList(i).name]; %take a file from the first half of ./datasets
   A=dlmread(cur_file);             %read the file
   
   maxx = max(A(:,1));
   minx = min(A(:,1));
      
   maxy = max(A(:,2));
   miny = min(A(:,2));
   
   maxvecx = [maxvecx maxx];
   maxvecy = [maxvecy maxy];
   minvecx = [minvecx minx];
   minvecy = [minvecy miny];
   
   plot(A(:,2)); title(strrep(fileList(i).name,'_',' '));
   
%    axvec = [min(minvecx), max(maxvecx), min(minvecy), max(maxvecy)];
%    axis(axvec);
   
   drawnow;
   if MANUAL_RUN
     input('Press Enter...');       
   end

    
end
