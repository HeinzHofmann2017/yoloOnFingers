function [ imageFileNamesW, imageFileNamesU,imageFileNamesA ] = ...
                                            getImageFileNames( directory )
%GETIMAGEFILENAMES gets a list of all pictures found in the directory
%   
%   Input: 
%       directory:          directory in which the images are saved
%
%   Output:
%       imageFileNamesW:    cell array with all png images ending with 'w' (white)
%       imageFileNamesU:    cell array with all png images ending with 'u' (uv)
%       imageFileNamesA:    cell array with all png images ending arbitrary
% 

imageFileNamesW = {};
imageFileNamesU = {};
imageFileNamesA = {};

if directory(end) ~= '/'
    directory(end+1) = '/';
end

listing = dir(directory);
files = {};

for i=1:length(listing)
    if ( listing(i).isdir == 0 )
        files{end+1} = listing(i).name;
    end
end

for i=1:length(files)
    file = files{i};
    if ( strcmp(file(end-4:end),'w.png') )
        % collect images with ending 'w' 
        imageFileNamesW{end+1} = [directory,file];
    elseif ( strcmp(file(end-4:end),'u.png') )
        % collect images with ending 'u' 
        imageFileNamesU{end+1} = [directory,file];
    elseif ( strcmp(file(end-3:end),'.png') )
        % collect images with arbitrary ending
        imageFileNamesA{end+1} = [directory,file];
    end
end

end

