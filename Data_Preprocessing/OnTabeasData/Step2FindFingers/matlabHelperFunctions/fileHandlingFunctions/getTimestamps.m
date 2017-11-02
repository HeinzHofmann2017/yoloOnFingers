function [ t ] = getTimestamps( directory )
%GETTIMESTAMPS read timespamps from .txt-file
%   
%   Input: 
%       dir:    directory in which the file is saved
%
%   Output:
%       t:      array with timestamps in milliseconds
% 

t = [];

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
    if ( strcmp(file(end-3:end),'.txt') )
        fid = fopen(strcat(directory,file));
        tline = fgetl(fid);
        while ischar(tline)
            C = strsplit(tline);
            if strcmp(C{end},'ms')
                t(end+1) = str2double(C{end-1});
            end
            tline = fgetl(fid);
        end
        fclose(fid);
        break;
    end
end

end

