function [ combinations ] = getMeasurementCombinations( numbMeas, minCams )
%GETMEASUREMENTCOMBINATIONS function that finds all possible combinations
% of the measurements form n different cameras-views
% 
%   Input:
%       numbMeas:           n-element array with numbers of measurements 
%                           form each camera
%       minCams:            minimal number of cameras to use for each combination
% 
%   Output:
%       combinations:       [M x n] array with all combinations.
%                           Zero-entries are entries with no valid measurement.
%                           (the combination has not n measurements)
% 

% check input parameters
nCams = length(numbMeas);
if any((size(numbMeas) ~= [1,nCams]) & (size(numbMeas) ~= [nCams,1]))
    error('numbMeas must have size of 1 x nCams or nCams x 1')
end

combinations = zeros(0,nCams);
for k = minCams:nCams
    camCombs = nchoosek(1:nCams,k);
    for j = 1:size(camCombs,1)
        argList = {};
        for i = camCombs(j,:)
            argList{end+1} = 1:numbMeas(i);
        end
        measCombs = combvec(argList{:}).';
        n = size(measCombs,1);
        combinations(end+1:end+n,camCombs(j,:)) = measCombs;
    end
end

end