function [ worldPointsHat, numberOfUsedViews, errorVal ] = extensiveWorldPointEstimation( imagePoints, cameraMatrices, nViewsToUse, maxErrorVal )
%EXTENSIVEWORLDPOINTESTIMATION function that finds for a set of image-points 
%   from different camera-views, all possible 3D-world-points
% 
%   Input:
%       imagePoints:        nView element cell-array containing undistorted 
%                           image-points (as [Ni x 2] arrays) of all camera views
%       cameraMatrices:     nView element cell-array containing 
%                           [3x4] camera matrizes of the differnt cameras
%       nViewsToUse:        array with number of Views to use for the
%                           estimation of one point (2...nViews)
%       maxErrorVal:        maximum permitted value of the error-surface
%       
%   Output:
%       worldPointsHat:     estimation of world-points [N x 3] sorted by
%                           the value of the error-surface
%       numberOfUsedViews:  number of camera-views, that have been used to
%                           estimate that point [N x 1]
%       errorVal:           value of the error-surface [N x 1]
% 

% check input parameters
if any((size(imagePoints) ~= size(cameraMatrices)) & (size(imagePoints) ~= fliplr(size(cameraMatrices))))
    error('imagePoints and cameraMatrices must have the same number of elements')
end
if any((size(imagePoints,1) ~= 1) & (size(imagePoints,2) ~= 1))
    error('imagePoints must have size of [1 x nView] or [nView x 1]')
end
if any((size(cameraMatrices,1) ~= 1) & (size(cameraMatrices,2) ~= 1))
    error('cameraMatrices must have size of [1 x nView] or [nView x 1]')
end
[~,nCol] = cellfun(@size,imagePoints);
if any(nCol ~= 2)
    error('elements of imagePoints must have size of [Ni x 2]')
end
[nRow,nCol] = cellfun(@size,cameraMatrices);
if any((nRow ~= 3) | (nCol ~= 4))
    error('elements of cameraMatrices must have size of [3x4]')
end
if ( any(nViewsToUse < 2) || any(nViewsToUse > length(cameraMatrices)) )
    error('elements of nViewsToUse must lie in the range of 2...nViews')
end

% get all possible measurement-combinations
[numbMeas,~] = cellfun(@size,imagePoints);
combinations = getMeasurementCombinations( numbMeas, 2 );
combinationsBin = logical(combinations);

% determine number of used views and discard invalid combinations
numberOfUsedViews = sum(combinationsBin,2);
index = false(size(numberOfUsedViews,1),1);
for nViews = nViewsToUse
    index = index | (numberOfUsedViews == nViews);
end
combinations = combinations(index,:);
combinationsBin = combinationsBin(index,:);
numberOfUsedViews = numberOfUsedViews(index);

% calculate for every combination the 3D-World-Point
N = size(combinations,1);
worldPointsHat = NaN(N,3);
errorVal = NaN(N,1);
camCombs = unique(combinationsBin,'rows');
nCams = sum(camCombs,2);
for k = 1:size(camCombs,1)
    
    % collect all measurements of the same camera-combination
    index = find(~sum(camCombs(k,:) ~= combinationsBin,2));
    nMeas = length(index);
    points = NaN(nCams(k),2,nMeas);
    for j = 1:nMeas
        points(:,:,j) = cell2mat(cellfun(@(p,row) p(row,:), ...
                                 imagePoints(camCombs(k,:)),....
                                 num2cell(combinations(index(j),camCombs(k,:))),...
                                 'UniformOutput',false).');
    end
    
    % calculate 3D-world-points 
    [worldPointsHat(index,:), errorVal(index)] = multiviewToWorld('minID', permute(points,[3 2 1]), cameraMatrices(camCombs(k,:)));
    
end

% only take points into account with an mean-squared-error of less then maxErrorVal
index = (errorVal <= maxErrorVal);
errorVal = errorVal(index);
worldPointsHat = worldPointsHat(index,:);
numberOfUsedViews = numberOfUsedViews(index);

% sort 3D-Points according to the error
[errorVal,indexOrder] = sort(errorVal);
worldPointsHat = worldPointsHat(indexOrder,:);
numberOfUsedViews = numberOfUsedViews(indexOrder);

end