function [ newBoardPoints, newboardSize ] = ...
                changeCheckerboardOrigin( x0, y0, boardPoints, boardSize )
%CHANGECHECKERBOARDORIGIN Rearanges points of a checkerboard to change the
%   origin of the checkerboard. if the new origin is not a corner, the
%   points that lie in the negative region are deleted.
% 
%   Inputs:
%       x0:             x-Koordinate of desired Origin
%       y0:             y-Koordinate of desired Origin
%       boardPoints:    boardPoints of checkerboard [N x 2] (N=nx*ny)
%       boardSize:      size of checkerboard [height, width]
%                       ny+1 = height, nx+1 = width
% 
%   Outputs:
%       newBoardPoints: boardPoints of checkerboard with new origin,
%                       [] if not successful
%       newboardSize:   size of new board, [0, 0] if not successful
%

% check input parameters
N = size(boardPoints,1);
if any(size(boardPoints) ~= [N,2])
    error('boardPoints must have size of Nx2')
end
if ( (boardSize(1)-1)*(boardSize(2)-1) ~= N )
    error('boardSize is not consistent with the size of boardPoints')
end

% check if selected origin is close enough to a detected point
[dist, indOrg] = min(sqrt(sum([boardPoints(:,1)-x0, boardPoints(:,2)-y0 ].^2,2))); 
maxDist = inf;
for j=1:length(boardPoints)-1
    for k=j+1:length(boardPoints)
        newDist = sqrt(sum((boardPoints(j,:)-boardPoints(k,:)).^2));
        if newDist < maxDist
            maxDist = newDist;
        end
    end
end

if ( dist < maxDist ) 
    % selected origin is a valid point -> reshape imagePoints
    
    % take the midpoint as (0,0)
    newBoardPoints = reshape(boardPoints,boardSize(1)-1,boardSize(2)-1,2);
    midPoint = boardSize/2;
    [indOrg(1),indOrg(2)] = ind2sub(boardSize-1,indOrg);
    indOrg = indOrg - midPoint;
    
    if( indOrg(1) <= 0 && indOrg(2) <= 0 )
        % origin is in upper left rectangle -> orientation already is correct
    elseif ( indOrg(1) > 0 && indOrg(2) <= 0 )
        % origin is in lower left rectangle -> rotate points 270°
        newBoardPoints = rot90(newBoardPoints,3);
        indOrg = fliplr(indOrg);
        indOrg(2) = -indOrg(2);
        midPoint = fliplr(midPoint);
    elseif ( indOrg(1) > 0 && indOrg(2) > 0 )
        % origin is in lower right rectangle -> rotate points 180°
        newBoardPoints = rot90(newBoardPoints,2);
        indOrg = -indOrg;        
    elseif ( indOrg(1) <= 0 && indOrg(2) > 0 )
        % origin is in upper right rectangle -> rotate points 90°
        newBoardPoints = rot90(newBoardPoints,1);
        indOrg = fliplr(indOrg);
        indOrg(1) = -indOrg(1);
        midPoint = fliplr(midPoint);
    end
    indOrg = indOrg + midPoint;
    newBoardPoints = newBoardPoints(indOrg(1):end, indOrg(2):end, :);
    newboardSize(1) = size(newBoardPoints,1)+1;
    newboardSize(2) = size(newBoardPoints,2)+1;
    newBoardPoints = reshape(newBoardPoints,[],2);

else    
    % selected origin is NOT a valid point -> return empty arrays
    newBoardPoints = [];
    newboardSize = [0,0];
end

end


