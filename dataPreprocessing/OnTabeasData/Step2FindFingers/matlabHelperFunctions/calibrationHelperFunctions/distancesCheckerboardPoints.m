function [ dist ] = distancesCheckerboardPoints( boardPoints, boardSize )
%DISTANCESCHECKERBOARDPOINTS calculates the distances between 
%   checkerboard-Points
% 
%   Input:  
%       boardPoints:    boardPoints of checkerboard 
%                       [N x 2] (N=nx*ny) or [ny x nx x 2]
%       boardSize:      size of checkerboard [height, width]
%                       ny+1 = height, nx+1 = width
% 
%   Outputs:
%       dist:           distances between checkerboard-Points [M x 1] 
%                       ( M = (nx-1)*(ny-1) )
% 

% check input parameters
if ( size(boardPoints,3) == 1 )
    N = size(boardPoints,1);
    if any(size(boardPoints) ~= [N,2])
        error('boardPoints must have size of [N x 2] or [ny x nx x 2]')
    end
    if ( (boardSize(1)-1)*(boardSize(2)-1) ~= N )
        error('boardSize is not consistent with the size of boardPoints')
    end
elseif( size(boardPoints,3) == 2 )
    nx = size(boardPoints,1);
    ny = size(boardPoints,2);
    if any(size(boardPoints) ~= [nx,ny,2])
        error('boardPoints must have size of [N x 2] or [ny x nx x 2]')
    end
    if ( (boardSize(1)-1)*(boardSize(2)-1) ~= nx*ny )
        error('boardSize is not consistent with the size of boardPoints')
    end
else
    error('boardPoints must have size of [N x 2] or [ny x nx x 2]')
end


% calculate distance of checkerboard-points
if ( size(boardPoints,3) == 1 )
    boardPoints = reshape(boardPoints,boardSize(1)-1,boardSize(2)-1,2);
end

dist = reshape(diff(boardPoints),[],2);
boardPoints = permute(boardPoints, [2,1,3]);
dist = [ dist; reshape(diff(boardPoints),[],2) ];
dist = sqrt(sum(dist.^2,2));

end

