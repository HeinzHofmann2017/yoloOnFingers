function [ cRows, cCols ] = findClusters( connMat )
%FINDCLUSTERS function that finds clusters in a connectivity-Matrix
%   Every column will be added to a cluster BUT NOT necessarily every row
% 
%   Input:
%       connMat:    matrix that shows the connectivity of rows and columns
%                   [M x N]-Matrix
%       
%   Output:
%       cRows:      [M x nClusters]-logical matrix that shows, for
%                   every found cluster the appropriate rows
%       cCols:      [N x nClusters]-logical matrix that shows, for
%                   every found cluster the appropriate columns
% 

% check input parameters
if length(size(connMat)) ~= 2
    error('connectivityMatrix must be of size [MxN]')
end

% find clusters
[nRow,nCol] = size(connMat);
cRows = false(nRow,0);
cCols = false(nCol,0);
usedColumns = false(1,nCol);
while any(~usedColumns)

    % find first unused column
    clusterCols = ((1:nCol) == find(usedColumns == 0,1));
    clusterRows = false(nRow,1);

    % find connected rows and colums (via true-entries in connectivityMatrix)
    changed = 1;
    while changed
        oldClusterRows = clusterRows;
        oldClusterCols = clusterCols;
        clusterRows = any(connMat(:,clusterCols),2);
        clusterCols = clusterCols | any(connMat(clusterRows,:),1);
        changed = (any(oldClusterRows ~= clusterRows) || any(oldClusterCols ~= clusterCols));
    end
    usedColumns = usedColumns | clusterCols;
    
    % add found cluster to cRows and cCols
    cRows(:,end+1) = clusterRows;
    cCols(:,end+1) = clusterCols;
end

end