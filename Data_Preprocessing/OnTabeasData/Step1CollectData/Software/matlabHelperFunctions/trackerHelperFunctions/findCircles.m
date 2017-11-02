function [ centers, radii ] = findCircles( Im )
%FINDCIRCLES function to find circles in binary-Image
% 
%   Input:
%       Im:         binary image, of the fingers
%       
%   Output:
%       centers:    centrers of the cirlces as [Nx2]-array
%       radii:      radii of the cirlces as [Nx1]-array
% 

[N,M] = size(Im);
se = strel('rectangle',[3,3]);

centers = zeros(0,2);

% find number of conected component
conComp = bwconncomp(Im);
ncc = conComp.NumObjects;

% find center of each conected component, by eroding until it disapears
i=1;
while i <= ncc
    % generate image with a single connected component
    ImTemp = false(N,M);
    ImTemp(conComp.PixelIdxList{i}) = true;
    
    % erode it, until the number of connected component changes
    conCompTemp = bwconncomp(ImTemp);
    nccTemp = conCompTemp.NumObjects;
    while nccTemp == 1
        ImTempOld = ImTemp;
        ImTemp = imerode(ImTemp,se);
        conCompTemp = bwconncomp(ImTemp);
        nccTemp = conCompTemp.NumObjects;
    end
    
    if nccTemp > 1
        % connected component has been split up
        % -> add new connected components to list
        for j=1:nccTemp
            conComp.PixelIdxList{end+1} = conCompTemp.PixelIdxList{j};
            conComp.NumObjects = conComp.NumObjects + 1;
        end
        ncc = conComp.NumObjects;
    elseif nccTemp < 1
        % connected component disapeard 
        % -> save its location
        conCompTemp = bwconncomp(ImTempOld);
        nccTemp = conCompTemp.NumObjects;
        if nccTemp ~= 1
            error('Number of connected componets should be 1.')
        end
        [y,x] = ind2sub([N,M],conCompTemp.PixelIdxList{1});
        centers(end+1,:) = [mean(x),mean(y)];
    end
    i = i + 1;
end


% find the radius as the minimal distance between the center and the boarder
boarder = find(Im - imerode(Im,se));
[y,x] = ind2sub([N,M],boarder);
radii = zeros(size(centers,1),1);
for i=1:size(centers,1)
    radii(i) = min(sqrt((x-centers(i,1)).^2 + (y-centers(i,2)).^2));
end


end