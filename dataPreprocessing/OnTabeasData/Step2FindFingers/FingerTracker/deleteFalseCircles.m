function [ center, radius ] = deleteFalseCircles(centers, radii )
%Function to delete false circles in the binary-Image
% 
%   Input:
%       centers:         centers of the cirlces as [Nx2]-array
%       radii:           radii of the cirlces as [Nx1]-array
%       
%   Output:
%       center:         coordinates from the biggest valid circle 
%       radius:         radius from the biggest valid circle
% 
        nPoints = size(centers,1);
        %Delete impossible radii
        for i=nPoints:-1:1            
            if radii(i)<3 || radii(i)>20 %Points smaller than 3 are maybe LED's & Points larger than 20 are maybe light-artifacts TODO: maybe raise the number of 20, because the radius of a finger is higher, as soon as tabea has changed this point...
                centers(i,:)=[];
                radii(i,:)=[];
            end
        end
        %number of remaining Points:
        nPoints = size(centers,1);
        %store the biggest radius in temp:
        temp = 0;
        for i=nPoints:-1:1
            if radii(i) > temp
                temp = radii(i);
            end
        end
        %number of remaining Points
        nPoints = size(centers,1);
        %Delete all radii, which are smaller than temp
        for i=nPoints:-1:1
            if radii(i) < temp
                centers(i,:)=[];
                radii(i,:)=[];
            end
        end
        center = centers;
        radius = radii;

end