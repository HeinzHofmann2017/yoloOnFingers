function [ imagePointsDist, pointValid ] = applyRadialDistortion(imagePointsUndist, intrinsicMat, distCoeff)
%APPLYRADIALDISTORTION function to apply radial distortion to undistortd points of an image
% 
%   Input:
%       imagePointsUndist:  Undistorted image-points as a [M x 2] array
%       intrinsicMat:       intrinsic camera matrix (upper-triangular [3x3])
%       distCoeff:          distortion coefficient of radial distortion as
%                           a 3-element array
%   Output:
%       imagePointsDist:    Distorted image-points as a [M x 2] array
%       pointValid:         logical M-element array that tells, if the
%                           distorted image-point is valid
% 

% check input parameters
M = size(imagePointsUndist,1);
if any(size(imagePointsUndist) ~= [M,2])
    error('imagePointsUndist must have size of Mx2')
end
if ( any(size(intrinsicMat) ~= [3,3]) || (intrinsicMat(3,1) ~= 0) )
    error('intrinsicMat must be an upper-triangular 3x3-Matrix')
end
if any((size(distCoeff) ~= [1,3]) & (size(distCoeff) ~= [3,1]))
    error('distCoeff must have size of 1x3 or 3x1')
end

% determine which points are in the valid region
pointValid = isPointInValidRegion(imagePointsUndist, 'u', intrinsicMat, distCoeff);

% normalized image-coordinates
imagePointsUndistNormed = [imagePointsUndist,ones(size(imagePointsUndist,1),1)]*inv(intrinsicMat).';

% squared Radius
r2 = imagePointsUndistNormed(:,1).^2 + imagePointsUndistNormed(:,2).^2;

% Apply radial-distortion
factor = ( 1 + distCoeff(1)*r2 + distCoeff(2)*r2.^2 + distCoeff(3)*r2.^3);
imagePointsDistNormed(:,1) = factor.*imagePointsUndistNormed(:,1);
imagePointsDistNormed(:,2) = factor.*imagePointsUndistNormed(:,2);
imagePointsDistNormed(:,3) = ones(size(factor));

% unnormalise points
imagePointsDist = imagePointsDistNormed*intrinsicMat.';
imagePointsDist = imagePointsDist(:,1:2);


end

