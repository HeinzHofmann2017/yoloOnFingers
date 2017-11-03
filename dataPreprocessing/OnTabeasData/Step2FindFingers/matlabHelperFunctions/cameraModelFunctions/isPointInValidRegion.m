function [ pointValid ] = isPointInValidRegion( imagePoints, pointType, intrinsicMat, distCoeff)
%ISPOINTINVALIDREGION function that determines if a distorted or undistorted image point is in 
%   the valid region of the image, where the camera model can be applied
% 
%   Input:
%       imagePoints:        image-points as a [M x 2] array
%       pointType:          type of image point:
%                           'd' : distorted image point
%                           'u' : undistorted image point
%       intrinsicMat:       intrinsic camera matrix (upper-triangular [3x3])
%       distCoeff:          distortion coefficient of radial distortion as
%                           a 3-element array
%   Output:
%       pointValid:         logical M-element array that tells, if the
%                           distorted image-point lies within the valid
%                           region
% 

% check input parameters
M = size(imagePoints,1);
if any(size(imagePoints) ~= [M,2])
    error('imagePoints must have size of Mx2')
end
if ( pointType ~= 'd' && pointType ~= 'u' )
    error('pointType must be either ''d'' or ''u''')
end
if ( any(size(intrinsicMat) ~= [3,3]) || (intrinsicMat(3,1) ~= 0) )
    error('intrinsicMat must be an upper-triangular 3x3-Matrix')
end
if any((size(distCoeff) ~= [1,3]) & (size(distCoeff) ~= [3,1]))
    error('distCoeff must have size of 1x3 or 3x1')
end

% get radius of valid region
rValid = getValidRadius( pointType, distCoeff);

% normalized image-coordinates
imagePointsNormed = [imagePoints,ones(size(imagePoints,1),1)]*inv(intrinsicMat).';

% squared Radius
r2 = imagePointsNormed(:,1).^2 + imagePointsNormed(:,2).^2;

% point is valid if the normalized radius is smaler then rValid
pointValid = (r2 < rValid^2);

end

