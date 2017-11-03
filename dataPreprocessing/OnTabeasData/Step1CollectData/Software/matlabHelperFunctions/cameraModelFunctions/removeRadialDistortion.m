function [ imagePointsUndist, pointValid ] = removeRadialDistortion(imagePointsDist, intrinsicMat, distCoeff)
%REMOVERADIALDISTORTION function to remove radial distortion of distortd image-points
% 
%   Input:
%       imagePointsDist:    Distorted image-points as a [M x 2] array
%       intrinsicMat:       intrinsic camera matrix (upper-triangular [3x3])
%       distCoeff:          distortion coefficient of radial distortion as
%                           a 3-element array
%   Output:
%       imagePointsUndist:  Undistorted image-points as a [M x 2] array.
%                           For points that could not be determined uniquely, NaN is returned
%       pointValid:         logical M-element array that tells, if the
%                           undistorted image-point is valid
% 


% check input parameters
M = size(imagePointsDist,1);
if any(size(imagePointsDist) ~= [M,2])
    error('imagePointsUndist must have size of Mx2')
end
if ( any(size(intrinsicMat) ~= [3,3]) || (intrinsicMat(3,1) ~= 0) )
    error('intrinsicMat must be an upper-triangular 3x3-Matrix')
end
if any((size(distCoeff) ~= [1,3]) & (size(distCoeff) ~= [3,1]))
    error('distCoeff must have size of 1x3 or 3x1')
end

% determine which points are in the valid region
pointValid = isPointInValidRegion(imagePointsDist, 'd', intrinsicMat, distCoeff);

% normalized image-coordinates
imagePointsdistNormed = [imagePointsDist,ones(size(imagePointsDist,1),1)]*inv(intrinsicMat).';

% calculate undistorted normed points
v = imagePointsdistNormed(:,1)./imagePointsdistNormed(:,2);
vv = (v.^2+1);
N = size(vv,1);
p = [distCoeff(3)*vv.^3, zeros(N,1), distCoeff(2)*vv.^2, zeros(N,1), distCoeff(1)*vv, zeros(N,1), ones(N,1), -imagePointsdistNormed(:,2)];
imagePointsUndistNormed = ones(N,3);
rValid = getValidRadius( 'u', distCoeff);
for j = 1:N
    % calculate x- and y-coordinate
    ytemp = roots(p(j,:));
    ytemp = ytemp(imag(ytemp)==0);  % ytemp must be real
    xtemp = v(j)*ytemp;
    rtemp = sqrt(xtemp.^2+ytemp.^2);
    ind = rtemp < rValid;
    if sum(ind) == 1
        % assign valid coordinates
        imagePointsUndistNormed(j,1:2) = [xtemp(ind), ytemp(ind)];
    else
        imagePointsUndistNormed(j,:) = NaN;
    end
end

% unnormalise points
imagePointsUndist = imagePointsUndistNormed*intrinsicMat.';
imagePointsUndist = imagePointsUndist(:,1:2);

end

