function [ rValid ] = getValidRadius( type, distCoeff)
%GETVALIDRADIUS function that calculates the radius of the valid region of
%   normalized image points
% 
%   Input:
%       type:       type of image point:
%                   'd' : distorted image point
%                   'u' : undistorted image point
%       distCoeff:  distortion coefficient of radial distortion as
%                   a 3-element array
%   Output:
%       rValid:     radius of valid region of normalized image points
% 

% check input parameters
if ( type ~= 'd' && type ~= 'u' )
    error('type must be either ''d'' or ''u''')
end
if any((size(distCoeff) ~= [1,3]) & (size(distCoeff) ~= [3,1]))
    error('distCoeff must have size of 1x3 or 3x1')
end

% find Radius of undistorted valid region
rValid = roots([7*distCoeff(3), 0, 5*distCoeff(2), 0, 3*distCoeff(1), 0, 1]);
rValid = rValid(abs(imag(rValid))==0);  % rValid must be real
rValid = rValid(rValid>0);  % rValid must be grater than 0
rValid = min(rValid);   % use smalest valid region

% find Radius of distorted valid region
if type == 'd'
    rValid = rValid*(1 + distCoeff(1)*rValid^2 + distCoeff(2)*rValid^4 + distCoeff(3)*rValid^6);
end

end
