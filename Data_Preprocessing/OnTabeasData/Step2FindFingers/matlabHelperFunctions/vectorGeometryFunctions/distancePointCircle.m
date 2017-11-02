function [ dist, circle ] = distancePointCircle( points, center, normal, radius )
%LEASTSQUARECIRCLE function that calculates the distance between the points 
%   and the circle
% 
%   Input:
%       points:     3D points as a [N x 3] array
%       center:     center of the circle [3x1]
%       normal:     normal vector of the circle [3x1]
%       radius:     radius of the circle
%       
%   Output:
%       dist:       distance between the circle and the points [N x 1]
%       circle:     circle points [361 x 3]
% 
%

% check input parameters
N = size(points,1);
if any(size(points) ~= [N,3])
    error('points must have size of [Nx3]')
end
if any(size(center) ~= [3,1])
    error('center must have size of [3x1]')
end
if any(size(normal) ~= [3,1])
    error('normal must have size of [3x1]')
end
if any(size(radius) ~= [1,1])
    error('radius must have size of [1x1]')
end

% calculate circle Points
t = 0:1:360;
r = cross(center,normal);
r = radius/sqrt(r.'*r)*r;
rr = cross(normal,r);
rr = radius/sqrt(rr.'*rr)*rr;
circle = repmat(center,1,length(t)) + r*cosd(t) + rr*sind(t);

% calculate distance between the points and the circle
points = points.';
diff = (points-repmat(center,1,N));
delta = sqrt(sum(diff.^2,1));
deltaPlane = normal.'*diff;
deltaRadius = sqrt( delta.^2 - deltaPlane.^2) - radius;
dist = sqrt(deltaRadius.^2+deltaPlane.^2);


end