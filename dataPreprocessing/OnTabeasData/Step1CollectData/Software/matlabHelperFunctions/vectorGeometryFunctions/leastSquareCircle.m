function [ n, c, r, error, circPoints ] = leastSquareCircle( points )
%LEASTSQUARECIRCLE function that fits a circle into a cluster of 3D-points 
% 
%   Input:
%       points:     3D points as a [N x 3] array
%       
%   Output:
%       n:          normal-vector of the plane, in which the circle lies
%       c:          center of the circle
%       r:          radius of the circle
%       error:      distances of the points to the circle [N x 1]
%       circPoints: estimated circle points [361 x 3]
% 
%   Calculations according to:
%   https://meshlogic.github.io/posts/jupyter/curve-fitting/fitting-a-circle-to-cluster-of-3d-points/
%

% check input parameters
N = size(points,1);
if any(size(points) ~= [N,3])
    error('points must have size of Nx3')
end

% calculate least-square-plane
p = mean(points,1).';
A = points-repmat(p.',N,1);
% n = pinv(A)*ones(N,1);
% n = 1/sqrt(n.'*n)*n;
[~,~,V] = svd(A);
n = V(:,3);

% project point onto the plane
distance = n.'*(points.'- repmat(p,1,N));
projPoints = points.' - n*distance;

% Transform projected point to the x/y-plane
[phi,theta,~] = cart2sph(n(1),n(2),n(3));
theta = pi/2-theta;
R = [cos(-phi) -sin(-phi) 0 ;
     sin(-phi)  cos(-phi) 0 ;
            0          0  1 ];
R = [ cos(-theta) 0 sin(-theta) ;
               0  1          0  ;
     -sin(-theta) 0 cos(-theta) ] * R;
transfPoints = R*(projPoints - repmat(p,1,N));


% Calculate the center and the radius of the transformed circle
A = [transfPoints(1:2,:); ones(1,N)].';
b = sum(transfPoints(1:2,:).^2,1).';
x = pinv(A)*b;
cp = x(1:2)/2;
r = sqrt(x(3)+cp(1)^2+cp(2)^2);

% transform the center back to the plane 
c = R.'*[cp;0]+p;

% calculate the distances between the circle and the points
dr = sqrt(sum((projPoints - repmat(c,1,N)).^2,1))-r;
error = sqrt(dr.^2 + distance.^2).';

% calculate ideal circle points
t = (0:1:360)/360*2*pi;
circPoints = cp+r*[cos(t);sin(t)];
circPoints = [circPoints; zeros(1,length(t))];
circPoints = (R.'*circPoints+repmat(p,1,length(t))).';

end