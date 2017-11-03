function [ worldPointsHat, errorVal ] = multiviewToWorld(method, imagePointsUndist, cameraMatrices )
%MULTIVIEWTOWORLD function that estimates the 3D-world-point from different
%       camera-views
% 
%   Input:
%       method:             method to use for calculating the 3D-wold-point
%                           'minWD': Minimize distanzes in the world
%                           'minID': Minimize distanzes on the image
%       imagePointsUndist:  undistorted image points as a [N x 2 x V] array
%                           (V = number of views )
%       cameraMatrices:     V-element cell-array of camera matrices ([3 x 4])
%       
%   Output:
%       worldPointsHat:     estimation of world-point [N x 3]
%       errorVal:           value of the error-surface (mean-squared-distance) [N x 1]
% 

% check input parameters
if ( ~strcmp(method,'minWD') && ~strcmp(method,'minID'))
    error('method must be either ''minWD'' or ''minID''')
end
N = size(imagePointsUndist,1);
V = size(imagePointsUndist,3);
if any(size(imagePointsUndist) ~= [N,2,V])
    error('imagePointsUndist must have size of Nx2xV')
end
if any((size(cameraMatrices) ~= [1,V]) & (size(cameraMatrices) ~= [V,1]))
    error('cameraMatrices must have size of 1xV or Vx1')
else
    for i=1:V
        if any(size(cameraMatrices{i}) ~= [3,4])
            error('Elements of cameraMatrices must have size of 3x4')
        end
    end
end



%% Minimize distanzes in the world

% Calculate camera-centers and ray-vectors
C = zeros(3,V);     % camera centers
m = zeros(3,V);     % principal-axis
r = zeros(3,V,N);   % ray-vectors
for cam = 1:V
    P = cameraMatrices{cam};
    M = P(:,1:3);
    m(:,cam) = M(3,:).'/sqrt(M(3,:)*M(3,:).');
    C(:,cam) = -inv(M)*P(:,4);
    D = inv(M)*[imagePointsUndist(:,:,cam),ones(N,1)].';
    r(:,cam,:) = D;
end

worldPointsHat = zeros(N,3);
errorVal = NaN(N,1);
for i=1:N
    % determine world point (least-square)
    A = [ V*eye(3),    -r(:,:,i);
         -r(:,:,i).',   (r(:,:,i).'*r(:,:,i)).*eye(V) ];
    b = [sum(C,2); -diag(C.'*r(:,:,i))];
    phi = inv(A)*b;
    worldPointsHat(i,:) = phi(1:3);

    % determine error-surface value
    X = C + repmat(phi(4:end).',3,1).*r(:,:,i);
    d = X-repmat(phi(1:3),1,V);
    errorVal(i) = 1/V*sum(diag(d.'*d));
end

%% Minimize distanzes on the image
% This is done by minimizing the distances in the world, but weighting the 
% different camera-views inversely proportional to their distances to the world-point 
if strcmp(method,'minID')    

    for i=1:N
        % weight the cameras-view according to their distance to the estimated world-point
        dist = diag(m.'*(repmat(worldPointsHat(i,:).',1,V)-C)).';
        w = 1./(dist);    

        % determine world point (weighted least-square)
        A = [ sum(w)*eye(3),  -r(:,:,i).*repmat(w,3,1);
                -r(:,:,i).',   (r(:,:,i).'*r(:,:,i)).*eye(V) ];
        b = [sum(C.*repmat(w,3,1),2); -diag(C.'*r(:,:,i))];
        phi = inv(A)*b;
        worldPointsHat(i,:) = phi(1:3);

        % determine error-surface value
        errorVal(i) = 0;
        for cam = 1:V
            x = imagePointsUndist(i,:,cam).';
            x_hat = cameraMatrices{cam}*[worldPointsHat(i,:),1].';
            x_hat = x_hat(1:2)/x_hat(3);
            errorVal(i) = errorVal(i) + (x-x_hat).'*(x-x_hat);
        end
        errorVal(i) = 1/V*errorVal(i);

    end
end
    
end