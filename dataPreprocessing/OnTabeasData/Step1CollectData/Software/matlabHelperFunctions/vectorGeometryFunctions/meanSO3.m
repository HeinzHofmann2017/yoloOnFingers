function [ A_hat ] = meanSO3( A )
%MEANSO3 calculates the mean of SO(3) matrices by averaging lie-algebra
% linearizations of the matrices
% 
%   Input:
%       A:      SO(3) matrices that need to be averaged [3 x 3 x N]
% 
%   Output:
%       A_hat:  average of the SO(3) matrices
% 

if ( size(A,1)~=3 || size(A,2)~=3 )
    error('Input Matrix needs to be of the size [3 x 3 x N]' );
end

N = size(A,3);
B = zeros(3,3,N);
C = zeros(3,3,N);
w = zeros(3,N);

A_hat = A(:,:,1);
rot_Axis_old = rotationMatrixToVector(A_hat).';
diff = 1;
while diff > 10^-20
    A_hat_inv = A_hat^(-1);

    for i=1:N
        B(:,:,i) = A(:,:,i)*A_hat_inv;
        w(:,i) = rotationMatrixToVector(B(:,:,i)).';
    end
    w_hat = mean(w,2);
    B_hat = rotationVectorToMatrix(w_hat);
    A_hat = B_hat*A_hat;
    
    rot_Axis_new = rotationMatrixToVector(A_hat).';
    diff = sqrt(sum((rot_Axis_new-rot_Axis_old).^2));
    rot_Axis_old = rot_Axis_new;
end
    
end

