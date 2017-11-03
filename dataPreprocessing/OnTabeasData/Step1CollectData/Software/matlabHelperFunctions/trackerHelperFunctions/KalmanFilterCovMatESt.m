%
%   Estimation of Kalman-Covariance-Matrix 
%   of the model error
%
%   tmendez, 09.07.2017
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all;
% clc;

x0 = randn(6,1);

% True modell
sigma = 2;
T = 0.1;
x_true = x0;


% Kalman modell
% A = [1 T 0 0 ;
%      0 1 0 0 ;
%      0 0 1 T ;
%      0 0 0 1 ];
A = [1 T 0 0 0 0;
     0 1 0 0 0 0;
     0 0 1 T 0 0;
     0 0 0 1 0 0;
     0 0 0 0 1 T;
     0 0 0 0 0 1];
x_hat = x0;
Qw = zeros(size(A));
 
for i=1:100000
    x_hat(:,end+1) = A*x_true(:,end);
    temp = x_true(:,end);
    x_true_new = zeros(size(temp));
    x_true_new([2,4,6]) = temp([2,4,6]) + sigma*randn(3,1);
    x_true_new([1,3,5]) = temp([1,3,5]) + T*x_true_new([2,4,6]);
    x_true(:,end+1) = x_true_new;
    w = x_true(:,end)-x_hat(:,end);
    Qw = Qw + w*w';
end
Qw = Qw/size(x_true,2)
 
plot3(x_true(1,:),x_true(3,:),x_true(5,:),'m'); hold on;
% quiver3(x_true(1,:),x_true(3,:),x_true(5,:),x_true(2,:),x_true(4,:),x_true(6,:),0.375,'Color',[1,0,0]); hold on;
plot3(x_hat(1,:),x_hat(3,:),x_hat(5,:),'c'); hold on;
% quiver3(x_hat(1,:),x_hat(3,:),x_hat(5,:),x_hat(2,:),x_hat(4,:),x_hat(6,:),0.375,'Color',[0,0,1]); hold on;
grid on;
hold off;
% view(0,90)
 
 
 