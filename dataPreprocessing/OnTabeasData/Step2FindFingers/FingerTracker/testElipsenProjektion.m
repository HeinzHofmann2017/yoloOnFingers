%
%   Test Ellipsenprojektion
%
%   tmendez, 12.06.2017
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


clear all;
close all;
clc;

addpath(genpath('../matlabHelperFunctions'));

%% Parameters

% paths 
inputPath = '../CameraCalibration/cameraParameters';

if inputPath(end) ~= '/'
    inputPath(end+1) = '/';
end

% geometric dimensions
N = 30^2;
nCams = 4;
ImageSize = [960, 1280];

% load camera parameters
file = sprintf('%sintrinsicParams.mat',inputPath);
load(file);
file = sprintf('%sextrinsicParams.mat',inputPath);
load(file);
file = sprintf('%scameraMatrices.mat',inputPath);
load(file);

%% show setup

dim = 3;
mu = [400;200;50];
Phi = rotx(360*rand)*roty(360*rand)*rotz(360*rand);
Lambda = diag([100^2; 50^2; 20^2]);%1000*diag(sort(rand(dim,1),'descend'))
sigma = Phi*Lambda*Phi.';
p = sqrt(diag(Lambda));
[THETA,PHI] = ndgrid(linspace(0,pi,sqrt(N)),linspace(0,2*pi,sqrt(N)));
x = p(1)*sin(THETA).*cos(PHI);
y = p(2)*sin(THETA).*sin(PHI);
z = p(3)*cos(THETA);
samples = Phi*[x(:).';y(:).';z(:).']+repmat(mu,1,N);
% samples = mvnrnd(mu,sigma,N).';

figure(1);
for cam = 1:nCams
    
        workspace = [0, 770; 0, 310; 0, 200];
        plot3([workspace(1,1) workspace(1,2) workspace(1,2) workspace(1,1) workspace(1,1) workspace(1,1) workspace(1,2) workspace(1,2) workspace(1,2) workspace(1,2) workspace(1,2) workspace(1,2) workspace(1,1) workspace(1,1) workspace(1,1) workspace(1,1)],...
              [workspace(2,1) workspace(2,1) workspace(2,2) workspace(2,2) workspace(2,1) workspace(2,1) workspace(2,1) workspace(2,1) workspace(2,1) workspace(2,2) workspace(2,2) workspace(2,2) workspace(2,2) workspace(2,2) workspace(2,2) workspace(2,1)],...
              [workspace(3,1) workspace(3,1) workspace(3,1) workspace(3,1) workspace(3,1) workspace(3,2) workspace(3,2) workspace(3,1) workspace(3,2) workspace(3,2) workspace(3,1) workspace(3,2) workspace(3,2) workspace(3,1) workspace(3,2) workspace(3,2)],...
              '-*','Color',[0.49,0.18,0.56]); hold on;
        plotCamera('Location', cameraParamsExtrinsic{cam}.location,...
                   'Orientation', cameraParamsExtrinsic{cam}.RotationMatrix, ...
                   'Label', sprintf('cam %d',cam-1) ,'AxesVisible',true,...
                   'Color',[0,0,1],'Size',10); hold on;
        title('Camera Positions');
        grid on;
        axis equal;
        axis([-100 900 -150 450 -200 300]);
        xlabel('x [mm]');
        ylabel('y [mm]');
        zlabel('z [mm]');
        
end

plot3(samples(1,:),samples(2,:),samples(3,:),'*'); hold off;


%% calculate projection of ellipsoid



for cam = 1:nCams
    
    P = cameraMatrices{cam};
    M = P(:,1:3);
    C = -inv(M)*P(:,4);
    
    % Berechnung der 2D-Ellipse aus sicht der Kamera
    r = mu-C;
    n = Phi.'*r;
    [phi,theta,~] = cart2sph(n(1),n(2),n(3));
    theta = pi/2-theta;
    D = rotz(phi/pi*180)*roty(theta/pi*180);
    T = Phi*D;
    sigmaDash = T.'*sigma*T;
    sigmaDash2D = sigmaDash(1:2,1:2);
    [PhiDash2D, LambdaDash2D] = eig(sigmaDash2D,'vector');
    [LambdaDash2D,ind] = sort(LambdaDash2D,'descend');
    PhiDash2D = PhiDash2D(:,ind);
    if (det(PhiDash2D) < 0)
        PhiDash2D(:,1) = -PhiDash2D(:,1);
    end
    a = T*[sqrt(LambdaDash2D(1))*PhiDash2D(:,1);0];
    b = T*[sqrt(LambdaDash2D(2))*PhiDash2D(:,2);0];
    A = mu + a;
    B = mu + b;
    e = [a/norm(a),b/norm(b),r/norm(r)];
    e.'*e
    det(e) 
%     e1(cam) = sum(sum(abs(e.'*e-eye(3))));
%     e2(cam) = det(e);
    
    % modifikation der Kamera-Matrize
%     K = cameraParamsIntrinsic{cam}.IntrinsicMatrix.';
%     K(1,1) = mean([K(1,1),K(2,2)]);
%     K(2,2) = mean([K(1,1),K(2,2)]);
%     P = K*[cameraParamsExtrinsic{cam}.RotationMatrix,cameraParamsExtrinsic{cam}.TranslationVector];

    % Transformation der 2D-Ellipse in die Bildebene    
    aPic = P*[A;1];
    aPic = aPic(1:2)/aPic(3);
    bPic = P*[B;1];
    bPic = bPic(1:2)/bPic(3);
    m = P*[mu;1];
    m = m(1:2)/m(3);
    vb1 = aPic-m;
    vb2 = bPic-m;
    PhiPic = [vb1/norm(vb1),vb2/norm(vb2)];
    PhiPic.'*PhiPic

    % Transformierte Ellipesen-Punkte
    samplesPic = P*[samples;ones(1,N)];
    samplesPic = samplesPic(1:2,:)./repmat(samplesPic(3,:),2,1);
    figure(2)
    subplot(2,2,cam)
    I = double(imread(sprintf('BG_Cam%d.png',cam-1)));
    imshow(I,[]); hold on
    plot(samplesPic(1,:),samplesPic(2,:),'*'); hold on;
    plot([m(1),aPic(1)],[m(2),aPic(2)],'-*'); hold on;
    plot([m(1),bPic(1)],[m(2),bPic(2)],'-*'); hold on;
    title(sprintf('camera %d',cam-1));
    hold off;
end



























