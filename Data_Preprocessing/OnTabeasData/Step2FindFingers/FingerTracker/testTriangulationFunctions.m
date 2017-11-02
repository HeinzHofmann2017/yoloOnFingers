%
%   CrossValidation of 3 different Triangulation functions:
%       - Minimizing cross-product: cross(x, PX)
%       - Minimizing Distances in 3D: di = X-X_hat
%       - Minimizing Distances in 3D: di = x-x_hat
%
%   tmendez, 19.06.2017
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


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

% Noise-Level (std in pixel)
noiseLevel = 10;

% Process
processCams = [1,2,3,4];

% geometric dimensions
nCams = 4;
ImageSize = [960, 1280];

% load camera parameters
file = sprintf('%sintrinsicParams.mat',inputPath);
load(file);    
file = sprintf('%sextrinsicParams.mat',inputPath);
load(file);    
file = sprintf('%scameraMatrices.mat',inputPath);
load(file);    


%% Create ideal image-points and add noise to them
Np = 10000;
worldPoints = [800*rand(Np,1), 400*rand(Np,1), 200*rand(Np,1)];
% worldPoints = [472.1, 125.8, 13.7];

imagePoints = NaN(size(worldPoints,1),2,nCams);
validImagePoints = false(size(worldPoints,1),nCams);
for cam = processCams
    [imagePoints(:,:,cam),validImagePoints(:,cam)] = ...
                myWorldToImage(cameraParamsIntrinsic{cam},...
                               cameraParamsExtrinsic{cam}.RotationMatrix,...
                               cameraParamsExtrinsic{cam}.location,...
                               worldPoints);
    imagePoints(:,:,cam) = imagePoints(:,:,cam) + noiseLevel*randn(size(imagePoints(:,:,cam)));
                           
    I = imread(sprintf('../CameraCalibration/cameraParameters/validRegion/cam%d.jpg',cam-1));
    fig = figure(1); set(fig,'units','normalized','outerposition',[0 0 1 1]);
    subplot(2,2,cam)
    imshow(I,[]); hold on;
    plot([0 ImageSize(2) ImageSize(2) 0 0],[0 0 ImageSize(1) ImageSize(1) 0]); hold on;
    plot(imagePoints(validImagePoints(:,cam),1,cam),imagePoints(validImagePoints(:,cam),2,cam),'g*'); hold on;
    title(sprintf('Camera %d',cam-1));
end

%% Calculate 3D Position of Objects 

vSet = viewSet;
for cam = processCams

    % undistort image points
    [ imagePointsUndist, ~ ] = removeRadialDistortion(imagePoints(validImagePoints(:,cam),:,cam), ...
                                                      cameraParamsIntrinsic{cam}.IntrinsicMatrix.', ...
                                                      cameraParamsIntrinsic{cam}.RadialDistortion);
    ind = isnan(imagePointsUndist(:,1));
    imagePointsUndist = imagePointsUndist(~ind,:);
    validImagePoints(validImagePoints(:,cam),cam) = validImagePoints(validImagePoints(:,cam),cam) & (~ind);
    
    % add points with small reprojection error to the viewSet
    vSet = addView(vSet, cam,'Points', imagePointsUndist,...
                             'Orientation', cameraParamsExtrinsic{cam}.RotationMatrix,...
                             'Location',cameraParamsExtrinsic{cam}.location.');
end

% add connections of points in different views
for camA = processCams
    for camB = processCams(find(camA==processCams)+1:end)
        indexPairs = cumsum(validImagePoints(:,[camA,camB]),1);
        indexPairs = indexPairs( (validImagePoints(:,camA) & validImagePoints(:,camB)) ,:);
        vSet = addConnection(vSet,camA,camB,...
                             'Matches',indexPairs);
    end
end
pointTracks = findTracks(vSet);
N = length(pointTracks);

% Calculate 3D-Location of object by minimizing the cross-product cross(x, PX)
[worldPointsHatMinCP,] = myTriangulateMultiview(pointTracks,...
                                                poses(vSet),...
                                                {cameraParamsIntrinsic{processCams}});

% Calculate 3D-Location of object by minimizing Distances in 3D
worldPointsHatMin3D = NaN(N,3);
for i=1:N
    [worldPointsHatMin3D(i,:),~] = multiviewToWorld('minWD', ...
                                                    permute(pointTracks(i).Points,[3,2,1]),...
                                                    cameraMatrices(pointTracks(i).ViewIds) );
end

% Calculate 3D-Location of object by minimizing Distances in 2D
worldPointsHatMin2D = NaN(N,3);
for i=1:N
    [worldPointsHatMin2D(i,:),~] = multiviewToWorld('minID', ...
                                                    permute(pointTracks(i).Points,[3,2,1]),...
                                                    cameraMatrices(pointTracks(i).ViewIds) );
end

% order estimated world-points according to generated world-points
indexOrder = NaN(N,1);
for i=1:N
    cam = pointTracks(i).ViewIds(1);
    [~, ind] = ismember(pointTracks(i).Points(1,:),vSet.Views.Points{vSet.Views.ViewId == cam},'rows');
    indexOrder(i) = find(cumsum(validImagePoints(:,cam)) == ind,1);
end
[~, indexOrder] = sort(indexOrder);
worldPointsHatMinCP = worldPointsHatMinCP(indexOrder,:);
worldPointsHatMin3D = worldPointsHatMin3D(indexOrder,:);
worldPointsHatMin2D = worldPointsHatMin2D(indexOrder,:);

%% Evaluate accuracy of the different methods

% calculate estimation error
ind = sum(validImagePoints,2,'omitnan')>=2;
errorValMinCP = sqrt(sum((worldPointsHatMinCP-worldPoints(ind,:)).^2,2));
errorValMin3D = sqrt(sum((worldPointsHatMin3D-worldPoints(ind,:)).^2,2));
errorValMin2D = sqrt(sum((worldPointsHatMin2D-worldPoints(ind,:)).^2,2));

fig = figure(2); set(fig,'units','normalized','outerposition',[0 0 1 1]);
subplot(1,3,1);
histogram(errorValMinCP,'Normalization','probability','FaceColor',[0,0.45,0.75]);
xlabel('Error [mm]');
title(sprintf('MinCP: m = %f, s = %f', mean(errorValMinCP), std(errorValMinCP)));
grid on;
subplot(1,3,2);
histogram(errorValMin3D,'Normalization','probability','FaceColor',[0.85,0.33,0.1]);
xlabel('Error [mm]')
title(sprintf('Min3D: m = %f, s = %f', mean(errorValMin3D), std(errorValMin3D)));
grid on;
subplot(1,3,3);
histogram(errorValMin2D,'Normalization','probability','FaceColor',[0.93,0.69,0.13]);
xlabel('Error [mm]')
title(sprintf('Min2D: m = %f, s = %f', mean(errorValMin2D), std(errorValMin2D)));
grid on;

fig = figure(3); set(fig,'units','normalized','outerposition',[0 0 1 1]);
plot3(worldPoints(:,1),worldPoints(:,2),worldPoints(:,3),'*','Color',[0.47,0.67,0.19]); hold on;
plot3(worldPointsHatMinCP(:,1),worldPointsHatMinCP(:,2),worldPointsHatMinCP(:,3),'*','Color',[0,0.45,0.75]); hold on;
plot3(worldPointsHatMin3D(:,1),worldPointsHatMin3D(:,2),worldPointsHatMin3D(:,3),'*','Color',[0.85,0.33,0.1]); hold on;
plot3(worldPointsHatMin2D(:,1),worldPointsHatMin2D(:,2),worldPointsHatMin2D(:,3),'*','Color',[0.93,0.69,0.13]); hold off;
xlabel('x')
ylabel('y')
zlabel('z')
axis equal;
axis([0, 800, 0, 400, 0, 200]);
grid on
legend('exact','MinCP','Min3D','Min2D');

