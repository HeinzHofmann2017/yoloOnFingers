%
%   Localize UV-Object in 3D
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
picPath = '../CollectFingertipData/tests_new/test_circle_left';

if inputPath(end) ~= '/'
    inputPath(end+1) = '/';
end
if picPath(end) ~= '/'
    picPath(end+1) = '/';
end

% geometric dimensions
nCams = 4;
nCamsToUse = 4;
ImageSize = [960, 1280];

% load camera parameters
file = sprintf('%sintrinsicParams.mat',inputPath);
load(file);
file = sprintf('%sextrinsicParams.mat',inputPath);
load(file);
file = sprintf('%scameraMatrices.mat',inputPath);
load(file);

%load positions of fingertips (UV-Object) in differernt views
allFingertips = {};
for cam = 1:nCams
    load(sprintf('%sCamera_%d/UV_Bin/fingers.mat',picPath,cam-1));
    allFingertips{cam} = fingers;
    clear fingers
end

%% Collect points form Images and estimate 3D-location

allWorldPointsHat = zeros(0,3);
nUsedPoints = [];
for i = 1:length(allFingertips{1})
    
    % collect points from different views and undistort them
    for cam = 1:nCams
        
        imagePoints{cam} = allFingertips{cam}{i}.centers;
                
        figure(1);
        subplot(2,2,cam)
        I = double(imread(sprintf('%sCamera_%d/UV_Bin/pic%d.png',picPath,cam-1,i-1)));
        imshow(I,[]); hold on;
        plot(imagePoints{cam}(:,1),imagePoints{cam}(:,2),'b*');
        
        [ imagePoints{cam}, pointValid ] = removeRadialDistortion(imagePoints{cam}, ...
                                                                  cameraParamsIntrinsic{cam}.IntrinsicMatrix.', ...
                                                                  cameraParamsIntrinsic{cam}.RadialDistortion );
        imagePoints{cam} = imagePoints{cam}(pointValid,:);
    end
    
    if any(cellfun(@isempty,imagePoints))
        continue;
    end
    % find all possible world-points
    [ worldPointsHat, numberOfUsedViews, error ] = extensiveWorldPointEstimation( imagePoints, cameraMatrices, nCamsToUse, 5^2 );
    
    % use nMax points with smallest error
    nMax = 2*factorial(nCams)/(factorial(nCamsToUse)*factorial(nCams-nCamsToUse));
    if size(worldPointsHat,1) > nMax
        worldPointsHat = worldPointsHat(1:nMax,:);
    end
    
    % Collect all world-points
    allWorldPointsHat(end+1:end+size(worldPointsHat,1),:) = worldPointsHat;
    nUsedPoints(end+1) = size(worldPointsHat,1);
    
    % plot found world-points
    for cam = 1:nCams
        [reproImagePoints,validImagePoints] = ...
                myWorldToImage(cameraParamsIntrinsic{cam},...
                               cameraParamsExtrinsic{cam}.RotationMatrix,...
                               cameraParamsExtrinsic{cam}.location,...
                               worldPointsHat);
        figure(1);
        subplot(2,2,cam)
        hold on;
        plot(reproImagePoints(:,1),reproImagePoints(:,2),'g*');
        hold off;
    end
    
    figure(2);
    plot3(worldPointsHat(:,1),worldPointsHat(:,2),worldPointsHat(:,3),'b*'); hold on;
    grid on;
    xlabel('x [mm]')
    ylabel('y [mm]')
    zlabel('z [mm]')
    axis equal;
    axis([0, 800, 0, 400, 0, 200]);
    view(0,90);
    
    pause(0.01);
end

%% Evaluate Results of estimation

% load circle-curves and match points to curves by calculating the error
load(sprintf('%scircleParams.mat',picPath));
nCirc = length(radius);
N = size(allWorldPointsHat,1);
error = NaN(N,nCirc);
for i = 1:nCirc
    dr = sqrt(sum((allWorldPointsHat.' - repmat(center(:,i),1,N)).^2,1))-radius(i);
    distance = normal(:,i).'*(allWorldPointsHat.' - repmat(center(:,i),1,N));
    error(:,i) = sqrt(dr.^2 + distance.^2).';
end
ind = ( repmat(min(error,[],2),1,nCirc) == error );
error = error(ind);

% show results
figure(2);
for i=1:nCirc
    plot3(center(1,i),center(2,i),center(3,i),'g*'); hold on;
    plot3(circlePoints(:,1,i),circlePoints(:,2,i),circlePoints(:,3,i),'g'); hold on;
end

figure(3)
histogram(error,0:0.1:ceil(max(error)/0.1)*0.1,'Normalization','probability');
xlabel('Error [mm]');
title(sprintf('Circles: m = %f, s = %f', mean(error), std(error)));
grid on;

figure(4)
hist(nUsedPoints);
xlabel('number of used points')

