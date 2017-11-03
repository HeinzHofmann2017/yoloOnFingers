%
%   Script for determining the accuracy of the
%   approximation-polynomial that is used
%   for the correction of lens distortions
%
%   tmendez, 25.07.2017
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


clear all;
close all;
clc;

addpath(genpath('../matlabHelperFunctions'));

%% Parameters

% Process Settings
nCams = 4;
ImageSize = [960, 1280];

% paths 
inputPath = './calibrationPics';
outputPath = './cameraParameters';

if inputPath(end) ~= '/'
    inputPath(end+1) = '/';
end

if outputPath(end) ~= '/'
    outputPath(end+1) = '/';
end

if ~exist(outputPath,'dir')
    mkdir(outputPath);
end

% load estimated intrinsic parameters
fprintf('Load Intrinsic Parameters...\n')
file = sprintf('%sintrinsicParams.mat',outputPath);
load(file);    
fprintf('    done!\n') 

% load estimated extrinsic parameters
fprintf('Load Extrinsic Parameters...\n')
file = sprintf('%sextrinsicParams.mat',outputPath);
load(file);    
fprintf('    done!\n') 

fprintf('Load Camera Matrices...\n')
file = sprintf('%scameraMatrices.mat',outputPath);
load(file);    
fprintf('    done!\n') 


%% Evaluate accuracy of the approximation-polynomial
    
error = {};
distPolynomial6 = [];
distPolynomial10 = [];
r = -0.5:0.01:3.5;

for cam = 1:nCams

    fprintf('Collect data of camera %d ...\n', cam-1);

    % Get distorted image-points
    directory = sprintf('%sCamera_%d/intrinsic',inputPath, cam-1);
    file = sprintf('%s/calPointsIntrinsic.mat',directory);
    load(file);
    imageFileNames = calPointsIntrinsic.imageFileNames;
    imagePointsDist = calPointsIntrinsic.imagePoints;
    boardSize = calPointsIntrinsic.boardSize;
    calPointsUsed = repmat(permute(calPointsIntrinsic.imagesUsed,[2,3,1]),180,2,1);

    % Generate world coordinates of the corners of the squares
    CheckerboardPoints = generateCheckerboardPoints(boardSize, 18);

    % reproject world-points to get undistorted reprojected-image-points
    imagePointsUndist = NaN(size(imagePointsDist));
    for i=1:length(imageFileNames)
        % estimate extrinsics
        [ imagePointsUndistTemp, pointValid ] = removeRadialDistortion(imagePointsDist(:,:,i), ...
                                                                       cameraParamsIntrinsic{cam}.IntrinsicMatrix.',...
                                                                       cameraParamsIntrinsic{cam}.RadialDistortion);
        [R,t] = extrinsics(imagePointsUndistTemp(pointValid,:),CheckerboardPoints(pointValid,:),cameraParamsIntrinsic{cam});

        % caclulate undistorted reprojected-image-points
        worldPoints = [CheckerboardPoints,zeros(length(CheckerboardPoints),1)];
        imagePointsUndist(:,:,i) = worldToImage(cameraParamsIntrinsic{cam}, R, t,...
                                                worldPoints, 'ApplyDistortion', false);
    end

    % reshape imagePointsDist and imagePointsUndist
    imagePointsDist = reshape(permute(imagePointsDist,[2,1,3]),2,[]).';
    imagePointsUndist = reshape(permute(imagePointsUndist,[2,1,3]),2,[]).';
    calPointsUsed = reshape(permute(calPointsUsed,[2,1,3]),2,[]).';
    calPointsUsed = calPointsUsed(:,1);
    
    % normalize distorted image-points
    K = cameraParamsIntrinsic{cam}.IntrinsicMatrix.';
    imagePointsDistNormed = [imagePointsDist,ones(size(imagePointsDist,1),1)]*inv(K).';
    
    % calculate radius of distorted image-points
    rDist = sqrt(imagePointsDistNormed(:,1).^2 + imagePointsDistNormed(:,2).^2);
    
    % normalize undistorted image-points
    imagePointsUndistNormed = [imagePointsUndist,ones(size(imagePointsUndist,1),1)]*inv(K).';

    % calculate radius of undistorted image-points
    rUndist = sqrt(imagePointsUndistNormed(:,1).^2 + imagePointsUndistNormed(:,2).^2);
    
    % sort radius
    [rUndist,sortInd] = sort(rUndist);
    rDist = rDist(sortInd);
    calPointsUsed = calPointsUsed(sortInd);
    
    % randomly take measurements form different intervals
    N = 50;
    interval = 0:0.25:r(end);
    usedMeas = [];
    for i = 2:length(interval)
        ind = find( (rUndist >= interval(i-1)) & (rUndist < interval(i)) );
        if length(ind) > N
            randInd = randperm(length(ind),N);
            ind = ind(randInd);
        end
        usedMeas = cat(1,usedMeas,ind);
    end
    temp = false(size(calPointsUsed));
    temp(usedMeas) = true;
    usedMeas = temp;

    % give more weight to the measurements at the edge
    rUndist = cat(1,rUndist,repmat(rUndist(end-50:end),10^3,1));
    rDist = cat(1,rDist,repmat(rDist(end-50:end),10^3,1));
    calPointsUsed = cat(1,calPointsUsed,repmat(calPointsUsed(end-50:end),10^3,1));

    % find valid-radius 
    distCoeff = cameraParamsIntrinsic{cam}.RadialDistortion;
    rUndistValid = getValidRadius( 'u', distCoeff)
    rDistValid = getValidRadius( 'd', distCoeff);
    indValid = rUndist < rUndistValid;

    % try other polynomial of degree 6 but using all points
    nCoeff = 4;
    A = [];
    for i=1:nCoeff
        A = [A, rUndist.^(2*i-1)];
    end
    distPolynomial6(cam,:) = (inv(A.'*A)*(A.'*rDist)).';
    p6Val = zeros(size(r));
    for i=1:nCoeff
        p6Val = p6Val + distPolynomial6(cam,i)*r.^(2*(i-1));
    end
    
    % try higher degree (10) polynomial approximations
    nCoeff = 6;
    A = [];
    for i=1:nCoeff
        A = [A, rUndist.^(2*i-1)];
    end
    distPolynomial10(cam,:) = (inv(A.'*A)*(A.'*rDist)).';
    p10Val = zeros(size(r));
    pValMeas = zeros(size(rUndist));
    for i=1:nCoeff
        p10Val = p10Val + distPolynomial10(cam,i)*r.^(2*(i-1));
        pValMeas = pValMeas + distPolynomial10(cam,i)*rUndist.^(2*(i-1));
    end
    
    % save measurements
    pCam = ( 1 + distCoeff(1)*r.^2 + distCoeff(2)*r.^4 + distCoeff(3)*r.^6);
    temp = [rUndist(usedMeas & calPointsUsed(1:length(usedMeas))),...
            rDist(usedMeas & calPointsUsed(1:length(usedMeas))),...
            rDist(usedMeas & calPointsUsed(1:length(usedMeas)))-pValMeas(usedMeas & calPointsUsed(1:length(usedMeas))).*rUndist(usedMeas & calPointsUsed(1:length(usedMeas)))];
%     save(sprintf('RadDistPointsUsedCam%d.dat',cam-1), 'temp', '-ascii')
    temp = [rUndist(usedMeas & ~calPointsUsed(1:length(usedMeas))),...
            rDist(usedMeas & ~calPointsUsed(1:length(usedMeas))),...
            rDist(usedMeas & ~calPointsUsed(1:length(usedMeas)))-pValMeas(usedMeas & ~calPointsUsed(1:length(usedMeas))).*rUndist(usedMeas & ~calPointsUsed(1:length(usedMeas)))];
%     save(sprintf('RadDistPointsNotUsedCam%d.dat',cam-1), 'temp', '-ascii')
    temp = [r.',(pCam.*r).',(p6Val.*r).',(p10Val.*r).',(pCam.*r).'-(p10Val.*r).',(p6Val.*r).'-(p10Val.*r).',(p10Val.*r).'-(p10Val.*r).'];
%     save(sprintf('approxPolyCam%d.dat',cam-1), 'temp', '-ascii')

    % plot measurement-points and estimated polynomial
    figure(1);
    subplot(2,2,cam)
    plot(rUndist(~calPointsUsed),rDist(~calPointsUsed),'*'); hold on;
    plot(rUndist(calPointsUsed),rDist(calPointsUsed),'*'); hold on;
    plot(r,pCam.*r,r,p6Val.*r,r,p10Val.*r); hold on;
    plot([rUndistValid rUndistValid],[0 1.5],'r'); hold on;
    grid on;
    axis([-0.5 3.5 -0.5 2])
    xlabel('r');
    ylabel('r_d');
    title(sprintf('Camera %d', cam-1));
    hold off;
    
    if 0
    % show errors
    rDistEst = ( 1 + distCoeff(1)*rUndist.^2 + distCoeff(2)*rUndist.^4 + distCoeff(3)*rUndist.^6).*rUndist;
    error{cam} = sqrt((rDist - rDistEst).^2);
    figure(2);
    subplot(2,2,cam)
    hist(error{cam},100);
    grid on;
    xlabel('|r_d-p(r)*r|');
    title(sprintf('Camera %d: m = %f, std = %f', cam-1, mean(error{cam}(indValid)), std(error{cam}(indValid)) ));    
    
    
    figure(3);
    subplot(2,2,cam)
    plot(rUndist(indValid), abs((rDistEst(indValid) - rDist(indValid)).*rUndist(indValid)),'-*');
    grid on;
    title(sprintf('Camera %d', cam-1));
    
    
    % show valid region
    I = double(rgb2gray(imread(sprintf('./cameraParameters/validRegion/cam%d.jpg',cam-1))));
    [M,N] = size(I);
    [X,Y] = meshgrid(1:N,1:M);
    f = cameraParamsIntrinsic{cam}.FocalLength;
    p = cameraParamsIntrinsic{cam}.PrincipalPoint;
    r = sqrt(((X-p(1))/f(1)).^2 + ((Y-p(2))/f(2)).^2);
    I(abs(r-rDistValid)<0.001) = 300;
    I(round(p(2))-3:round(p(2))+3,round(p(1))-3:round(p(1))+3) = 300;
    figure(4);
    subplot(2,2,cam);
    imshow(I,[]);
    title(sprintf('Camera %d', cam-1));
%     imwrite(I/max(I(:)),sprintf('validRegionCam%d.png',cam-1))
    
    end
    
    fprintf('    done!\n')
    
end































