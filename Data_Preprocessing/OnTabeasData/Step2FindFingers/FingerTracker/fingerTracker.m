%
%   Tracking UV-Object in 3D
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
% picPath = '../CollectFingertipData/test_fingers/';
picPath = '../CollectFingertipData/test_circles_all/';

if inputPath(end) ~= '/'
    inputPath(end+1) = '/';
end
if picPath(end) ~= '/'
    picPath(end+1) = '/';
end

% Process Settings
startPic = 10;                  % number of picture to start with
nFingertips = 6;                % number of fingertips to track
T = 1/45;                       % sampling time


showPointsInViews = false;      % if true, reprojection of estimated points is shown
showFingertipTracks = false;    % if ture, ongoing tracking of fingertip is shown
evaluateResults = true;         % if ture, result is evaluated.
                                % only possible, if the curve is a known circle-curve

exportVideo = false;            % if true, video is exported
videoFileName = 'trackerVideo.avi';
if exportVideo
    showFingertipTracks = true;
end

% Parameters
su = 200;                       % standard deviation of the velocity noise [mm/s]
gateBorder = 2.5;               % in number of standard deviations of gate-border
assignmentMethod = 'indiv';     % method to use to assign measurements to tracks
                                %   'indiv':    treat every camera individual 
                                %   'combi':    combine measurements from different cameras
if strcmp(assignmentMethod,'combi')
    a = 0.5;                    % weighting of the ditance (track-measurement)
    b = 8.5;                    % weighting of the punishment-term
end

% geometric dimensions
nCams = 4;
ImageSize = [960, 1280];

% colors
camColors = [0.64 0.08 0.18 ; 
             0.47 0.67 0.19 ; 
             0 0.45 0.74 ; 
             0.87 0.49 0 ];
trackColor = [0.49, 0.18, 0.56 ;    % violett
              0.00, 0.75, 0.75 ;    % türkis
              0.80, 0.00, 0.53 ;    % pink
              0.85, 0.33, 0.10 ;    % orange
              0.00, 0.50, 0.00 ;    % grün
              0.08, 0.17, 0.55 ;    % blau
              0.47, 0.67, 0.19 ;    % hellgrün
              0.30, 0.75, 0.93 ;    % hellblau
              1.00, 0.60, 0.78 ;    % rosa
              0.93, 0.69, 0.13 ];   % gelb
                  
%% Initialize 

% load camera parameters
file = sprintf('%sintrinsicParams.mat',inputPath);
load(file);    
file = sprintf('%sextrinsicParams.mat',inputPath);
load(file);    
file = sprintf('%scameraMatrices.mat',inputPath);
load(file);
C = NaN(3,nCams);   % camera centers
m = NaN(3,nCams);   % view direction of cameras
for cam = 1:nCams
    P = cameraMatrices{cam};
    M = P(:,1:3);
    m(:,cam) = M(3,:).'/sqrt(M(3,:)*M(3,:).');
    C(:,cam) = -inv(M)*P(:,4);
end

% load timestamps
t_sec = getTimestamps( picPath );
t_sec = t_sec/1000;
dt_sec = diff(t_sec);

%load positions of fingertips (UV-Object) in differernt views
allFingertips = {};
for cam = 1:nCams
    load(sprintf('%sCamera_%d/UV_Bin/fingers.mat',picPath,cam-1));
    allFingertips{cam} = fingers;
    clear fingers
end
nCycles = length(allFingertips{1});

% generate necessary directories to save video
if exportVideo
    videoPath = sprintf('%sVideo/',picPath);
    if ~exist(videoPath,'dir')
        mkdir(videoPath);
    end
    for cam = 1:nCams
        outputPath = sprintf('%sCamera_%d/',videoPath,cam-1);
        if ~exist(outputPath,'dir')
            mkdir(outputPath);
        end
    end
end

% show setup
fig = figure(1); set(fig,'units','normalized','outerposition',[0 0 1 1]);
ax = axes('Parent',fig);
for cam = 1:nCams
    plotCamera('Location', cameraParamsExtrinsic{cam}.location,...
               'Orientation', cameraParamsExtrinsic{cam}.RotationMatrix, ...
               'Label', sprintf('cam %d',cam-1) ,...
               'AxesVisible',true,...
               'Color',camColors(cam,:),'Size',10); hold on;
end
set(ax,'CameraPosition',...
    [-1838.8739444507 1521.91812400702 4527.84979032433],'CameraTarget',...
    [409.586861914701 171.785078896797 98.226243933515],'CameraUpVector',...
    [0.118910802239649 0.964993685059684 -0.233767852592385],'CameraViewAngle',...
    5.73061100120959,'DataAspectRatio',[1 1 1],'PlotBoxAspectRatio',...
    [3 1.33333333333333 1]);
grid(ax,'on');
xlim(ax,[-50 850]);
ylim(ax,[-50 350]);
zlim(ax,[-50 250]);
xlabel('x [mm]');
ylabel('y [mm]');
zlabel('z [mm]');
hold off;

%% Collect points form Images and estimate 3D-location

nMeas = zeros(1,nCams);
measInd = zeros(nCams,2);
imagePoints = cell(1,nCams);
imagePointsRadius = cell(1,nCams);
imagePointsMeas = NaN(nFingertips,2,nCams);
worldPointsMeas = NaN(nFingertips,3);
allWorldPointsEst = NaN(nCycles,3,nFingertips);
allWorldPointsMeas = NaN(nCycles,3,nFingertips);
allNumUsedViews = NaN(nCycles,nFingertips);
allErrors = NaN(nCycles,nFingertips);
plotHandle = gobjects(0,0);
for i = startPic:nCycles
    
    % delete redundant plots
    while ~isempty(plotHandle)
        delete(plotHandle(end));
        plotHandle = plotHandle(1:end-1);
    end
    
    for cam = 1:nCams
        
        % collect fingertip-points from different views and undistort them
        imagePoints{cam} = allFingertips{cam}{i}.centers;
        imagePointsRadius{cam} = allFingertips{cam}{i}.radii;
        [ imagePoints{cam}, pointValid ] = removeRadialDistortion(imagePoints{cam}, ...
                                                                  cameraParamsIntrinsic{cam}.IntrinsicMatrix.', ...
                                                                  cameraParamsIntrinsic{cam}.RadialDistortion );
        imagePoints{cam} = imagePoints{cam}(pointValid,:);
        imagePointsRadius{cam} = imagePointsRadius{cam}(pointValid);
        
        nMeas(cam) = size(imagePoints{cam},1);
        measInd(cam,:) = [sum(nMeas(1:cam-1))+1, sum(nMeas(1:cam))];
    end
    
    if (i == startPic)  % initialization
        
        % calculate all possible points, but only take points into account that have been seen in all views
        [ worldPointsMeas, numberOfUsedViews, errorVal ] = extensiveWorldPointEstimation( imagePoints, cameraMatrices, nCams, 10^2 );
        
        % take nFingertips with smallest errorValue
        if length(errorVal) >= nFingertips
            worldPointsMeas = worldPointsMeas(1:nFingertips,:);
            errorVal = errorVal(1:nFingertips);
            numberOfUsedViews = numberOfUsedViews(1:nFingertips);
        else
            error('Could not find enough fingertips.')
        end
                
        % initialize Kalman-Filter for every found fingertip
        worldPointsEst = NaN(size(worldPointsMeas));
        for j = 1:nFingertips
            kalmanFilt(j) = KalmanFilter(T,su);
            kalmanFilt(j).setX([worldPointsMeas(j,1); 0; worldPointsMeas(j,2); 0; worldPointsMeas(j,3); 0]);
            worldPointsEst(j,:) = (kalmanFilt(j).C*kalmanFilt(j).x).';
        end
        allNumUsedViews(i,:) = numberOfUsedViews;
        
    else    % tracking
        
        % estimate next position and size of fingertips
        for j = 1:nFingertips
            kalmanFilt(j).setT(dt_sec(i-1));
            kalmanFilt(j).kalPredict();
            worldPointsEst(j,:) = (kalmanFilt(j).C*kalmanFilt(j).x).';
        end
        
        % draw Covariance-Ellipsoid
        if showFingertipTracks
            figure(1);
            hold on;
            for j = 1:nFingertips
                N = 20^2;
                Sigma = kalmanFilt(j).C*kalmanFilt(j).P*kalmanFilt(j).C.' + kalmanFilt(j).Qv{max(allNumUsedViews(i-1,j),1)};
                [Phi,Lambda] = eig(Sigma);
                if det(Phi) < 0
                    Phi(:,1) = -Phi(:,1);
                end
                kalX = kalmanFilt(j).x;
                mu = kalX([1,3,5]);
                dv = 0.1*kalX([2,4,6]);
                p = sqrt(diag(Lambda));
                [THETA,PHI] = ndgrid(linspace(0,pi,sqrt(N)),linspace(0,2*pi,sqrt(N)));
                x = gateBorder*p(1)*sin(THETA).*cos(PHI);
                y = gateBorder*p(2)*sin(THETA).*sin(PHI);
                z = gateBorder*p(3)*cos(THETA);
                samples = Phi*[x(:).';y(:).';z(:).']+repmat(mu,1,N);
                plotHandle(end+1) = surf(reshape(samples(1,:),sqrt(N),sqrt(N)),...
                                         reshape(samples(2,:),sqrt(N),sqrt(N)),...
                                         reshape(samples(3,:),sqrt(N),sqrt(N)),'FaceAlpha',0);
                plotHandle(end+1) = plot3(mu(1),mu(2),mu(3),'o','Color',trackColor(j,:),'MarkerFaceColor',trackColor(j,:));
                plotHandle(end+1) = quiver3(mu(1),mu(2),mu(3),dv(1),dv(2),dv(3),'Color',trackColor(j,:));
            end
            hold off;
        end
        
        if strcmp(assignmentMethod,'combi')     % solve assignment-problem by combining measurements form different views

            % calculate distances between all rays of all cameras
            camCenter = NaN(3,sum(nMeas));
            rayDir = NaN(3,sum(nMeas));
            for cam = 1:nCams
                P = cameraMatrices{cam};
                M = P(:,1:3);
                camCenter(:,measInd(cam,1):measInd(cam,2)) = repmat(-inv(M)*P(:,4),1,nMeas(cam));
                rayDir(:,measInd(cam,1):measInd(cam,2)) = inv(M)*[imagePoints{cam},ones(nMeas(cam),1)].';
            end
            centers = repmat(permute(camCenter,[3,2,1]),sum(nMeas),1,1) - repmat(permute(camCenter.',[1,3,2]),1,sum(nMeas),1);
            normals = cross( repmat(permute(rayDir,[3,2,1]),sum(nMeas),1,1), repmat(permute(rayDir.',[1,3,2]),1,sum(nMeas),1) );
            distancesRays = abs(sum(centers.*normals,3))./sqrt(sum(normals.*normals,3));
            distancesRays(isnan(distancesRays)) = 0;

            % calculate euclidean distance and mahalanobis distance between all tracks and all rays
            distancesEuclideanRaysTracks = NaN(sum(nMeas),nFingertips);
            distancesMahalanobisRaysTracks =  NaN(sum(nMeas),nFingertips);
            for cam = 1:nCams

                % calculate the 3D-world-points on each ray, that is closest to the estimated point
                tempWorldPointsEst = reshape(repmat(worldPointsEst,1,nMeas(cam)).',3,[]).';
                [ eucDist, tempWorldPointsMeas ] = distanceWorldToRay(tempWorldPointsEst,...
                                                                      repmat(imagePoints{cam},nFingertips,1),...
                                                                      cameraMatrices{cam});

                % draw measurement-rays
                if showFingertipTracks
                    figure(1);
                    hold on;
                    n = size(tempWorldPointsMeas,1);
                    plotHandle(end+1:end+n) = plot3([repmat(cameraParamsExtrinsic{cam}.location(1),n,1),tempWorldPointsMeas(:,1)].',...
                                                    [repmat(cameraParamsExtrinsic{cam}.location(2),n,1),tempWorldPointsMeas(:,2)].',...
                                                    [repmat(cameraParamsExtrinsic{cam}.location(3),n,1),tempWorldPointsMeas(:,3)].',...
                                                    '-*','Color',camColors(cam,:));
                    hold off;
                end

                % calculate the euclidean distance: eucDist(i,j): euclidean distance between measurement i and estimation j
                eucDist = reshape(eucDist,nMeas(cam),nFingertips);

                % calculate the mahalanobis distance: mahDist(i,j): mahalanobis distance between measurement i and estimation j
                vecEstMeas = permute(reshape((tempWorldPointsEst-tempWorldPointsMeas).',3,nMeas(cam),nFingertips),[2,1,3]);
                mahDist = NaN(nMeas(cam),nFingertips);
                for j=1:nFingertips
                    Sigma = kalmanFilt(j).C*kalmanFilt(j).P*kalmanFilt(j).C.' + kalmanFilt(j).Qv{max(allNumUsedViews(i-1,j),1)};
                    mahDist(:,j) = sqrt(diag(vecEstMeas(:,:,j)*inv(Sigma)*vecEstMeas(:,:,j).'));
                end

                % collect calculated values
                distancesEuclideanRaysTracks(measInd(cam,1):measInd(cam,2),:) = eucDist;
                distancesMahalanobisRaysTracks(measInd(cam,1):measInd(cam,2),:) = mahDist;
            end

            % Threshold mahDist-matrix to decide which measurements are inside
            % the gate and which ones outside
            gateMat = distancesMahalanobisRaysTracks  < gateBorder;

            % find all combinations of measurements
            combinations = getMeasurementCombinations(nMeas, 2);
            nCombs = size(combinations,1);

            % calculate cost-matrix and cluster-matrix
            costMat = zeros(nCombs,nFingertips);
            clustMat = zeros(nCombs,nFingertips);
            for k = 1:nCombs
                boolInd = combinations(k,:)~=0;
                n = sum(boolInd);
                intInd = measInd(:,1).' + combinations(k,:) - 1;
                intInd = intInd(boolInd);
                costMat(k,:) = a*mean(distancesEuclideanRaysTracks(intInd,:).^2,1) + ...                    % distance of measurement to track
                               (1-a)*sum(sum(triu(distancesRays(intInd,intInd).^2)))/nchoosek(n,2) + ...    % distance of measurement to measurement
                               b*(nCams-n).^2;                                                              % cost for using less views
                clustMat(k,:) = all(gateMat(intInd,:),1);
            end

            % reduce cost-matrix and cluster-matrix to reasonable combinations
            % (combinations which are inside the gate of at least one track )
            index = any(clustMat,2);
            clustMat = clustMat(index,:);
            costMat = costMat(index,:);
            combinations = combinations(index,:);
            nCombs = size(combinations,1);

            % make costs for combinations outside the gate forbiddingly high
            costMat(~clustMat) = 10000;

            % find clusters and solve for each cluser
            % the assignment-problem using munkres-algorithm
            [clusterRows, clusterCols] = findClusters( clustMat );
            fullAssignment = false(nCombs, nFingertips);
            fullCost = 0;
            for j=1:size(clusterRows,2)

                % solve assignment-problem for found cluster
                [assignment,cost] = munkres(costMat(clusterRows(:,j),clusterCols(:,j)));

                % update global assignment variables
                fullCost = fullCost + cost;
                fullAssignment(clusterRows(:,j),clusterCols(:,j)) = assignment;
            end
            fullAssignment(~clustMat) = false;

            % assign measurements to finger-tracks
            imagePointsMeas = NaN(nFingertips,2,nCams);
            measurementNumbs = zeros(nFingertips,nCams);
            [row,col] = find(fullAssignment);
            measurementNumbs(col,:) = combinations(row,:);
            boolInd = (measurementNumbs ~= 0);
            for cam = 1:nCams
                imagePointsMeas(boolInd(:,cam),:,cam) = imagePoints{cam}(measurementNumbs(boolInd(:,cam),cam),:);
            end

        elseif strcmp(assignmentMethod,'indiv')  % sovle assignment problem for each camera individual

            % sovle assignment problem for each camera (measurement <-> track)
            for cam = 1:nCams

                % calculate the 3D-world-points on each ray, that is closest to the estimated point
                tempWorldPointsEst = reshape(repmat(worldPointsEst,1,nMeas(cam)).',3,[]).';
                [ eucDist, tempWorldPointsMeas ] = distanceWorldToRay(tempWorldPointsEst,...
                                                                      repmat(imagePoints{cam},nFingertips,1),...
                                                                      cameraMatrices{cam});

                % draw measurement-rays
                if showFingertipTracks
                    figure(1);
                    hold on;
                    n = size(tempWorldPointsMeas,1);
                    plotHandle(end+1:end+n) = plot3([repmat(cameraParamsExtrinsic{cam}.location(1),n,1),tempWorldPointsMeas(:,1)].',...
                                                    [repmat(cameraParamsExtrinsic{cam}.location(2),n,1),tempWorldPointsMeas(:,2)].',...
                                                    [repmat(cameraParamsExtrinsic{cam}.location(3),n,1),tempWorldPointsMeas(:,3)].',...
                                                    '-*','Color',camColors(cam,:));
                    hold off;
                end

                % calculate the euclidean distance: eucDist(i,j): euclidean distance between measurement i and estimation j
                eucDist = reshape(eucDist,nMeas(cam),nFingertips);

                % calculate the mahalanobis distance: mahDist(i,j): mahalanobis distance between measurement i and estimation j
                vecEstMeas = permute(reshape((tempWorldPointsEst-tempWorldPointsMeas).',3,nMeas(cam),nFingertips),[2,1,3]);
                mahDist = NaN(nMeas(cam),nFingertips);
                for j=1:nFingertips
                    Sigma = kalmanFilt(j).C*kalmanFilt(j).P*kalmanFilt(j).C.' + kalmanFilt(j).Qv{max(allNumUsedViews(i-1,j),1)};
                    mahDist(:,j) = sqrt(diag(vecEstMeas(:,:,j)*inv(Sigma)*vecEstMeas(:,:,j).'));
                end


                % Threshold mahDist-matrix to decide which measurements are inside 
                % the gate and which ones outside
                gateMat = mahDist  < gateBorder;

                % find clusters and solve for each cluser 
                % the assignment-problem using munkres-algorithm
                costMat = eucDist.^2;
                [clusterRows, clusterCols] = findClusters( gateMat );
                fullAssignment = false(nMeas(cam), nFingertips);
                fullCost = 0;
                for j=1:size(clusterRows,2)

                    % solve assignment-problem for found cluster                
                    costMatCluster = costMat(clusterRows(:,j),clusterCols(:,j));
                    costMatCluster(~gateMat(clusterRows(:,j),clusterCols(:,j))) = 1000;
                    [assignment,cost] = munkres(costMatCluster);
                    assignment(~gateMat(clusterRows(:,j),clusterCols(:,j))) = false;

                    % update global assignment variables
                    fullCost = fullCost + cost;
                    fullAssignment(clusterRows(:,j),clusterCols(:,j)) = assignment;
                end

                % assign measurements to finger-tracks
                [row,col] = find(fullAssignment);
                imagePointsMeas(~sum(fullAssignment,1),:,cam) = NaN;
                imagePointsMeas(col,:,cam) = imagePoints{cam}(row,:);
            end
        else
            error('Chose a valid assignment method.')
        end
        
        % calculate 3D-position of measured fingertip-points and update the Kalman-Filter
        for j = 1:nFingertips
            
            if strcmp(assignmentMethod,'combi')
                ind = (measurementNumbs(j,:) ~= 0);
                nViews = sum(ind);
                allNumUsedViews(i,j) = nViews;
            elseif strcmp(assignmentMethod,'indiv')
                ind = squeeze(~isnan(imagePointsMeas(j,1,:)));
                nViews = sum(ind);
                allNumUsedViews(i,j) = nViews;
            end
            
            if nViews >= 2
                % calculate 3D-position of measured fingertip-points,
                % if points are available in at least 2 views
                [worldPointsMeas(j,:), error] = multiviewToWorld('minID',...
                                                             imagePointsMeas(j,:,ind),...
                                                             cameraMatrices(ind));
                allErrors(i,j) = error;

                % make measurement update of Kalman-filters
                y = worldPointsMeas(j,:).';
                kalmanFilt(j).kalCorrect(y, nViews);
                worldPointsEst(j,:) = (kalmanFilt(j).C*kalmanFilt(j).x).';
            else
                worldPointsMeas(j,:) = NaN;
                fprintf('No measurement-update for fingertip-track %d in cycle %d\n',j,i);
                fprintf('    -> Velocity is slowed down.\n');

                % Velocity of fingertip is slowed down
                x_new = kalmanFilt(j).x;
                x_new([2,4,6]) = 0.9*x_new([2,4,6]);
                kalmanFilt(j).setX(x_new);
            end
        end
    end
    allWorldPointsEst(i,:,:) = permute(worldPointsEst,[3,2,1]);
    allWorldPointsMeas(i,:,:) = permute(worldPointsMeas,[3,2,1]);
    
    % show fingertip-tracks
    if showFingertipTracks
        figure(1);
        hold on;
        for j=1:nFingertips
            plotHandle(end+1) = plot3(allWorldPointsEst(:,1,j),...
                                      allWorldPointsEst(:,2,j),...
                                      allWorldPointsEst(:,3,j),'-o','Color',trackColor(j,:));
            plotHandle(end+1) = plot3(allWorldPointsMeas(:,1,j),...
                                      allWorldPointsMeas(:,2,j),...
                                      allWorldPointsMeas(:,3,j),'-*','Color',trackColor(j,:));
        end
        hold off;
        
        % collect vidoe frames
        if exportVideo
            videoFrames(i-startPic+1) = getframe(gcf);
        end
    end
    
    % show estimated points in different views
    if exportVideo || showPointsInViews
        for cam = 1:nCams
            [reproImagePointsEst,~] = ...
                    myWorldToImage(cameraParamsIntrinsic{cam},...
                                   cameraParamsExtrinsic{cam}.RotationMatrix,...
                                   cameraParamsExtrinsic{cam}.location,...
                                   worldPointsEst);
            I = double(imread(sprintf('%sCamera_%d/markedFingers/pic%d.png',picPath,cam-1,i-1)));
            I = insertShape(I,'FilledCircle',[reproImagePointsEst repmat(10,nFingertips,1)],'Color',trackColor(1:nFingertips,:)*255,'opacity',0.7);
            I = I/max(I(:));
            if exportVideo
                imwrite(I,sprintf('%sCamera_%d/pic%d.png',videoPath,cam-1,i-1));
            end
            if showPointsInViews
                figure(2);
                subplot(2,2,cam)
                imshow(I);
                title(sprintf('Camera %d',cam-1))
            end
        end
    end
    
    % show progress
    if showFingertipTracks
        pause(0.001);
        i
    end
end

% save video
if exportVideo
    v = VideoWriter(sprintf('%s%s',videoPath, videoFileName));
    open(v)
    writeVideo(v,videoFrames);
    close(v)
end


%% evaluate Results

if evaluateResults

    % load circle parameters
    load(sprintf('%scircleParams.mat',picPath));

    % assign tracks to circle-curves
    dist = NaN(nFingertips,nFingertips);
    for j=1:nFingertips
        [dist(:,j),~] = distancePointCircle(squeeze(allWorldPointsEst(startPic,:,:)).',...
                                            center(:,j), normal(:,j), radius(j) );

    end
    assignment = munkres(dist);
    [row,col] = find(assignment);
    [~,sortInd] = sort(row);
    center = center(:,sortInd);
    normal = normal(:,sortInd);
    radius = radius(:,sortInd);
    circlePoints = circlePoints(:,:,sortInd);

    % plot circle-curves
    figure(1);
    hold on;
    for j=1:nFingertips
        plotHandle(end+1) = plot3(circlePoints(:,1,j),...
                                  circlePoints(:,2,j),...
                                  circlePoints(:,3,j),'Color',trackColor(j,:));
    end
    hold off;

    % calculate distance of estimated 3D-points to circle-curve
    error = NaN(nCycles,nFingertips);
    for j=1:nFingertips
        [error(:,j),~] = distancePointCircle(allWorldPointsEst(:,:,j),...
                                             center(:,j), normal(:,j), radius(j) );
    end

    % show result
    figure(3)
    subplot(1,3,1);
    histogram(error(:),0:0.1:max(error(:)),'Normalization','probability');
    xlabel('Error [mm]');
    title(sprintf( 'Circles: m = %f, s = %f', mean(error(:),'omitnan'), std(error(:),'omitnan') ));
    grid on;
    subplot(1,3,2);
    histogram(allNumUsedViews(:),-0.5:1:nCams+0.5);
    xlim([-0.5,nCams+0.5])
    xlabel('Number of used camera-views');
    grid on;
    subplot(1,3,3);
    histogram(allErrors(:),1000);
    xlabel('Mean-Squared Errors of measured 3d-world-points');
    grid on;
    
end

% show tracks
if ~showFingertipTracks
    figure(1);
    hold on;
    for j=1:nFingertips
        plotHandle(end+1) = plot3(allWorldPointsEst(:,1,j),...
                                  allWorldPointsEst(:,2,j),...
                                  allWorldPointsEst(:,3,j),'-o','Color',trackColor(j,:));
        plotHandle(end+1) = plot3(allWorldPointsMeas(:,1,j),...
                                  allWorldPointsMeas(:,2,j),...
                                  allWorldPointsMeas(:,3,j),'-*','Color',trackColor(j,:));
    end
    hold off;    
end

