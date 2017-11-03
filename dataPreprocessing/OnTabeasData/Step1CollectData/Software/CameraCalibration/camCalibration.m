%
%   Script for calibrating the cameras
%
%   tmendez, 18.05.2017
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all;
close all;
clc;

addpath(genpath('../matlabHelperFunctions'));

%% Parameters

% paths 
inputPath = './calibrationPics';
outputPath = './cameraParameters';

% geometric dimensions
squareSize = 18;  % in units of 'mm'
nCams = 4;
boardSizeChessSmall = [13, 16];
boardSizeChessLarge = [13, 20];
ImageSize = [960, 1280];

% Process
processCams = [1 2 3 4];
collectPointsForIntrinsicCalibration = 0;
collectPointsForExtrinsicCalibration = 0;

estIntrinsicParams = 0;
showReprojections = 0;
estExtrinsicParams = 0;
evalCalibration = 0;

if inputPath(end) ~= '/'
    inputPath(end+1) = '/';
end

if outputPath(end) ~= '/'
    outputPath(end+1) = '/';
end

if ~exist(outputPath,'dir')
    mkdir(outputPath);
end


%% Collect all checkerboard-points for intrinsic calibration

if collectPointsForIntrinsicCalibration
    
    for cam = processCams

        fprintf('Collecting checkerboard-points for intrinsic calibration of camera %d ...\n', cam-1);

        % Get images to process
        directory = sprintf('%sCamera_%d/intrinsic',inputPath, cam-1);
        [ ~, ~, imageFileNames ] = getImageFileNames( directory );

        % Detect checkerboards in images
        imagesUsed = zeros(length(imageFileNames),1);
        imagePoints = zeros((boardSizeChessSmall(1)-1)*(boardSizeChessSmall(2)-1),2,0);
        for i=1:length(imageFileNames)
            I = double(imread(imageFileNames{i}));
            [imagePoints_temp, ~ ] = findCheckerboardPoints( I, boardSizeChessSmall );
            if ~isempty(imagePoints_temp)
                % Checkerboard was detected correctly
                imagesUsed(i) = 1;
                imagePoints(:,:,end+1) = imagePoints_temp;
            else
                % Checkerboard was NOT detected correctly
                % -> try again after histogram equalization
                I = histeq(I/max(I(:)),2^8);
                [imagePoints_temp, ~ ] = findCheckerboardPoints( I, boardSizeChessSmall );
                if ~isempty(imagePoints_temp)
                    % Checkerboard was detected correctly
                    imagesUsed(i) = 1;
                    imagePoints(:,:,end+1) = imagePoints_temp;
                end
            end
        end
        boardSize = boardSizeChessSmall;
        imagesUsed = logical(imagesUsed);
        imageFileNames = imageFileNames(imagesUsed);

        % save collected checkerboard-points
        calPointsIntrinsic = struct('imageFileNames', {imageFileNames}, ...
                                    'imagePoints', imagePoints, ...
                                    'boardSize', boardSize, ...
                                    'imagesUsed', logical(ones(length(imageFileNames),1)));
        file = sprintf('%s/calPointsIntrinsic.mat',directory);
        save(file, 'calPointsIntrinsic'); 

        fprintf('    done!\n')

    end
end

clearvars -except inputPath outputPath squareSize nCams boardSizeChessSmall ...
    boardSizeChessLarge ImageSize processCams collectPointsForIntrinsicCalibration ...
    collectPointsForExtrinsicCalibration estIntrinsicParams showReprojections ...
    estExtrinsicParams evalCalibration cameraParamsIntrinsic estimationErrorsIntrinsic...
    cameraParamsExtrinsic cameraMatrices error2D error3D

%% Collect all checkerboard-points for extrinsic calibration

if collectPointsForExtrinsicCalibration
    
    for cam = processCams
        
        fprintf('Collecting checkerboard-points for extrinsic calibration of camera %d ...\n', cam-1);

        % Get images to process
        directory = sprintf('%sCamera_%d/extrinsic',inputPath, cam-1);
        [ ~, imageFileNames, ~ ] = getImageFileNames( directory );

        % calculate the image points and the world points for each image
        allImageFileNames = {};
        allImagePoints = {};
        allWorldPoints = {};
        for i = 1:length(imageFileNames)

            % Detect checkerboard in image
            I = double(imread(imageFileNames{i}));
            [imagePoints, boardSize] = findCheckerboardPoints(I,[]);            
            if any(boardSize == 0)
                fprintf('No checkerboard detected.\n    Discarded %s ...\n', imageFileNames{i});
                continue;
            end
            
            % Check whether the origin is in the right corner
            b = 0;
            x = 0;
            y = 0;
            cnt = 0;
            while( b ~= 'y' && b ~= 27 )
                figure(1);
                imshow(I,[]); hold on;
                plot(imagePoints(:,1),imagePoints(:,2),'g*');
                plot(imagePoints(1,1),imagePoints(1,2),'ro','MarkerSize', 10, 'LineWidth',2);
                str = {['(x_0,y_0)?'], 'y | n | esc'};
                text(imagePoints(1,1)+20,imagePoints(1,2), str,'Color','red',...
                                                               'FontSize', 15,...
                                                               'FontWeight', 'bold');
                hold off;
                
                [~, ~, b] = ginput(1);
                if ( b == 'n' )
                    % rotate checkerboard by -90°
                    [ imagePoints, boardSize ] = changeCheckerboardOrigin( imagePoints(boardSize(1)-1,1), ...
                                                                           imagePoints(boardSize(1)-1,2), ...
                                                                           imagePoints, boardSize);
                elseif ( b == 'y' )
                    % read in corner coordinates
                    x = input('x_0 = ');
                    y = input('y_0 = ');
                    if isempty(x) || isempty(y)
                        b = ' ';
                    end
                end
            end
            
            if (b == 27)
                % esc -> no valid origin selected -> discard image
                fprintf('No Valid origin.\n    Discarded %s ...\n', imageFileNames{i});
                continue;
            else
                fprintf('\n');                
            end
            
            % Get world-coordinates of calibration-points
            ind = strfind(imageFileNames{i},'_');
            ind = ind(end-5:end);
            device = (imageFileNames{i}(ind(1)+1:ind(2)-1) == 'r')*2 + ...
                     (imageFileNames{i}(ind(2)+1:ind(3)-1) == 'h')*1 + 1;
            position = str2double(imageFileNames{i}(ind(3)+1:ind(4)-1));
            angle = str2double(imageFileNames{i}(ind(4)+1:ind(5)-1))/180*pi;
            viewDir = str2double(imageFileNames{i}(ind(5)+1:ind(6)-1));
            [worldPoints, ~, ~, ~] = calcRefPoints(device, position, angle, viewDir, 2);

            % Discard not detected worldPoints
            worldPoints = worldPoints(y+1:y+boardSize(1)-1,x+1:x+boardSize(2)-1,:);
            worldPoints = reshape(worldPoints,[],3);

            % Collect world- and image-points
            allImageFileNames{end+1} = imageFileNames{i};
            allImagePoints{end+1} = imagePoints;
            allWorldPoints{end+1} = worldPoints;
            
            % show collected word-points
            figure(2);
            hold on;
            plot3(worldPoints(:,1), worldPoints(:,2), worldPoints(:,3),'*','Color',[0.49,0.18,0.56]);
            plot3(worldPoints(1,1), worldPoints(1,2), worldPoints(1,3),'*','Color',[0.85,0.33,0.10]); 
            hold off;
            grid on;
            axis equal;
            axis([-100 900 -150 450 0 300]);
            xlabel('x [mm]');
            ylabel('y [mm]');
            zlabel('z [mm]');

        end
        
        % save collected checkerboard-points
        calPointsExtrinsic = struct('imageFileNames', {allImageFileNames}, ...
                                    'imagePoints', {allImagePoints}, ...
                                    'worldPoints', {allWorldPoints} );
        file = sprintf('%s/calPointsExtrinsic.mat',directory);
        save(file, 'calPointsExtrinsic'); 
        
        fprintf('    done!\n')
        
    end   
end

clearvars -except inputPath outputPath squareSize nCams boardSizeChessSmall ...
    boardSizeChessLarge ImageSize processCams collectPointsForIntrinsicCalibration ...
    collectPointsForExtrinsicCalibration estIntrinsicParams showReprojections ...
    estExtrinsicParams evalCalibration cameraParamsIntrinsic estimationErrorsIntrinsic...
    cameraParamsExtrinsic cameraMatrices error2D error3D

%% estimate intrinsic parameters of the cameras

if estIntrinsicParams
    
    cameraParamsIntrinsic = cell(nCams,1);
    estimationErrorsIntrinsic = cell(nCams,1);

    for cam = processCams

        fprintf('Estimating intrinsic parameters of camera %d ...\n', cam-1);

        % Get collected checkerboard-points
        directory = sprintf('%sCamera_%d/intrinsic',inputPath, cam-1);
        file = sprintf('%s/calPointsIntrinsic.mat',directory);
        load(file);
        imageFileNames = calPointsIntrinsic.imageFileNames;
        imagePoints = calPointsIntrinsic.imagePoints;
        boardSize = calPointsIntrinsic.boardSize;

        % Generate world coordinates of the corners of the squares
        CheckerboardPoints = generateCheckerboardPoints(boardSize, squareSize);

        % Calibrate the intrinsics of the camera
        [cameraParams, imagesUsed, estimationErrors] = ...
            estimateCameraParameters(imagePoints, CheckerboardPoints, ...
                                     'ImageSize', ImageSize, ...
                                     'EstimateSkew', false, ...
                                     'EstimateTangentialDistortion', false, ...
                                     'NumRadialDistortionCoefficients', 3, ...
                                     'WorldUnits', 'mm', ...
                                     'InitialIntrinsicMatrix', [], ...
                                     'InitialRadialDistortion', []);
        reprojectionErrors = squeeze(mean(sqrt(sum(cameraParams.ReprojectionErrors.^2,2)),1));
        threshold = (mean(reprojectionErrors)+4*std(reprojectionErrors));
        ind = reprojectionErrors < threshold;
        
        if (cam ~= processCams(1))
            close 3;
        end
        fig = figure(3); set(fig,'units','normalized','outerposition',[0 0 1 1]);
        j = 1; subplot(4,4,j); j=j+1;
        plot(reprojectionErrors); hold on;
        plot([0, length(reprojectionErrors)], [threshold, threshold]); hold off;
        grid on;
        xlabel('Images');
        ylabel('Mean Error [Pixels]');
        pause(0.1);
        
        % Recalibrate if some reprojection Errors are really high
        while any(reprojectionErrors >= threshold)
        
            remImagePoints = imagePoints(:,:,ind);
            [cameraParams, imagesUsed, estimationErrors] = ...
                estimateCameraParameters(remImagePoints, CheckerboardPoints, ...
                                         'ImageSize', ImageSize, ...
                                         'EstimateSkew', false, ...
                                         'EstimateTangentialDistortion', false, ...
                                         'NumRadialDistortionCoefficients', 3, ...
                                         'WorldUnits', 'mm', ...
                                         'InitialIntrinsicMatrix', [], ...
                                         'InitialRadialDistortion', []);
            reprojectionErrors = squeeze(mean(sqrt(sum(cameraParams.ReprojectionErrors.^2,2)),1));
            threshold = (mean(reprojectionErrors)+4*std(reprojectionErrors));
            ind(ind) = reprojectionErrors < threshold;
            
            figure(3);
            if (j>16)
                j=1;
            end
            subplot(4,4,j); j=j+1;
            plot(reprojectionErrors); hold on;
            plot([0, length(reprojectionErrors)], [threshold, threshold]); hold off;
            xlabel('Images');
            ylabel('Mean Error [Pixels]');
            grid on;
            pause(0.1);
            
        end
        threshold = (mean(reprojectionErrors)+2*std(reprojectionErrors));
        
        % save which images have been used
        calPointsIntrinsic.imagesUsed = ind;
        save(file, 'calPointsIntrinsic'); 
        
        % collect estimated parameters
        cameraParamsIntrinsic{cam} = cameraParams;
        estimationErrorsIntrinsic{cam} = estimationErrors;

        fprintf('    done!\n')
    end

    % save estimated intrinsic parameters
    fprintf('Save Intrinsic Parameters...\n')
    file = sprintf('%sintrinsicParams.mat',outputPath);
    save(file, 'cameraParamsIntrinsic', 'estimationErrorsIntrinsic');    
    fprintf('    done!\n')

else
    
    % load estimated intrinsic parameters
    fprintf('Load Intrinsic Parameters...\n')
    file = sprintf('%sintrinsicParams.mat',outputPath);
    load(file);    
    fprintf('    done!\n') 
end

% show performanec of intrinsic camera calibration
for cam = processCams
    
        displayErrors(estimationErrorsIntrinsic{cam}, cameraParamsIntrinsic{cam});
        fig = figure(4); set(fig,'units','normalized','outerposition',[0 0 1 1]);
        subplot(2,4,2*cam-1);
        showReprojectionErrors(cameraParamsIntrinsic{cam});
        subplot(2,4,2*cam);
        showExtrinsics(cameraParamsIntrinsic{cam}, 'CameraCentric');
        
end

clearvars -except inputPath outputPath squareSize nCams boardSizeChessSmall ...
    boardSizeChessLarge ImageSize processCams collectPointsForIntrinsicCalibration ...
    collectPointsForExtrinsicCalibration estIntrinsicParams showReprojections ...
    estExtrinsicParams evalCalibration cameraParamsIntrinsic estimationErrorsIntrinsic...
    cameraParamsExtrinsic cameraMatrices error2D error3D

%% show reprojection-error and undistorted images

if showReprojections
    for cam = processCams

        fprintf('Show reprojection-error and undistorted images of camera %d ...\n', cam-1);
        
        % Get collected checkerboard-points
        directory = sprintf('%sCamera_%d/intrinsic',inputPath, cam-1);
        file = sprintf('%s/calPointsIntrinsic.mat',directory);
        load(file);
        imageFileNames = calPointsIntrinsic.imageFileNames;
        imagePoints = calPointsIntrinsic.imagePoints;
        boardSize = calPointsIntrinsic.boardSize;
        imagesUsed = calPointsIntrinsic.imagesUsed;

        % Generate world coordinates of the corners of the squares
        CheckerboardPoints = generateCheckerboardPoints(boardSize, squareSize);
        
        % show reprojection-error and undistorted images
        reprojectionErrors = zeros(length(imageFileNames),1);
        reprojectionErrors(imagesUsed) = squeeze(mean(sqrt(sum(cameraParamsIntrinsic{cam}.ReprojectionErrors.^2,2)),1));
        for i=1:length(imageFileNames)
            % calculate reprojection-error
            [ imagePointsUndist, pointValid ] = removeRadialDistortion(imagePoints(:,:,i), ...
                                                                       cameraParamsIntrinsic{cam}.IntrinsicMatrix.',...
                                                                       cameraParamsIntrinsic{cam}.RadialDistortion);
            [R,t] = extrinsics(imagePointsUndist(pointValid,:),CheckerboardPoints(pointValid,:),cameraParamsIntrinsic{cam});
            [imagePoints_hat, ~] = myWorldToImage(cameraParamsIntrinsic{cam}, R.', -R*(t.'), ...
                                                 [CheckerboardPoints(pointValid,:),zeros(length(CheckerboardPoints(pointValid,:)),1)]);

            if (imagesUsed(i) == 0)
                reprojectionErrors(i) = mean(sqrt(sum((imagePoints_hat-imagePoints(pointValid,:,i)).^2,2)));
            end

            % show reprojected-points if the reprojection-error is high
            reproErrorUsedImages = squeeze(mean(sqrt(sum(cameraParamsIntrinsic{cam}.ReprojectionErrors.^2,2)),1));
            threshold = (mean(reproErrorUsedImages)+2*std(reproErrorUsedImages));
            if reprojectionErrors(i) > threshold
                imOrig = double(imread(imageFileNames{i}));
                [imUndist, newOrigin] = undistortImage(imOrig, cameraParamsIntrinsic{cam}, 'linear', 'OutputView','full');
                imagePointsUndist = [imagePointsUndist(:,1) - newOrigin(1), ...
                                     imagePointsUndist(:,2) - newOrigin(2)];

                fig = figure(5); set(fig,'units','normalized','outerposition',[0 0 1 1]);
                subplot(1,2,1);
                imshow(imOrig,[]); hold on;
                plot(imagePoints(:,1,i),imagePoints(:,2,i),'g*');
                if (imagesUsed(i) == 0)
                    plot(imagePoints_hat(:,1),imagePoints_hat(:,2),'r*');
                else
                    j = sum(imagesUsed(1:i));
                    plot(cameraParamsIntrinsic{cam}.ReprojectedPoints(:,1,j),cameraParamsIntrinsic{cam}.ReprojectedPoints(:,2,j),'b*');
                end                    
                title(imageFileNames{i})
                hold off;            

                subplot(1,2,2);
                imshow(imUndist,[]); hold on;
                plot(imagePointsUndist(:,1),imagePointsUndist(:,2),'r*');
                pause(2);
            end
        end

        fig = figure(6); set(fig,'units','normalized','outerposition',[0 0 1 1]);
        subplot(2,2,cam);
        bar(reprojectionErrors); hold on ;
        plot([0,length(reprojectionErrors)+1],[mean(reprojectionErrors),mean(reprojectionErrors)]); hold off;
        legend(sprintf('Overall Mean Error: %0.3f pixels',mean(reprojectionErrors)));
        axis([0,length(reprojectionErrors)+1,0,max(reprojectionErrors)+0.5]);
        xlabel('images');
        ylabel('Mean Error in Pixels');
        title(sprintf('Camera %d',cam-1));
        
        fprintf('    done!\n')

    end
end

clearvars -except inputPath outputPath squareSize nCams boardSizeChessSmall ...
    boardSizeChessLarge ImageSize processCams collectPointsForIntrinsicCalibration ...
    collectPointsForExtrinsicCalibration estIntrinsicParams showReprojections ...
    estExtrinsicParams evalCalibration cameraParamsIntrinsic estimationErrorsIntrinsic...
    cameraParamsExtrinsic cameraMatrices error2D error3D
    

%% estimate extrinsic camera parameters of the cameras

if estExtrinsicParams
    
    cameraParamsExtrinsic = cell(nCams,1);
    cameraMatrices = cell(nCams,1);

    for cam = processCams

        fprintf('Estimating extrinsic parameters of camera %d ...\n', cam-1);

        % collect all image-points and world-points
        allImagePoints = [];
        allWorldPoints = [];
        directory = sprintf('%sCamera_%d/extrinsic',inputPath, cam-1);
        file = sprintf('%s/calPointsExtrinsic.mat',directory);
        load(file); 
        for i = 1:length(calPointsExtrinsic.imageFileNames)
            % only use every second point for calibration and leave the others for evaluation
            temp = calPointsExtrinsic.imagePoints{i};
            allImagePoints = cat(1, allImagePoints, temp(1:2:length(temp),:));
            temp = calPointsExtrinsic.worldPoints{i};
            allWorldPoints = cat(1, allWorldPoints, temp(1:2:length(temp),:));
        end
        
        % undistort image-points and discard points that do not lie within the valid region
        [ allImagePointsUndist, pointValid ] = removeRadialDistortion(allImagePoints, ...
                                                                      cameraParamsIntrinsic{cam}.IntrinsicMatrix.',...
                                                                      cameraParamsIntrinsic{cam}.RadialDistortion);
        allImagePoints = allImagePoints(pointValid,:);
        allImagePointsUndist = allImagePointsUndist(pointValid,:);
        allWorldPoints = allWorldPoints(pointValid,:);
        
        % Estimate extrinsic parameters
        [rotMatrix_oc, origin_c, inlierIdx, status] = estimateWorldCameraPose(allImagePointsUndist, ...
                                                                              allWorldPoints, ...
                                                                              cameraParamsIntrinsic{cam},...
                                                                              'MaxNumTrials', 10000);
        origin_c = origin_c.';
        transVec_oc = -rotMatrix_oc * origin_c;
        K = cameraParamsIntrinsic{cam}.IntrinsicMatrix.';
        P = K*[rotMatrix_oc, transVec_oc];

        % Calculate reprojection errors
        [ allImagePoints_hat, ~ ] = myWorldToImage(cameraParamsIntrinsic{cam}, ...
                                                   rotMatrix_oc, origin_c, allWorldPoints );
        reprojectionError = sqrt(sum((allImagePoints_hat-allImagePoints).^2,2));

        % collect estimated parameters
        cameraParamsExtrinsic{cam} = struct('RotationMatrix', rotMatrix_oc, ...
                                            'TranslationVector', transVec_oc, ...
                                            'location', origin_c, ...
                                            'ReprojectionErrors', reprojectionError);
        cameraMatrices{cam} = P;
        
        fprintf('    done!\n')

    end

    % save estimated parameters
    fprintf('Save Extrinsic Parameters...\n')
    file = sprintf('%sextrinsicParams.mat',outputPath);
    save(file, 'cameraParamsExtrinsic');    
    fprintf('    done!\n')

    fprintf('Save Camera Matrices...\n')
    file = sprintf('%scameraMatrices.mat',outputPath);
    save(file, 'cameraMatrices');    
    fprintf('    done!\n')
    
else
        
    % load estimated extrinsic parameters
    fprintf('Load Extrinsic Parameters...\n')
    file = sprintf('%sextrinsicParams.mat',outputPath);
    load(file);    
    fprintf('    done!\n') 
    
    fprintf('Load Camera Matrices...\n')
    file = sprintf('%scameraMatrices.mat',outputPath);
    load(file);    
    fprintf('    done!\n') 
end

for cam = processCams
    
        % show camera pose 
        fig = figure(7); set(fig,'units','normalized','outerposition',[0 0 1 1]);
        hold on;
        workspace = [0, 770; 0, 310; 0, 200];
        plot3([workspace(1,1) workspace(1,2) workspace(1,2) workspace(1,1) workspace(1,1) workspace(1,1) workspace(1,2) workspace(1,2) workspace(1,2) workspace(1,2) workspace(1,2) workspace(1,2) workspace(1,1) workspace(1,1) workspace(1,1) workspace(1,1)],...
              [workspace(2,1) workspace(2,1) workspace(2,2) workspace(2,2) workspace(2,1) workspace(2,1) workspace(2,1) workspace(2,1) workspace(2,1) workspace(2,2) workspace(2,2) workspace(2,2) workspace(2,2) workspace(2,2) workspace(2,2) workspace(2,1)],...
              [workspace(3,1) workspace(3,1) workspace(3,1) workspace(3,1) workspace(3,1) workspace(3,2) workspace(3,2) workspace(3,1) workspace(3,2) workspace(3,2) workspace(3,1) workspace(3,2) workspace(3,2) workspace(3,1) workspace(3,2) workspace(3,2)],...
              '-*','Color',[0.49,0.18,0.56]); hold on;
        plotCamera('Location', cameraParamsExtrinsic{cam}.location,...
                   'Orientation', cameraParamsExtrinsic{cam}.RotationMatrix, ...
                   'Label', sprintf('cam %d',cam-1) ,'AxesVisible',true,...
                   'Color',[0,0,1],'Size',10); hold off;
        title('Camera Positions');
        grid on;
        axis equal;
        axis([-100 900 -150 450 0 300]);
        xlabel('x [mm]');
        ylabel('y [mm]');
        zlabel('z [mm]');
        
end

clearvars -except inputPath outputPath squareSize nCams boardSizeChessSmall ...
    boardSizeChessLarge ImageSize processCams collectPointsForIntrinsicCalibration ...
    collectPointsForExtrinsicCalibration estIntrinsicParams showReprojections ...
    estExtrinsicParams evalCalibration cameraParamsIntrinsic estimationErrorsIntrinsic...
    cameraParamsExtrinsic cameraMatrices error2D error3D

%% Evaluate Calibration

if evalCalibration
    
    error3D = {};
    error2D = {};
    location3D = {};
    location2D = {};
    
    for cam = processCams

        fprintf('Evaluate calibration for camera %d ...\n', cam-1);

        % collect all image-points and world-points
        allImagePoints = [];
        allWorldPoints = [];
        directory = sprintf('%sCamera_%d/extrinsic',inputPath, cam-1);
        file = sprintf('%s/calPointsExtrinsic.mat',directory);
        load(file); 
        for i = 1:length(calPointsExtrinsic.imageFileNames)
            % use every second point for evaluation
            temp = calPointsExtrinsic.imagePoints{i};
            allImagePoints = cat(1, allImagePoints, temp(2:2:length(temp),:));
            temp = calPointsExtrinsic.worldPoints{i};
            allWorldPoints = cat(1, allWorldPoints, temp(2:2:length(temp),:));
        end
        
        % undistort image-points and discard points that do not lie within the valid region
        [ allImagePointsUndist, pointValid ] = removeRadialDistortion(allImagePoints, ...
                                                                      cameraParamsIntrinsic{cam}.IntrinsicMatrix.',...
                                                                      cameraParamsIntrinsic{cam}.RadialDistortion);
        allImagePoints = allImagePoints(pointValid,:);
        allImagePointsUndist = allImagePointsUndist(pointValid,:);
        allWorldPoints = allWorldPoints(pointValid,:);
                
        % Calculate the minimal Distance between the world-point and the
        % ray generated by the back-projection of the image-point
        N = length(allWorldPoints);
        P = cameraMatrices{cam};
        M = P(:,1:3);
        camCenter = -M^(-1)*P(:,4);
        D = M^(-1)*[allImagePointsUndist,ones(N,1)].';
        
        allWorldPoints_hat = zeros(size(allWorldPoints));
        for i=1:N
            mu = ( D(:,i).'*(allWorldPoints(i,:).'-camCenter) )/( D(:,i).'*D(:,i) );
            allWorldPoints_hat(i,:) = ( camCenter + mu*D(:,i) ).';
        end
        error3D{cam} = sqrt(sum((allWorldPoints_hat-allWorldPoints).^2,2));
        
        % calculate reprojection error
        [ allImagePoints_hat, ~ ] = myWorldToImage(cameraParamsIntrinsic{cam}, ...
                                                   cameraParamsExtrinsic{cam}.RotationMatrix, ...
                                                   cameraParamsExtrinsic{cam}.location,...
                                                   allWorldPoints );
        error2D{cam} = sqrt(sum((allImagePoints_hat-allImagePoints).^2,2));
      
        % collect location of image-points and world-points
        location3D{cam} = allWorldPoints;
        location2D{cam} = allImagePoints;
        
        fprintf('    done!\n')
        
    end
    

    % save evaluation-results
    fprintf('Save Evaluation-Results...\n')
    file = sprintf('%scalibrationResults.mat',outputPath);
    save(file, 'error2D', 'error3D', 'location2D', 'location3D');    
    fprintf('    done!\n')
    
else
        
    % load evaluation-results
    fprintf('Load Evaluation-Results...\n')
    file = sprintf('%scalibrationResults.mat',outputPath);
    load(file);    
    fprintf('    done!\n') 
    
end

for cam = processCams

    % show histogram of errors in 2D
    fig = figure(8); set(fig,'units','normalized','outerposition',[0 0 1 1]);
    subplot(2,2,cam);
    histogram(error2D{cam},0:0.25:25,'Normalization','probability');
    grid on;
    xlabel('\Deltax [pixel]');
    title(sprintf('Errors 2D cam %d: m = %f, std = %f', cam, mean(error2D{cam}), std(error2D{cam})));

    % show histogram of errors in 3D
    fig = figure(9); set(fig,'units','normalized','outerposition',[0 0 1 1]);
    subplot(2,2,cam);
    histogram(error3D{cam},0:0.25:10,'Normalization','probability');
    grid on;
    xlabel('\Deltax [mm]');
    title(sprintf('Errors 3D cam %d: m = %f, std = %f', cam, mean(error3D{cam}), std(error3D{cam})));

    % show 3D errors with respect to their location on the image
    limit = 6;
    color = NaN*size(error3D{cam});
    color(error3D{cam} >= limit) = 10;
    for i=1:limit
        ind = (error3D{cam} < i) & (error3D{cam} >= (i-1));
        color(ind) = i;
    end
    fig = figure(10); set(fig,'units','normalized','outerposition',[0 0 1 1]);
    subplot(2,2,cam);
    scatter3(location2D{cam}(:,1),location2D{cam}(:,2),error3D{cam},20,color,'o','filled');
    grid on;
    colormap jet;
    axis([0, ImageSize(2)+1, 0, ImageSize(1)+1]);
    caxis([0 10])
    title(sprintf('3D-Errors of Camera %d',cam-1));
    colorbar;
    
end

clearvars -except inputPath outputPath squareSize nCams boardSizeChessSmall ...
    boardSizeChessLarge ImageSize processCams collectPointsForIntrinsicCalibration ...
    collectPointsForExtrinsicCalibration estIntrinsicParams showReprojections ...
    estExtrinsicParams evalCalibration cameraParamsIntrinsic estimationErrorsIntrinsic...
    cameraParamsExtrinsic cameraMatrices error2D error3D location2D location3D