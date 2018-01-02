%
%   skript to find fingertips in UV-images.
%   Found fingertips are marked in a binary image
%   are saved.
%
%   tmendez, 01.07.2017
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


clear all;
close all;
clc;

addpath(genpath('../matlabHelperFunctions'));

%% Parameters

inputPath = '/home/hhofmann/Schreibtisch/Data/9000/';
calcBackgroundIm = false;               % if false, default-BG is loaded
circleFindMethod = 'SKEL';              % method to use for finding circles
                                        %   'CHT':  Circular Hough Transform 
                                        %   'SKEL': get skeleton by erosion
separateOverlappingCircles = false;     % if true, overlapping circles are separated with the watershed-algo
detectFingersOrSpheres = 'f';           % 'f' for fingertips, 's' for spheres

if detectFingersOrSpheres == 'f'
    % parameters for fingertip-finder
    circleFinderSensitivity = 0.925;    % sensitivity of circle finder
    minFilledRatio = 0.4;               % minimum filled ratio of circle to be kept
    thresholdSmallLargeOverlap = 0.5;   % threshold to decide between large and small overlap
    minDiffFilledRatioByOverlap = 0.10; % threshold to decide between large and small difference regarding to the filledRatio
else
    % parameters for sphere-finder
    circleFinderSensitivity = 0.9;      % sensitivity of circle finder
    minFilledRatio = 0.5;               % minimum filled ratio of circle to be kept
    thresholdSmallLargeOverlap = 0.5;   % threshold to decide between large an small overlap
    minDiffFilledRatioByOverlap = 0.10; % threshold to decide between large and small difference regarding to the filledRatio
end

nCams = 4;
ImageSize = [960, 1280];
imRes = 2^8-1;

if inputPath(end) ~= '/'
    inputPath(end+1) = '/';
end


%% Process UV-Images
for cam = 1:nCams

    % create directory to save the images
    outputPathBin = sprintf('%sCamera_%d/UV_Bin/',inputPath,cam-1);
    if ~exist(outputPathBin, 'dir')
        mkdir(outputPathBin);
    end
    outputPathRGB = sprintf('%sCamera_%d/markedFingers/',inputPath,cam-1);
    if ~exist(outputPathRGB, 'dir')
        mkdir(outputPathRGB);
    end
    
    % find all images in directory
    [ ~, ~,imageFileNames ] = getImageFileNames( sprintf('%sCamera_%d/UV/',inputPath,cam-1) );
    N = length(imageFileNames);
    if calcBackgroundIm
        % Calculate Background as mean of all images
        fprintf('Calculate background of camera %d ...\n',cam-1);
        BG = zeros(ImageSize);
        for i = 1:N
            picName = sprintf('%sCamera_%d/UV/pic%d.png',inputPath,cam-1,i-1);
            if ( exist(picName, 'file') == 2 )
                % file exists -> read in 
                BG = BG + double(imread(picName));
            else
                % file does not exist
                error(sprintf('File %s does not exist!',picName));
            end
        end
        BG = BG/N;
        fprintf('    Done...\n');
    else
        % laod default background-image
        BG = double(imread(sprintf('../FingerTracker/BG_Cam%d.png',cam-1)));
    end
    imwrite(BG/imRes,sprintf('%sBackground.png',outputPathBin));

    % process every image
    fprintf('Detect fingertips in camera %d ...\n',cam-1);
    fingers = {};
    S=zeros(50);
    for i=1:N
        clearvars -except inputPath outputPathBin outputPathRGB calcBackgroundIm ...
            separateOverlappingCircles detectFingersOrSpheres circleFinderSensitivity ...
            minFilledRatio thresholdSmallLargeOverlap minDiffFilledRatioByOverlap ...
            nCams ImageSize imRes BG N cam i fingers circleFindMethod
        
        % read in image
        I = double(imread(sprintf('%sCamera_%d/UV/pic%d.png',inputPath,cam-1,i-1)));
        
        % subtract background and normalize intentityvalues
        D = I - BG;
        %Delete picture if it is too bright(value 5500000 is found with
        %evaluation of about 200 pictures)wrote by Heinz
        if sum(sum(D)) > -5510000;
            %Do nothing with too bright Pictures!!!!!! wrote by Heinz
            fprintf('didnt take into account: Camera_%d pic%d.png \n',cam-1,i-1)
        else

            D = (D+imRes)/(2*imRes);


            % increase contrast
            D(D <= 0.5) = 0.5;
            D = (D-0.5)*2;

            % threshold image
            thresh = graythresh(D);
            thresh = max(thresh,0.1);
            I_bin = (D > thresh);

            % fill small holes
            I_bin = imclose(I_bin,strel('disk',2,0));

            % ged rid of very small connected components
            I_bin = imopen(I_bin,strel('disk',2,0));

            % find circles
            if strcmp(circleFindMethod,'CHT')   % with Circular-Hough-Transform-Method

                % determine radius-range to search in
                conComp = bwconSchreibtischncomp(I_bin);
                r = sqrt(cellfun(@length, conComp.PixelIdxList)/pi);
                rmin = floor(max(min(r)-10,6));
                rmax = ceil(min(max(r)+10,120));
                if isempty(rmin)
                    rmin = 6;
                    rmax = 10;
                end
                range = [rmin:10:rmax, rmax];
                range = [range(1), range(2:end-1);
                         range(2:end-1)+1, range(end) ].';

                % find circles
                centers = [];
                radii = [];
                for j = 1:size(range,1)
                    [tempCenters, tempRadii] = imfindcircles(D,range(j,:),...
                                                             'ObjectPolarity','bright',...
                                                             'Sensitivity',0.925,...
                                                             'Method','TwoStage');
                    centers = cat(1,centers,tempCenters);
                    radii = cat(1,radii,tempRadii);
                end

            elseif strcmp(circleFindMethod,'SKEL')  % with Skeleton-By-Erosion-Method

                [ centers, radii ] = findCircles( I_bin );

            else
                error('No Valid Method for searching circles in image.')
            end

            %For Heinz's Trainingset max one circle in the picture is valid
            %==> Delete all invalid Circles:
            [centers,radii] = deleteFalseCircles(centers, radii);

    %         % show found circles
    %         figure(1)
    %         subplot(1,2,1)
    %         imshow(cat(3,I_bin,D,zeros(size(I_bin))),[]); hold on;
    %         viscircles(centers,radii); hold off;

            % only keep circles that are at least minFilledRatio % filled
            nCirc = length(radii);
            I_circ = false(ImageSize(1),ImageSize(2),nCirc);
            [X,Y] = meshgrid(1:ImageSize(2),1:ImageSize(1));
            for j = 1:nCirc
                I_circ(:,:,j) = (X-centers(j,1)).^2+(Y-centers(j,2)).^2 <= radii(j)^2;
            end
            A = sum(sum(I_circ,1),2);
            filledRatio = squeeze(sum(sum(repmat(I_bin,1,1,nCirc).*I_circ,1),2)./A);
            ind = filledRatio > minFilledRatio;
            I_circ = I_circ(:,:,ind);
            A = squeeze(A(ind));
            filledRatio = filledRatio(ind);
            centers = centers(ind,:);
            radii = radii(ind);
            nCirc = length(radii);

            % find overlapping circles
            discard = false(nCirc,1);
            for j = 1:nCirc
                for k = j+1:nCirc
                    dilOverlap = sum(sum( imdilate(I_circ(:,:,j),strel('rectangle',[3,3])) & ...
                                          imdilate(I_circ(:,:,k),strel('rectangle',[3,3])) ));
                    overlap = sum(sum( I_circ(:,:,j) & I_circ(:,:,k) ));
                    if dilOverlap
                        if ( (overlap/A(j) < thresholdSmallLargeOverlap) && ...
                             (overlap/A(k) < thresholdSmallLargeOverlap) )
                            % overlap is small 
                            % -> keep both and separate circles if required
                            if separateOverlappingCircles
                                C = false([ImageSize,2]);
                                figure;
                                C(round(centers(j,2)),round(centers(j,1)),1) = 1;
                                C(round(centers(k,2)),round(centers(k,1)),2) = 1;
                                nOld = 0;
                                n = sum(C(:));
                                while n > nOld
                                    nOld = n;
                                    C(:,:,1) = imdilate(C(:,:,1),strel('rectangle',[3,3]));
                                    C(:,:,2) = imdilate(C(:,:,2),strel('rectangle',[3,3]));
                                    boarder = C(:,:,1) & C(:,:,2);
                                    C(:,:,1) = C(:,:,1) & I_circ(:,:,j) & ~boarder;
                                    C(:,:,2) = C(:,:,2) & I_circ(:,:,k) & ~boarder;
                                    n = sum(C(:));
                                end
                                I_circ(:,:,j) = C(:,:,1);
                                I_circ(:,:,k) = C(:,:,2);
                            end
                        else
                            % overlap is large 
                            if ( abs(filledRatio(j)-filledRatio(k)) > minDiffFilledRatioByOverlap )
                                % difference in filledRatio is large
                                % -> discard bader filled one
                                if filledRatio(j) < filledRatio(k)
                                    discard(j) = 1;
                                else
                                    discard(k) = 1;
                                end
                            else
                                % difference in filledRatio is small
                                % -> discard the smaller one
                                if A(j) < A(k)
                                    discard(j) = 1;
                                else
                                    discard(k) = 1;
                                end
                            end
                        end
                    end
                end
            end
            I_circ = I_circ(:,:,~discard);
            A = squeeze(A(~discard));
            centers = centers(~discard,:);
            radii = radii(~discard);
            nCirc = length(radii);
            I_bin = any(I_circ,3);

            % collect centers and radii
            fingers{end+1} = struct('picName',sprintf('pic%d.png',i-1),...
                                    'centers',centers,'radii',radii);

            % save difference-images
            I = I/max(I(:));
            I = insertShape(I,'FilledCircle',[centers,radii],'Color',[181/255,0,0],'Opacity',0.5);
            imwrite(I,sprintf('%spic%d.png',outputPathRGB,i-1));

            % save binary image
            imwrite(I_bin,sprintf('%spic%d.png',outputPathBin,i-1));

            % show found fingers
    %         figure(1)
    %         subplot(1,2,2)
    %         imshow(cat(3,I_bin,D,zeros(size(I_bin))),[]); hold on;
    %         viscircles(centers,radii); hold off;
    %         pause(0.01);
        end
    end
    
    % save centers and radii
    save(sprintf('%sfingers.mat',outputPathBin),'fingers');
    
    fprintf('    Done...\n');
    
end

