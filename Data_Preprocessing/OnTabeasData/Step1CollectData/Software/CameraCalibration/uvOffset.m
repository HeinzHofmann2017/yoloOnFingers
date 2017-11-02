%
%   Skript for checking how big the difference
%   is, between the refraction of normal light 
%   and the refraction of uv light.
%
%   tmendez, 18.05.2017
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5

clear all;
close all;
clc;

addpath(genpath('../matlabHelperFunctions'));

%% Parameters

inputPath = './calibrationPics/UV_Offset';

squareSize = 18;  % in units of 'mm'
nCams = 4;

if inputPath(end) ~= '/'
    inputPath(end+1) = '/';
end

%% find checkerboards in normal images and uv images and calculate distances

screensize = get( groot, 'Screensize' );
distW_mm = [];
distU_mm = [];
dist_px = [];
numImag = 0;

for cam = 1:nCams
    
    % Get images to process
    directory = sprintf('%sCamera_%d/',inputPath, cam-1);
    [ imageFileNamesW, imageFileNamesU, ~ ] = getImageFileNames( directory );

    for j=1:length(imageFileNamesW)
        close all;
        
        % load images
        imW = double(imread(imageFileNamesW{j}));
        imU = double(imread(imageFileNamesU{j}));
        imW = histeq(imW/max(imW(:)),2^8);
        imU = imU/max(imU(:));
        
        % Detect checkerboards in images
        [imagePointsW, boardSizeW, ~] = detectCheckerboardPoints(imW);
        [imagePointsU, boardSizeU, ~] = detectCheckerboardPoints(imU);
        if ( any(~boardSizeW) || any(~boardSizeU) )
            fprintf(['No checkerboard detected.\n'...    
                     '    Discarded %s ...\n    Discarded %s ...\n'], ...
                     imageFileNamesW{j},imageFileNamesU{j});
            continue;
        end
        
        % show detected checkerboards
        f1 = figure(1);
        ax1 = axes('Parent',f1);
        imshow(imW,[]); hold(ax1,'on');
        set(f1, 'Position', [screensize(1) screensize(2) screensize(3)/2 screensize(4)])
        if ~isempty(imagePointsW)
            plot(ax1, imagePointsW(:,1),imagePointsW(:,2),'r*');
            plot(ax1, imagePointsW(1,1),imagePointsW(1,2),'g*');
        end
        hold(ax1,'off');
        
        f2 = figure(2);
        ax2 = axes('Parent',f2);
        imshow(imU,[]); hold(ax2,'on');
        set(f2, 'Position', [screensize(3)/2 screensize(2) screensize(3)/2 screensize(4)])
        if ~isempty(imagePointsU)
            plot(ax2, imagePointsU(:,1),imagePointsU(:,2),'r*'); 
            plot(ax2, imagePointsU(1,1),imagePointsU(1,2),'g*');
        end
        hold(ax2,'off');
        
        % make sure both checkerboards have the same origin
        figure(f1); pause(0.01);
        [x0, y0, b] = ginput(1);
        if ( b ~= ' ' )
            [ imagePointsW, boardSizeW ] = changeCheckerboardOrigin( x0, y0, imagePointsW, boardSizeW );
            if any(~boardSizeW)
                fprintf(['No checkerboard detected.\n'...    
                         '    Discarded %s ...\n    Discarded %s ...\n'], ...
                         imageFileNamesW{j},imageFileNamesU{j});
                continue;
            end
        end
        imagePointsW = reshape(imagePointsW,boardSizeW(1)-1,boardSizeW(2)-1,2);
        
        figure(f2); pause(0.01);
        [x0, y0, b] = ginput(1);
        if ( b ~= ' ' )
            [ imagePointsU, boardSizeU ] = changeCheckerboardOrigin( x0, y0, imagePointsU, boardSizeU );
            if any(~boardSizeU)
                fprintf(['No checkerboard detected.\n'...    
                         '    Discarded %s ...\n    Discarded %s ...\n'], ...
                         imageFileNamesW{j},imageFileNamesU{j});
                continue;
            end
        end
        imagePointsU = reshape(imagePointsU,boardSizeU(1)-1,boardSizeU(2)-1,2);
        
        % make sure both checkerboards have the same size
        boardsize = min([boardSizeW; boardSizeU]);
        imagePointsW = imagePointsW(1:boardsize(1)-1,1:boardsize(2)-1,:);
        imagePointsU = imagePointsU(1:boardsize(1)-1,1:boardsize(2)-1,:);
        
        % Calculate distances between detected points in "white-image" and "uv-image"
        dist_px_temp = sqrt(sum((imagePointsU - imagePointsW).^2,3));
        dist_px_temp = reshape(dist_px_temp,[],1);

        distW_px = distancesCheckerboardPoints( imagePointsW, boardSizeW );
        mmPerPixW = squareSize/mean(distW_px);

        distU_px = distancesCheckerboardPoints( imagePointsU, boardSizeU );
        mmPerPixU = squareSize/mean(distU_px);

        distW_mm = [distW_mm; dist_px_temp*mmPerPixW];
        distU_mm = [distU_mm; dist_px_temp*mmPerPixU];
        dist_px = [dist_px ; dist_px_temp];

        % mark used checkerboard-points
        imagePointsW = reshape(imagePointsW,[],2);
        hold(ax1,'on');
        plot(ax1, imagePointsW(:,1),imagePointsW(:,2),'b*'); 
        plot(ax1, imagePointsW(1,1),imagePointsW(1,2),'g*');
        hold(ax1,'off');
        
        imagePointsU = reshape(imagePointsU,[],2);
        hold(ax2,'on');
        plot(ax2, imagePointsU(:,1),imagePointsU(:,2),'b*'); 
        plot(ax2, imagePointsU(1,1),imagePointsU(1,2),'g*');
        hold(ax2,'off');
        pause(1);
        numImag = numImag+1
    end

end

%% Results
figure;
hist([distW_mm,distU_mm],0:0.1:max(max(distW_mm),max(distU_mm)));
grid on;
figure;
hist(dist_px,50);
grid on;
fprintf('"Difference in Pixels"\n');
fprintf('    Mean Error:         %f mm\n' ,mean(dist_px));
fprintf('    Standard Deviation: %f mm\n' ,std(dist_px));
fprintf('Difference in "white-image"-mm\n');
fprintf('    Mean Error:         %f mm\n' ,mean(distW_mm));
fprintf('    Standard Deviation: %f mm\n' ,std(distW_mm));
fprintf('Difference in "uv-image"mm\n');
fprintf('    Mean Error:         %f mm\n' ,mean(distU_mm));
fprintf('    Standard Deviation: %f mm\n' ,std(distU_mm));

save(sprintf('%suv_offset.mat',inputPath), 'distW_mm','distU_mm', 'dist_px');

