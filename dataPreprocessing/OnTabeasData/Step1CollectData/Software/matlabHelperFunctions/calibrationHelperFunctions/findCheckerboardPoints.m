function [ imagePoints, boardSize ] = findCheckerboardPoints( I, setBoardSize )
%FINDCHECKERBOARDPOINTS trys to find a checkerboard in image I of size 
%       boardSize. If checkerboard is not found, I is smoothed and again
%       searched for the checkerboard.
%
%   Inputs:
%       I:              filename of image, where the checkerboard is searched
%       setBoardSize:   Size of the checkerboard to be found
%                       or [] if the size is not known 
% 
%   Outputs:
%       imagePoints:    detected checkerboard Points [N x 2] (N=nx*ny)
%                       empty, if no checkerboard could be found
%       boardSize:      size of checkerboard, [0, 0] if no checkerboard was 
%                       found
% 
    
[imagePoints, boardSize, ~] = detectCheckerboardPoints(I);

if ( isempty(setBoardSize) || any(boardSize ~= setBoardSize) ) && all(boardSize)

    for f = 0.05:0.05:0.25
        % Chessboard was NOT detected correctly
        % -> try again with smoothed image
        dist = distancesCheckerboardPoints( imagePoints, boardSize );
        h = ones(round(f*mean(dist)));
        h = h/sum(h(:));
        I_Smoothed = filter2(h,I);
        [imagePointsSmoothed, boardSizeSmoothed, ~] = detectCheckerboardPoints(I_Smoothed);

        if isempty(setBoardSize)
            % setBoardSize is empty 
            % -> take the lager checkerboard
            if ( boardSizeSmoothed(1)*boardSizeSmoothed(2) > boardSize(1)*boardSize(2) )
                % checkerboard of smoothed image is larger
                imagePoints = imagePointsSmoothed;
                boardSize = boardSizeSmoothed;
            end
        else
            % setBoardSize is not empty 
            % -> only take the checkerboard, if its size is correct
            if all(boardSizeSmoothed == setBoardSize)
                % Chessboard was detected correctly in smothed image
                imagePoints = imagePointsSmoothed;
                boardSize = boardSizeSmoothed;
                break;
            else
                % Chessboard was NOT detected correctly in smothed image
                if f == 0.25
                    % only return empty arrays, if f ist the last smothing-factor
                    imagePoints = [];
                    boardSize = [0,0];
                end
            end
        end
    end
end


end

