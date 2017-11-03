function [ assignment, cost ] = myMunkres( costMat )
%MYMUNKRES 
% https://www.youtube.com/watch?v=LDmOgQzZEOc

% costMat = [1 2 3 4; 2 4 6 8;3 6 9 12;4 8 12 16];
% costMat = [0 0 0 2 2; 3 0 5 2 1; 2 0 0 0 0; 9 4 7 0 1; 0 1 7 0 0];
% costMat = [3 4 3 5 4; 3 1 5 2 0; 5 4 3 3 2; 10 6 8 1 1; 3 5 10 3 2];
n = max(size(costMat));
cm = costMat;

% Step 1: Reduce the arry by both row and column substractions
cm = cm - repmat(min(cm,[],2),1,n);
cm = cm - repmat(min(cm,[],1),n,1);

% Step 2 + 3:
nLines = 0;
while nLines < n
    % Step 2: Cover the zero elements with the minimum number of lines. 
    % If this minimum number is same as the size of the array, go to step 4
    coverageMat = zeros(size(cm));
    assignMat = false(size(cm));
    nLines = 0;
    zeroMat = ~cm;
    while any(zeroMat(:))
        nZerosCol = sum(zeroMat,1);
        [minCol,indCol] = min(nZerosCol + n^2*(sum(assignMat,1) | ~nZerosCol));
        nZerosRow = sum(zeroMat,2);
        [minRow,indRow] = min(nZerosRow + n^2*(sum(assignMat,2) | ~nZerosRow));
        if minCol < minRow
            % look for the zero in the column and get its row index
            indRow = find(zeroMat(:,indCol),1);
            % take its row
            zeroMat(indRow,:) = false;
            coverageMat(indRow,:) = coverageMat(indRow,:) + 1;
        else
            % look for the zero in the row and get its column index 
            indCol = find(zeroMat(indRow,:),1);
            % take its column
            zeroMat(:,indCol) = false;
            coverageMat(:,indCol) = coverageMat(:,indCol) + 1;
        end
        assignMat(indRow,indCol) = true;
        nLines = nLines + 1;
    end

    if (nLines < n)
        % Step 3: Let m be the minimum uncovered element. The array is 
        % augmented by reducing all uncoverd elements by m and increasing 
        % all elements coverd by two lines my m. Return to step 2.
        uncoverdValues = cm(coverageMat==0);
        m = min(uncoverdValues(:));
        cm(coverageMat==0) = cm(coverageMat==0) - m;
        cm(coverageMat==2) = cm(coverageMat==2) + m;
    elseif (nLines == n)
        % Step 4: Make the optimal final assignment
        pause(1);
        assignment = assignMat;
        cost = sum(sum(costMat(assignMat)));
    else
        error('help, something went teribliy wrong!!!');
    end
    
end


end

