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
matTOcsvf("/media/hhofmann/dgx/data_hhofmann/Data/indexfinger_right/9000/",4)
function matTOcsvf(origin_path, num_of_cams)


    for camNr = 1:num_of_cams
        path = strcat(origin_path, "Camera_", num2str(camNr-1), "/UV_Bin/fingers.mat")
        load(path, '-mat')

        path = strcat(origin_path, "Camera_",num2str(camNr-1),"/UV_Bin/validData.csv")
        B = []
        csvwrite(path,B);
        fid = fopen(path,"w")
        nrOfValidationFingers = cast(length(fingers)/10,'uint32')
        for i = (1:nrOfValidationFingers)
            %if more than one finger was detected
            [k,n] = size(fingers{i}.radii)
            if k>=2
                picName_new     = fingers{i}.picName
                x_coord_new     = fingers{i}.centers(1,1)
                y_coord_new     = fingers{i}.centers(2,1)
                bb_diameter_new = 2*fingers{i}.radii(1)
                P_new           = 1
            else 
                %if no finger is detected
                if isempty(fingers{i}.centers)
                    picName_new     = fingers{i}.picName
                    x_coord_new     = 0
                    y_coord_new     = 0
                    bb_diameter_new = 0
                    P_new           = 0
                else
                    %if exactli one finger is detected
                    picName_new     = fingers{i}.picName
                    x_coord_new     = fingers{i}.centers(1)
                    y_coord_new     = fingers{i}.centers(2)
                    bb_diameter_new = 2*fingers{i}.radii
                    P_new           = 1
                end        
            end
            %print this info as line in the csv
            fprintf(fid, '%s,',picName_new);
            fprintf(fid, '%f,',x_coord_new);
            fprintf(fid, '%f,',y_coord_new);
            fprintf(fid, '%f,',bb_diameter_new);
            fprintf(fid, '%f,',P_new);
            fprintf(fid, '\n')
        end
        
        
        path = strcat(origin_path, "Camera_",num2str(camNr-1),"/UV_Bin/trainData.csv")
        B = []
        csvwrite(path,B);
        fid = fopen(path,"w")
        if length(fingers)<11
            nrOfValidationFingers=1
        end
        for i = nrOfValidationFingers:length(fingers)
            %if more than one finger was detected
            [k,n] = size(fingers{i}.radii)
            if k>=2
                picName_new     = fingers{i}.picName
                x_coord_new     = fingers{i}.centers(1,1)
                y_coord_new     = fingers{i}.centers(2,1)
                bb_diameter_new = 2*fingers{i}.radii(1)
                P_new           = 1
            else 
                %if no finger is detected
                if isempty(fingers{i}.centers)
                    picName_new     = fingers{i}.picName
                    x_coord_new     = 0
                    y_coord_new     = 0
                    bb_diameter_new = 0
                    P_new           = 0
                else
                    %if exactli one finger is detected
                    picName_new     = fingers{i}.picName
                    x_coord_new     = fingers{i}.centers(1)
                    y_coord_new     = fingers{i}.centers(2)
                    bb_diameter_new = 2*fingers{i}.radii(1)
                    P_new           = 1
                end        
            end
            %print this info as line in the csv
            fprintf(fid, '%s,',picName_new);
            fprintf(fid, '%f,',x_coord_new);
            fprintf(fid, '%f,',y_coord_new);
            fprintf(fid, '%f,',bb_diameter_new);
            fprintf(fid, '%f,',P_new);
            fprintf(fid, '\n')
        end
    end
end