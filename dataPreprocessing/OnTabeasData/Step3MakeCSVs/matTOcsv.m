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
matTOcsvf("/home/hhofmann/Schreibtisch/Data/9000/",4)
function matTOcsvf(origin_path, num_of_cams)


    for picNr = 1:num_of_cams
        path = strcat(origin_path, "Camera_", num2str(picNr-1), "/UV_Bin/fingers.mat")
        load(path, '-mat')

        path = strcat(origin_path, "Camera_",num2str(picNr-1),"/UV_Bin/fingersValid.csv")
        B = []
        csvwrite(path,B);
        fid = fopen(path,"w")
        nrOfValidationFingers = cast(length(fingers)/10,'uint32')
        for i = (1:nrOfValidationFingers)
            %if more than one finger was detected
            [k,n] = size(fingers{i}.radii)
            if k>=2
                picName_new = fingers{i}.picName
                x_coord_new = fingers{i}.centers(1,1)
                y_coord_new = fingers{i}.centers(2,1)
                width_new   = 2*fingers{i}.radii(1)
                height_new  = 2*fingers{i}.radii(1)
                C_new       = 0.99
                P_new       = 0.99
            else 
                %if no finger is detected
                if isempty(fingers{i}.centers)
                    picName_new = fingers{i}.picName
                    x_coord_new = 0
                    y_coord_new = 0
                    width_new   = 0
                    height_new  = 0
                    C_new       = 0
                    P_new       = 0
                else
                    %if exactli one finger is detected
                    picName_new = fingers{i}.picName
                    x_coord_new = fingers{i}.centers(1)
                    y_coord_new = fingers{i}.centers(2)
                    width_new   = 2*fingers{i}.radii
                    height_new  = 2*fingers{i}.radii
                    C_new       = 0.99
                    P_new       = 0.99
                end        
            end
            %print this info as line in the csv
            fprintf(fid, '%s,',picName_new);
            fprintf(fid, '%f,',x_coord_new);
            fprintf(fid, '%f,',y_coord_new);
            fprintf(fid, '%f,',width_new);
            fprintf(fid, '%f,',height_new);
            fprintf(fid, '%f,',C_new);
            fprintf(fid, '%f,',P_new);
            fprintf(fid, '\n')
        end
        
        
        path = strcat(origin_path, "Camera_",num2str(picNr-1),"/UV_Bin/fingersTrain.csv")
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
                picName_new = fingers{i}.picName
                x_coord_new = fingers{i}.centers(1,1)
                y_coord_new = fingers{i}.centers(2,1)
                width_new   = 2*fingers{i}.radii(1)
                height_new  = 2*fingers{i}.radii(1)
                C_new       = 0.99
                P_new       = 0.99
            else 
                %if no finger is detected
                if isempty(fingers{i}.centers)
                    picName_new = fingers{i}.picName
                    x_coord_new = 0
                    y_coord_new = 0
                    width_new   = 0
                    height_new  = 0
                    C_new       = 0
                    P_new       = 0
                else
                    %if exactli one finger is detected
                    picName_new = fingers{i}.picName
                    x_coord_new = fingers{i}.centers(1)
                    y_coord_new = fingers{i}.centers(2)
                    width_new   = 2*fingers{i}.radii
                    height_new  = 2*fingers{i}.radii
                    C_new       = 0.99
                    P_new       = 0.99
                end        
            end
            %print this info as line in the csv
            fprintf(fid, '%s,',picName_new);
            fprintf(fid, '%f,',x_coord_new);
            fprintf(fid, '%f,',y_coord_new);
            fprintf(fid, '%f,',width_new);
            fprintf(fid, '%f,',height_new);
            fprintf(fid, '%f,',C_new);
            fprintf(fid, '%f,',P_new);
            fprintf(fid, '\n')
        end
    end
end