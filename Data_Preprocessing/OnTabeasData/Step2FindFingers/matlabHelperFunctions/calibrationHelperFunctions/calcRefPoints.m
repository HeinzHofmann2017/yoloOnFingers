function [ ref_Points, origin_chess, rot_chess, basis_vectors  ] = calcRefPoints( device, position, angle, viewDir, figNr )
%CALCREFPOINTS Function to calculates the absolut position of the 
%   chess-board reference points 
% 
%   Inputs:
%       device:     calibration device that was used:
%                   1:    left vertical <-|->
%                   2:    left horizontal
%                   3:    right vertical <-|->
%                   4:    right horizontal
%
%       position:   position where the fence is locked (0...8)
%
%       angle:      angle of the board
%                   -pi/4:  towards the origin (down/left)
%                       0:  no slope
%                    pi/4:  away of the origin (up/right)
%
%       viewDir:    viewing direction of the board:
%                    1: (in x/y direction)
%                   -1: (in -x/-y direction)
% 
%                     ^ y
%                     |
%                     |
%                     ---->x
% 
%       figNr:      number of figure
% 
%   Outputs:
%       ref_Points:     absolute Position of all reference Points [NY x NX x 3 ]
% 
%       origin_chess:   origion of the chess coordinate system
% 
%       rot_chess:      orientation of the chess coordinate system
%                       (Rotation -Matrix)
% 
%       basis_vectors:  basis vectors of the chess coordinate system
% 

%% check input parameters
if ( device < 1 || device > 4 )
    error('device must lie in the range of 1...4')
end
if ( position < 0 || position > 8 )
    error('position must lie in the range of 1...8')
end
if ( (angle ~= 0) && (angle ~= -pi/4) && (angle ~= pi/4) )
    error('angle must eighter be -pi/4 | 0 | pi/4')
end
if ( (viewDir ~= 1) && (viewDir ~= -1) )
    error('viewDir must eighter be - | 1')
end
if ( figNr < 0 )
    error('figNr must be > 0')
end

%% Geometric dimensions
start_fence = [ 30, 30+380 ];
step_fence = 35;
start_x = 31.9;
step_x = [323 57 323];
start_y = 29.9;
step_y = 247;
start_z = 8.5 + (22.5-7.5/tan(pi/8));
step_z = [7.5/tan(pi/8)+33 150];
offset_board = 10.5;
square_Size = 18;
chess_Size = [ 14, 18 ;
               11, 11 ];

%% Calculate origin of chessboard-pattern

% offset of the rotation center from the global origin 
%   -------|-|-------
%           |
%           |<- Rotation center
%           |
%   -------|-|-------
%
rotCent = zeros(3,4);
    % device 1: left vertical <-|->
    rotCent(1,1) = start_fence(1) + step_fence * (position+1/2);
    rotCent(2,1) = start_y + step_y/2;
    rotCent(3,1) = start_z;

    % device 2: left horizontal
    rotCent(1,2) = start_x + step_x(1)/2;
    rotCent(2,2) = start_fence(1) + step_fence * (position+1/2);
    rotCent(3,2) = start_z;

    % device  3: right vertical <-|->
    rotCent(1,3) = start_fence(2) + step_fence * (position+1/2);
    rotCent(2,3) = start_y + step_y/2;
    rotCent(3,3) = start_z;

    % device 4: right horizontal
    rotCent(1,4) = start_x + step_x(1) + step_x(2) + step_x(3)/2;
    rotCent(2,4) = start_fence(1) + step_fence * (position+1/2); 
    rotCent(3,4) = start_z;

% Offset of chess-board origin from the rotation center
offset_org_chess = zeros(3,4);

    % device 1: left vertical <-|->
    offset_org_chess(1,1) = offset_board * viewDir;
    offset_org_chess(2,1) = chess_Size(1,1)/2*square_Size * viewDir;
    offset_org_chess(3,1) = step_z(1) + step_z(2)/2 - chess_Size(2,1)/2*square_Size;

    % device 2: left horizontal
    offset_org_chess(1,2) = chess_Size(1,2)/2*square_Size * -viewDir;
    offset_org_chess(2,2) = offset_board * viewDir;
    offset_org_chess(3,2) = step_z(1) + step_z(2)/2 - chess_Size(2,2)/2*square_Size;

    % device  3: right vertical <-|->
    offset_org_chess(:,3) = offset_org_chess(:,1);

    % device 4: right horizontal
    offset_org_chess(:,4) = offset_org_chess(:,2);

% basis vectors of chess board
chess_basis_vector_x = [  0,  1,  0,  1 ;
                         -1,  0, -1,  0 ;
                          0,  0,  0,  0 ] * viewDir;

chess_basis_vector_y = [  0,  0,  0,  0 ;
                          0,  0,  0,  0 ;
                          1,  1,  1,  1 ];
                        
chess_basis_vector_z = [ -1,  0, -1,  0 ;
                          0, -1,  0, -1 ;
                          0,  0,  0,  0 ] * viewDir;
                 
% Rotated Basis Vectors
rotMat = [  cos(angle), 0, sin(angle) ;     % rotation around y
                    0 , 1,         0  ;
           -sin(angle), 0, cos(angle) ];
offset_org_chess(:,[1,3])     = rotMat*offset_org_chess(:,[1,3]);
chess_basis_vector_x(:,[1,3]) = rotMat*chess_basis_vector_x(:,[1,3]);
chess_basis_vector_y(:,[1,3]) = rotMat*chess_basis_vector_y(:,[1,3]);
chess_basis_vector_z(:,[1,3]) = rotMat*chess_basis_vector_z(:,[1,3]);

rotMat = [ 1,         0 ,          0  ;     % rotation around -x
           0,  cos(angle), sin(angle) ;
           0, -sin(angle), cos(angle) ];
offset_org_chess(:,[2,4])     = rotMat*offset_org_chess(:,[2,4]);
chess_basis_vector_x(:,[2,4]) = rotMat*chess_basis_vector_x(:,[2,4]);
chess_basis_vector_y(:,[2,4]) = rotMat*chess_basis_vector_y(:,[2,4]);
chess_basis_vector_z(:,[2,4]) = rotMat*chess_basis_vector_z(:,[2,4]);

% origin of chessboard-pattern
origin_chess = rotCent(:,device) + offset_org_chess(:,device);
basis_vectors = [chess_basis_vector_x(:,device), ...
                 chess_basis_vector_y(:,device), ...
                 chess_basis_vector_z(:,device) ];
rot_chess = basis_vectors^(-1);


%% Calculate crossing points of chessboard-pattern

if ( device == 1 || device == 3);   
    % short chessboard
    nx = chess_Size(1,1);
    ny = chess_Size(2,1);
else
    % long chessboard
    nx = chess_Size(1,2);
    ny = chess_Size(2,2);
end

ref_Points = zeros(ny+1,nx+1,3);
x = 0:nx;
y = 0:ny;
for i = 1:length(x)
    for j=1:length(y)
        ref_Points(j,i,:) = origin_chess + x(i)*square_Size*basis_vectors(:,1) + ... 
                                           y(j)*square_Size*basis_vectors(:,2);
    end
end
B1 = [origin_chess,origin_chess+basis_vectors(:,1)*square_Size*10];
B2 = [origin_chess,origin_chess+basis_vectors(:,2)*square_Size*10];
B3 = [origin_chess,origin_chess+basis_vectors(:,3)*square_Size*10];

figure(figNr);
plot3(ref_Points(:,:,1),ref_Points(:,:,2),ref_Points(:,:,3),'ko'); hold on;
plot3(B1(1,:),B1(2,:),B1(3,:),'r-*'); hold on;
plot3(B2(1,:),B2(2,:),B2(3,:),'g-*'); hold on;
plot3(B3(1,:),B3(2,:),B3(3,:),'b-*'); hold on;
hold off;
grid on;
axis equal;
axis([0 800 -150 350 0 300]);
xlabel('x [mm]')
ylabel('y [mm]')
zlabel('z [mm]')


end
