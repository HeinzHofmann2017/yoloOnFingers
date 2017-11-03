%
%   Implementation of the Kalman-Filter for
%   the Finger-Tracker
%
%   tmendez, 11.07.2017
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

classdef KalmanFilter < handle
    
    % The following properties can be set only by class methods
    properties (SetAccess = private)
        T;   % sample period
        x;   % state vector [px; vx; py; vy; pz; vz; r]
        A;   % state transition matrix
        C;   % measurement matrix
        P;   % error covariance matrix
        su;  % standard deviation of the velocity noise
        Qw;  % process noise covariance matrix
        Qv;  % measurement noise covariance matrix
        K;   % Kalman gain matrix
    end
    
    methods
        % Consturctor
        function KFilt = KalmanFilter(T, su)
            
            % initialize the Kalman filter
            KFilt.T = T;
            KFilt.su = su;
            KFilt.x = [ 0; 0; 0; 0; 0; 0; 0];
            KFilt.A = [ 1 KFilt.T 0       0 0       0 ;
                        0       1 0       0 0       0 ;
                        0       0 1 KFilt.T 0       0 ;
                        0       0 0       1 0       0 ;
                        0       0 0       0 1 KFilt.T ;
                        0       0 0       0 0       1 ];
            KFilt.C = [ 1 0 0 0 0 0 ;
                        0 0 1 0 0 0 ;
                        0 0 0 0 1 0 ];
            KFilt.P = [ 100     0   0     0   0     0 ;
                          0 10000   0     0   0     0 ;
                          0     0 100     0   0     0 ;
                          0     0   0 10000   0     0 ;
                          0     0   0     0 100     0 ;
                          0     0   0     0   0 10000 ];
            KFilt.Qw = [ KFilt.T.^2 KFilt.T       0          0       0          0 ;
                         KFilt.T          1       0          0       0          0 ;
                               0          0 KFilt.T.^2 KFilt.T       0          0 ;
                               0          0 KFilt.T          1       0          0 ;
                               0          0       0          0 KFilt.T.^2 KFilt.T ;
                               0          0       0          0 KFilt.T          1 ]*KFilt.su^2;
            KFilt.Qv = {2*diag([7.2213, 2.6533, 0.4607]), diag([7.2213, 2.6533, 0.4607]), diag([2.1297, 0.9311, 0.1769]), diag([0.6315, 0.4684, 0.0634])};
        end
        
        %% Functions
        
        function KFilt = setT(KFilt, T_new)
            KFilt.T = T_new;
            KFilt.A = [ 1 KFilt.T 0       0 0       0 ;
                        0       1 0       0 0       0 ;
                        0       0 1 KFilt.T 0       0 ;
                        0       0 0       1 0       0 ;
                        0       0 0       0 1 KFilt.T ;
                        0       0 0       0 0       1 ];
            KFilt.Qw = [ KFilt.T.^2 KFilt.T       0          0       0          0 ;
                         KFilt.T          1       0          0       0          0 ;
                               0          0 KFilt.T.^2 KFilt.T       0          0 ;
                               0          0 KFilt.T          1       0          0 ;
                               0          0       0          0 KFilt.T.^2 KFilt.T ;
                               0          0       0          0 KFilt.T          1 ]*KFilt.su^2;
        end
        
        function KFilt = setX(KFilt, x_new)
            KFilt.x = x_new;
        end
        
        function KFilt = setA(KFilt, A_new)
            KFilt.A = A_new;
        end
        
        function KFilt = setC(KFilt, C_new)
            KFilt.C = C_new;
        end
        
        function KFilt = setP(KFilt, P_new)
            KFilt.P = P_new;
        end
        
        function KFilt = setQw(KFilt, Qw_new)
            KFilt.Qw = Qw_new;
        end
        
        function KFilt = setQv(KFilt, Qv_new)
            KFilt.Qv = Qv_new;
        end
        
        % Calculate the prediction of the state. (time update)
        function KFilt = kalPredict(KFilt)
            KFilt.x = KFilt.A*KFilt.x;
            KFilt.P = KFilt.A*KFilt.P*KFilt.A'+KFilt.Qw;
        end

        % Correct the Kalman Filter and calculates the corrected
        % estimation of the state. (measurement update)
        % y:        measurement vector [px; py; pz]
        % nViews:   number of used camera views for calculating y
        function KFilt = kalCorrect(KFilt, y, nViews)
            %%measurement update
            KFilt.K = KFilt.P*KFilt.C'*inv(KFilt.C*KFilt.P*KFilt.C'+KFilt.Qv{nViews});
            KFilt.x = KFilt.x + KFilt.K*(y-KFilt.C*KFilt.x);
            KFilt.P = (eye(size(KFilt.A))- KFilt.K*KFilt.C)*KFilt.P;
        end
        
    end
end
