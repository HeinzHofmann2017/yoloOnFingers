function [ A ] = crossMat( a )
%CROSSMAT Transform 3-element vector into cross-product-matrix
% 
%   Input:
%       a:  3-element vector [3x1]
% 
%   Output:
%       A:  cross-produt-matrix [3x3]
% 

if any( size(a)~=[3,1] )
    error('Input vector needs to be of the size [3x1]' );
end

A = [   0  -a(3)  a(2) ;
      a(3)    0  -a(1) ;
     -a(2)  a(1)    0  ];
end

