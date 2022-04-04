function output = loglik_HawkesExp_reparam(t,StartTime,T,reparam)
% -------------------------------------------------------------------------
% Purpose: log-likelihood of Hawkes process with exponential kernel
%          given observations t(1) ... t(n) over [0, T]
%          with the new parameterization [mu, br, bet]
% -------------------------------------------------------------------------
% Input:
%   t      : observed event times (n by 1 vector)
%            n is the number of observed events
%   StartTime:  start time of the observed data (including the burn-in)
%   T      : end time of the observed data 
%   reparam: [mu; br; bet]
%            mu  : baseline intensity 
%            br  : branching ratio: alp/bet
%            bet : memory parameter in the normalized kernel function
% -------------------------------------------------------------------------
% Ye Lu, 2020-10-11
% Contact: ye.lu1@sydney.edu.au
% -------------------------------------------------------------------------

mu  = reparam(1);
br  = reparam(2);
bet = reparam(3);
alp = bet * br;

param = zeros(3,1);
param(1) = mu; 
param(2) = alp;
param(3) = bet;

output = loglik_HawkesExp(t,StartTime,T,param);

