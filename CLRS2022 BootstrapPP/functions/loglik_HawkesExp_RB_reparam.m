function output = loglik_HawkesExp_RB_reparam(t_star,t,StartTime,T,reparam)
% -------------------------------------------------------------------------
% Purpose: Recursive Bootstrap log-likelihood of Hawkes process 
%          of bootstrap event times t_star(1),...,t_star(n_star)
%          over [0,T]
%          with the new parameterization [mu, br, bet]
%          (note that we actually don't need original time t)
% -------------------------------------------------------------------------
% Input:
%   t_star : bootstrap event times (n_star by 1 vector)
%   t      : original event times (n by 1 vector) 
%            (not used here, included to match loglik_HinfExp_FB.m)
%   T      : end time of the observed data (assume T0=0)
%   reparam: [mu; br; bet]
%            mu  : baseline intensity 
%            br  : branching ratio: alp/bet
%            bet : memory parameter in the normalized kernel function
% -------------------------------------------------------------------------
% Ye Lu, 2020-10-24
% Contact: ye.lu1@sydney.edu.au
% -------------------------------------------------------------------------

output = loglik_HawkesExp_reparam(t_star,StartTime,T,reparam);


