function output = loglik_HawkesExp_FB_reparam(t_star,t,StartTime,T,reparam)
% -------------------------------------------------------------------------
% Purpose: Fix-Intensity Bootstrap log-likelihood of Hawkes process 
%          with exponential kernel function with parameteres mu, br, bet
%          of bootstrap event times t_star(1),...,t_star(n_star)
%          from original event times t(1) ... t(n) over [0,T]
% -------------------------------------------------------------------------
% Input:
%   t_star  : bootstrap event times (n_star by 1 vector)
%   t       : original event times (n by 1 vector)
%            n is the number of observed events
%   StartTime: start time of the data (including burn-in)
%   T       : end time of the data  
%   reparam : [mu,br,bet] 
% -------------------------------------------------------------------------
% Note: This function is similar to loglik_HawkesExp.m
% -------------------------------------------------------------------------
% Ye Lu, 2020-10-24
% Contact: ye.lu1@sydney.edu.au
% -------------------------------------------------------------------------

% Intensity function and integrated intensity function  
Model_Intensity     = @HawkesExp_Intensity;
Model_Int_Intensity = @HawkesExp_Integrated_Intensity;

% parameters and reparameterizatoin
mu  = reparam(1); br  = reparam(2); bet = reparam(3); alp = bet * br;
param = zeros(3,1); param(1) = mu; param(2) = alp; param(3) = bet;

% check t_star is a column vector
if ~iscolumn(t_star)
    t_star = t_star';
end

ind_star = find(t_star>0);  % indices of event times in (t(0),T] or (0,T]

if isempty(ind_star)       
    % disp('warning: no data available for likelihood evaluation')
    v = Model_Int_Intensity([t;T],StartTime,param);
    ind = find(t>0);
    Lambda_T = sum(v(ind(1):end));
    output = -Lambda_T;
else
    n_star = numel(ind_star);  % number of events > 0
    
    % Part 1: compute lambda(t_star(i); theta), i=1,...,n_star
    lambda = zeros(n_star,1);
    for i = 1:n_star
      k = ind_star(i);  % only consider events in (0,T]
      % *** only different part than loglik_HinfExp.m ***
      History   = t(t<t_star(k)); 
      lambda(i) = Model_Intensity(t_star(k),History,param);
    end

    % Part 2: Lambda(0,T; theta) only depends on the original data
    v = Model_Int_Intensity([t;T],StartTime,param);
    ind = find(t>0);
    Lambda_T = sum(v(ind(1):end));

    % log-likelihood function
    output = sum(log(lambda)) - Lambda_T;
end



