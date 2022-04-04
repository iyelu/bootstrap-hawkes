function output = loglik_HawkesExp(t,StartTime,T,param)
% -------------------------------------------------------------------------
% Purpose: log-likelihood of Hawkes process with exponential kernel
%          given observations t(1) ... t(n) over (t(0), T]
% -------------------------------------------------------------------------
% Input:
%   t      : observed event times (n by 1 vector)
%            n is the number of observed events
%   StartTime: starting time of the events (including the burn-in period)
%   T      : end time of the observed data (assume T0=0)
%   param:   [mu; alp; bet]
%            mu  : baseline intensity
%            alp : jump size in the intensity
%            bet : memory parameter in the normalized kernel function
% -------------------------------------------------------------------------
% Ye Lu, 2020-10-11
% Contact: ye.lu1@sydney.edu.au
% -------------------------------------------------------------------------

% Intensity function and integrated intensity function  
Model_Intensity     = @HawkesExp_Intensity;
Model_Int_Intensity = @HawkesExp_Integrated_Intensity;

% check t is a column vector
if ~iscolumn(t)
    t = t';
end

ind = find(t>0);  % indices of event times in (t(0),T] or (0,T]

if isempty(ind)
    % disp('warning: no data available for likelihood evaluation')
    v = Model_Int_Intensity([t;T],StartTime,param);
    Lambda_T = v(end);
    output = -Lambda_T;
else    
    nT  = numel(ind); % number of events in (t(0),T] or (0,T]
    
    % -- Part 1: compute lambda(t(i); theta), i=1,...,nT --
    lambda = zeros(nT,1);
    for i = 1:nT
        k = ind(i);  % only consider events in (0,T]
        History   = t(t<t(k));
        lambda(i) = Model_Intensity(t(k),History,param);
    end
    
    % -- Part 2: compute Lambda(t0,T; theta) --
    v = Model_Int_Intensity([t;T],StartTime,param);
    Lambda_T = sum(v(ind(1):end));
    
    % Output: log-likelihood function
    output = sum(log(lambda)) - Lambda_T;
end


