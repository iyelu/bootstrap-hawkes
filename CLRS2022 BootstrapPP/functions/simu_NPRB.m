function output = simu_NPRB(Seq,param)
% -------------------------------------------
% Purpose: Simulate Bootstrap event time
%          Non-parametric (NP) recursive (RB) bootstrap
%          with mean scaling
% -------------------------------------------
% Input:
%   Seq: structure variable for one data sample
%          Seq.Param (parameter in the data DGP)
%          Seq.StartTime (-M)
%          Seq.StopTime (T)
%          Seq.Points (simulated event times)
%          Seq.NumPoints
%          Seq.NumPositivePoints
%   param: parameters in bootstrap DGP
%          mu, alp, bet
% -------------------------------------------
% Ye Lu, 2020-11-02
% ye.lu1@sydney.edu.au
% -------------------------------------------

Model_Int_Intensity = @HawkesExp_Integrated_Intensity;

output = struct('Param', [], ...
    'StartTime', [], ...
    'StopTime', [], ...
    'Points', [], ...
    'NumPoints', [], ...
    'NumPositivePoints', [] ...
    );

mu = param(1); alp = param(2); bet = param(3);

t = Seq.Points;  % data in (-M,T]
T = Seq.StopTime; M = Seq.StartTime;
InitialHistory = t(t<0); m = numel(InitialHistory);

% 1. compute S_m
backtimes = t(m)-t(1:m);
S = sum(exp(-bet*backtimes));

% 2. n_T time changed waiting times vi's for NP bootstrap
[v, ~] = Model_Int_Intensity(t,M,param);
v    = v(m+1:end); % v_{m+1},v_{m+2},...,v_{m+n_T}
v_sc = v/mean(v);  % scaled waiting times
n    = numel(v_sc);

% 3. generate bootstrap event times using time change method
t_star = t(m); t_star_all = [];
while t_star <= T
    v_star = v_sc(randi(n));
    fun = @(x)tc_fun(x,S,v_star,param);
    x0 = v_star/mu; w_star = fzero(fun, x0);
    S = exp(-bet*w_star)*S+1;
    t_star = t_star+w_star;
    if t_star <= T
        t_star_all = [t_star_all; t_star];
    end
end
Points = [InitialHistory; t_star_all];

%% Output %%
output.Param     = param;
output.StartTime = Seq.StartTime;
output.StopTime  = Seq.StopTime;
output.Points    = Points;
output.NumPoints = numel(Points);
output.NumPositivePoints = numel(Points(Points>0));
end

%% Helper Function
function output = tc_fun(x,S,v,para)
% ----------------------------------------------------------------
% Input:
%   x:       waiting time of Hawkes Point Process
%   S:       S_k function which has iterative structure
%   v:       transformed waiting time
%   para:    parameters in Hawkes process with exponential kernel
% ----------------------------------------------------------------
mu = para(1); alp = para(2); bet = para(3);
output = mu * x + alp/bet * (1-exp(-bet*x))*S - v;
end
