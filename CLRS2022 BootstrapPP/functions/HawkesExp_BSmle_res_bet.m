function output = HawkesExp_BSmle_res_bet(BSintensity,BSdata,data,StartTime,T,param_res,param_start,optim_options)
% -------------------------------------------------------------------------
% Purpose: Restricted (bet=betbar) 
%          Bootstrap MLE of Hawkes process with exponential kernel given observations
%          of event times t_star(1) ... t_star(n_star) in [0,T]
%          of event times t1 ... tn in [0,T]
% -------------------------------------------------------------------------
% Input:
%   BSintensity: 'FB' or 'RB'
%   BSdata : bootstrap data (n_star by 1 vector)
%   data   : original event times (n by 1 vector)
%            n is the number of observed events
%   StartTime: start time including the burn-in period
%   T      : time span of the observed data (scalar)
%   param_res: restricted parameter value (bet_bar this case)
%   param_start  : initial guess of parameters [mu, alp]
%   optim_options: MLE optimization option
% -------------------------------------------------------------------------
% Ye Lu, 2020-10-24
% Contact: ye.lu1@sydney.edu.au
% -------------------------------------------------------------------------

switch BSintensity
  case 'FB'
    BSloglikfun = @loglik_HawkesExp_FB_reparam;
  case 'RB'
    BSloglikfun = @loglik_HawkesExp_RB_reparam;
end 

bet_bar = param_res;

% restricted (bet=bet_bar) loglikelihood function with param = [mu, alp]
% see the helper function
objfun = @(param)(-loglik_temp(BSloglikfun,BSdata,data,StartTime,T,bet_bar,param));

A = []; b = []; % no inequality constraints
Aeq=[]; beq=[]; % no equality constraints

%% Optimization
[param,fval,exitflag]=fmincon(objfun,param_start,A,b,Aeq,beq,[],[],[],optim_options);

% MLE estimates
bet_hat=bet_bar;
mu_hat =param(1);
alp_hat=param(2);

if exitflag == 1
    output.exitmsg = 'fminsearch converged to a solution.';
elseif exitflag == 0
    output.exitmsg = 'Maximum number of function evaluations or iterations was reached.';
elseif exitflag == -1
    output.exitmsg = 'Algorithm was terminated by the output function.';
elseif exitflag == 2
  output.exitmsg = 'Change in x was less than options.StepTolerance and maximum constraint violation was less than options.ConstraintTolerance';
end

%% Save outputs
output.param  = [mu_hat;alp_hat;bet_hat];
output.loglik = -fval;
output.exitflag = exitflag;
output.optim_options = optim_options;

end

%% Helper function 

  function output = loglik_temp(BSloglikfun,BSdata,data,StartTime,T,bet_bar,param)
    mu = param(1); alp = param(2);
    reparam = [mu, alp/bet_bar, bet_bar];
    output = BSloglikfun(BSdata,data,StartTime,T,reparam);
  end 

