function output = HawkesExp_mle_res_mu(data,StartTime,T,mu_bar,param_start,optim_options)
% -------------------------------------------------------------------------
% Purpose: Restricted (mu=mubar) 
%          MLE of Hawkes process with exponential kernel given observations 
%          of event times t1 ... tn in (StartTime,T]
% -------------------------------------------------------------------------
% Input:
%   data   : observed event times (n by 1 vector)
%            n is the number of observed events
%   StartTime: start time including the burn-in period
%   T      : time span of the observed data (scalar)
%   param_start  : initial guess of parameters [alp, bet]
%   optim_options: MLE optimization option
% -------------------------------------------------------------------------
% Ye Lu, 2020-10-11
% ye.lu1@sydney.edu.au
% -------------------------------------------------------------------------

% restricted loglikelihood function with param = [br, bet]
objfun = @(param)(-loglik_temp(data,StartTime,T,mu_bar,param));

% initial guess of mle
alp_start = param_start(1);
bet_start = param_start(2);
br_start  = alp_start/bet_start; % initial guess of branching ratio

% initial guess of mle in term of branching ratio parameterization
reparam_start = [br_start,bet_start];

A = []; b = []; % no inequality constraints
Aeq=[]; beq=[]; % no equality constraints

%% Optimization
[param,fval,exitflag]=fmincon(objfun,reparam_start,A,b,Aeq,beq,[],[],[],optim_options);

% MLE estimates
mu_hat =mu_bar;
br_hat =param(1);
bet_hat=param(2);
alp_hat=br_hat*bet_hat;

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

  function output = loglik_temp(data,StartTime,T,mu_bar,param)
    br = param(1); bet = param(2);
    reparam = [mu_bar,br,bet];
    output = loglik_HawkesExp_reparam(data,StartTime,T,reparam);
  end 

