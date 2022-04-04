function output = HawkesExp_mle_res_alp(data,StartTime,T,alp_bar,param_start,optim_options)
% -------------------------------------------------------------------------
% Purpose: Restricted (alp=alpbar) 
%          MLE of Hawkes process with exponential kernel given observations 
%          of event times t1 ... tn in [0,T]
% -------------------------------------------------------------------------
% Input:
%   data   : observed event times (n by 1 vector)
%            n is the number of observed events
%   StartTime: start time including the burn-in period
%   T      : time span of the observed data (scalar)
%   param_start  : initial guess of parameters [mu, bet]
%   optim_options: MLE optimization option
% -------------------------------------------------------------------------
% Ye Lu, 2020-09-06
% ye.lu1@sydney.edu.au
% -------------------------------------------------------------------------

% objective function: restricted likelihood with alp=alp_bar
objfun = @(param)(-loglik_temp(data,StartTime,T,alp_bar,param));

A = []; b = [];
Aeq=[]; beq=[]; % no equality constraints

%% Optimization
[param,fval,exitflag]=fmincon(objfun,param_start,A,b,Aeq,beq,[],[],[], optim_options);

% MLE estimates
alp_hat=alp_bar;
mu_hat =param(1);
bet_hat=param(2);

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

  function output = loglik_temp(data,StartTime,T,alp_bar,param)
    mu = param(1); bet = param(2);
    reparam = [mu, alp_bar/bet, bet];
    output = loglik_HawkesExp_reparam(data,StartTime,T,reparam);
  end 

