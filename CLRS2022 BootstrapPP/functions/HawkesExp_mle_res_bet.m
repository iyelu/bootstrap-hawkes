function output = HawkesfExp_mle_res_bet(data,StartTime,T,bet_bar,param_start,optim_options)
% -------------------------------------------------------------------------
% Purpose: Restricted (bet=betbar) 
%          MLE of Hawkes process with exponential kernel given observations 
%          of event times t1 ... tn in [0,T]
% -------------------------------------------------------------------------
% Input:
%   data   : observed event times (n by 1 vector)
%            n is the number of observed events
%   T      : time span of the observed data (scalar)
%   param_start  : initial guess of parameters [mu, alp]
%   optim_options: MLE optimization option
% -------------------------------------------------------------------------
% Ye Lu, 2020-10-11
% ye.lu1@sydney.edu.au
% -------------------------------------------------------------------------

% objective function: restricted likelihood with bet=bet_bar, param = [mu, alp]
objfun = @(param)(-loglik_temp(data,StartTime,T,bet_bar,param));

A = []; b = [];
Aeq=[]; beq=[]; % no equality constraints

%% Optimization
[param,fval,exitflag]=fmincon(objfun,param_start,A,b,Aeq,beq,[],[],[], optim_options);

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
  output.exitmsg = 'Change in x was less than options.StepTolerance and maximum constraint violation was less than options.ConstraijntTolerance';
end

%% Save outputs
output.param  = [mu_hat;alp_hat;bet_hat];
output.loglik = -fval;
output.exitflag = exitflag;
output.optim_options = optim_options;

end

%% Helper function 

  function output = loglik_temp(data,StartTime,T, bet_bar,param)
    mu = param(1); alp = param(2);
    reparam = [mu, alp/bet_bar, bet_bar];
    output = loglik_HawkesExp_reparam(data,StartTime,T,reparam);
  end 

