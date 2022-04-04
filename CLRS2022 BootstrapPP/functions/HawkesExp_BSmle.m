function output = HawkesExp_BSmle(BSintensity,BSdata,data,StartTime,T,param_start,screen_print,optim_options)
% -------------------------------------------------------------------------
% Purpose: Bootstrap MLE of Hawkes process with exponential kernel given observations
%          of event times t_star(1) ... t_star(n_star) in (StartTime,T]
% -------------------------------------------------------------------------
% Input:
%   BSintensity: 'FB' or 'RB'
%   BSdata : bootstrap data (n_star by 1 vector)
%   data   : original event times (n by 1 vector)
%            n is the number of observed events
%   StartTime: starting time including the burn-in period
%   T      : time span of the observed data (scalar)
%   param_start  : initial guess of parameters [mu, alp, bet]
%   screen_print : 'on' / 'off'
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

% objective function param = [mu, br, bet]
objfun = @(reparam)(-BSloglikfun(BSdata,data,StartTime,T,reparam));

% -- Pre MLE Optimization --
% obtain initial guess "reparam_start" in term of mu, br, bet parameterization
mu_start  = param_start(1);
alp_start = param_start(2);
bet_start = param_start(3);
br_start  = alp_start/bet_start;
reparam_start = [mu_start,br_start,bet_start];

A = []; b = []; % no inequalit constraints
Aeq=[]; beq=[]; % no equality constraints

% --MLE Optimization --
[reparam_mle,fval,exitflag]=fmincon(objfun,reparam_start,A,b,Aeq,beq,[],[],[], optim_options);

% MLE estimates
mu_hat    =reparam_mle(1);
br_hat    =reparam_mle(2);
bet_hat   =reparam_mle(3);
alp_hat   =br_hat * bet_hat;
param_mle = [mu_hat;alp_hat;bet_hat];

% exitmsg of the optimization of this sample
if exitflag == 1
  output.exitmsg = 'fminsearch converged to a solution.';
elseif exitflag == 0
  output.exitmsg = 'Maximum number of function evaluations or iterations was reached.';
elseif exitflag == -1
  output.exitmsg = 'Algorithm was terminated by the output function.';
elseif exitflag == 2
  output.exitmsg = 'Change in x was less than options.StepTolerance and maximum constraint violation was less than options.ConstraintTolerance';
end

% screen prints
if strcmp(screen_print, 'on')
    disp(repmat('*',1,50))
    disp('MLE result')
    
    fprintf('Parameter mu = %9.4f\n', mu_hat);
    fprintf('Parameter alpha = %9.4f\n', alp_hat);
    fprintf('Parameter beta = %9.4f\n', bet_hat);
    fprintf('Parameter br = %9.4f\n', br_hat);
    
    disp(repmat('*',1,50))
    disp(output.exitmsg)
end

% -- Compute bootstrap variance -- 

% Hessian for theta_old = (mu, alp, bet)
loglik_old = @(param)(loglik_old_param(BSloglikfun,BSdata,data,StartTime,T,param));
H_param = hessian(loglik_old,param_mle);
[~,p] = chol(-H_param);
if strcmp(screen_print, 'on')
  if p==0
    fprintf('Numerically computed negative Hessian is positive definite\n')
  else
    fprintf('Numerically computed negative Hessian is NOT positive definite\n')
  end
end

% Hessian for theta_new = (mu, br, bet)
loglik_new = @(reparam)(BSloglikfun(BSdata,data,StartTime,T,reparam));
H_reparam = hessian(loglik_new,reparam_mle);
[~,p] = chol(-H_reparam);
if strcmp(screen_print, 'on')
  if p==0
    fprintf('Numerically computed negative Hessian is positive definite\n')
  else
    fprintf('Numerically computed negative Hessian is NOT positive definite\n')
  end
end

var_param_hess   = inv(-H_param);
var_reparam_hess = inv(-H_reparam);
se_param_hess    = diag(sqrt(var_param_hess));
se_reparam_hess  = diag(sqrt(var_reparam_hess));

%% Save outputs
output.loglik        = -fval;
output.exitflag      = exitflag;
output.optim_options = optim_options;
output.param         = [mu_hat;alp_hat;bet_hat;br_hat];
output.se_hess       = [se_param_hess;se_reparam_hess(2)];

% - Hessian matrix for old parameterization (mu, alp, bet) -
output.hess_psd_oldparam = prod(eig(H_param))>=0;
output.hess_oldparam = H_param;

% - Hessian matrix for new parameterization (mu, br, bet) -
output.hess_psd_newparam = prod(eig(H_reparam))>=0;
output.hess_newparam = H_reparam;

%% Helper function
  function output = loglik_old_param(BSloglikfun,BSdata,data,StartTime,T,param)
    % -------------------------------------------------------------------------
    % Purpose: log-likelihood of Hawkes process with exponential kernel
    %          given observations t(1) ... t(n) over [T_start, T_end]
    %          with the "old" parameterization 
    %             theta = (mu, alp, bet)
    %          where 
    %          - mu: baseline intensity 
    %          - alp: jump size in the intensity
    %          - bet: memory decay parameter
    % -------------------------------------------------------------------------

    % baseline intensity 
    mu = param(1);
    % jump size of the intensity
    alp = param(2);
    % memory parameter
    bet = param(3);
    % branching ratio 
    br = alp/bet;

    % reparameterization by branching ratio 
    reparam = [mu, br, bet];
    output  = BSloglikfun(BSdata,data,StartTime,T,reparam);
  end

end 
