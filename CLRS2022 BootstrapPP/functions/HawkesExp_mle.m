function output = HawkesExp_mle(data,StartTime,T,param_start,screen_print,optim_options)
% -------------------------------------------------------------------------
% Purpose: MLE of Hawkes process with exponential kernel given observations
%          of event times t1 ... tn in [T_start,T_end]
% -------------------------------------------------------------------------
% Input:
%   data   : observed event times (n by 1 vector)
%            n is the number of observed events
%   StartTime: starting time including the burn-in period
%   T      : time span of the observed data (scalar)
%   param_start  : initial guess of parameters [mu, alp, bet]
%   screen_print : 'on' / 'off'
%   optim_options: MLE optimization option
% -------------------------------------------------------------------------
% Call functions:
%     The "DERIVEST suit" for numerical differentiation
%     loglikelihood for new parameterization: @loglik_HawkesExp_reparam
%     loglikelihood for old parameterization: @loglik_HawkesExp
% -------------------------------------------------------------------------
% Ye Lu, 2020-10-24
% ye.lu1@sydney.edu
% -------------------------------------------------------------------------

Model_Intensity = @HawkesExp_Intensity;

% check data is a column vector
if ~iscolumn(data)
    data = data';
end

% objective function in optimization
objfun = @(reparam)(-loglik_HawkesExp_reparam(data,StartTime,T,reparam));

% -- Pre MLE Optimization --

% obtain initial guess "reparam_start" in term of mu, br, bet parameterization
mu_start      = param_start(1);
alp_start     = param_start(2);
bet_start     = param_start(3);
br_start      = alp_start/bet_start;
reparam_start = [mu_start;br_start;bet_start];

A = []; b = []; % no inequalit constraints
Aeq=[]; beq=[]; % no equality constraints

% -- MLE Optimization --
[reparam_mle,fval,exitflag]=fmincon(objfun,reparam_start,A,b,Aeq,beq,[],[],[], optim_options);

% -- Post MLE Optimization --

% MLE estimates for both parameterizations
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
    
    fprintf('Estimation finished.\n\n')
    
    fprintf('Computing variance....\n\n')
end

% -- Compute Variance --
% Recall that Information = - Hessian
% We report the inverse(-Hessian), inverse(Information), as well as the robust sandwich matrix.

% Information matrix by numerical derivatives
g = 200; % average number of grids between event times for integration
[Info_param,Info_reparam]=intensity2info(Model_Intensity,data,T,param_mle,g);

% Hessian for theta_old = (mu, alp, bet)
loglik_old = @(param)(loglik_HawkesExp(data,StartTime,T,param));
H_param = hessian(loglik_old,param_mle);
[~,po] = chol(-H_param);
if strcmp(screen_print, 'on')
    if po==0
        fprintf('Old parameterization: Hessian is negative definite (optimum found) \n')
    else
        fprintf('Old parameterization: Hessian is NOT negative definite (not optimum)\n')
    end
end

% Hessian for theta_new = (mu, br, bet)
loglik_new = @(reparam)(loglik_HawkesExp_reparam(data,StartTime,T,reparam));
H_reparam = hessian(loglik_new,reparam_mle);
[~,pn] = chol(-H_reparam);
if strcmp(screen_print, 'on')
    if pn==0
        fprintf('New parameterization: Hessian is negative definite (optimum found) \n')
    else
        fprintf('New parameterization: Hessian is NOT negative definite (not optimum)\n')
    end
end

var_param_hess   = inv(-H_param);
var_param_info   = inv(Info_param);
var_param_rbst   = inv(H_param) * Info_param * inv(H_param); % robust variance by sandwich form

var_reparam_hess = inv(-H_reparam);
var_reparam_info = inv(Info_reparam);
var_reparam_rbst = inv(H_reparam) * Info_reparam * inv(H_reparam); % robust variance by sandwich form

se_param_hess = diag(sqrt(var_param_hess));
se_param_info = diag(sqrt(var_param_info));
se_param_rbst = diag(sqrt(var_param_rbst));

se_reparam_hess = diag(sqrt(var_reparam_hess));
se_reparam_info = diag(sqrt(var_reparam_info));
se_reparam_rbst = diag(sqrt(var_reparam_rbst));

%% -- Save outputs --
output.loglik = -fval;
output.exitflag = exitflag;
output.optim_options = optim_options;
output.param   = [mu_hat;alp_hat;bet_hat;br_hat];
output.se_hess = [se_param_hess;se_reparam_hess(2)];
output.se_info = [se_param_info;se_reparam_info(2)];
output.se_rbst = [se_param_rbst;se_reparam_rbst(2)];

% - variance matrix for old parameterization (mu, alp, bet) -
output.hess_psd_oldparam = ~(po==0);
output.var_hess_oldparam = var_param_hess;
output.var_info_oldparam = var_param_info;
output.var_rbst_oldparam = var_param_rbst;

% - variance matrix for new parameterization (mu, br, bet) -
% output.hess_psd_newparam = prod(eig(H_reparam))>=0;
output.hess_psd_newparam = ~(pn==0);
output.var_hess_newparam = var_reparam_hess;
output.var_info_newparam = var_reparam_info;
output.var_rbst_newparam = var_reparam_rbst;

% ---------------------------------
% Screen prints of estimated variance
if strcmp(screen_print, 'on')
    % -- Printing the MLE result --
    p = {'mu', 'alpha', 'beta', 'br'};
    disp(repmat('*',1,60))
    fprintf('%6s%10s%14s%14s%14s\n','PARAM', 'COEF', 'Hess SE', 'Info SE', 'Robust SE');
    for k=1:length(output.param)
        fprintf('%6s%10.4f%14.4f%14.4f%14.4f \n',p{k},output.param(k),output.se_hess(k),output.se_info(k),output.se_rbst(k));
    end
    disp(repmat('*',1,60))
    disp(output.exitmsg);
end

end

% Helper function

function  [output1,output2] = intensity2info(Model_Intensity,data,T,param,g)


ind = find(data>0);  % indices of event times in (t(0),T] or (0,T]
n   = numel(ind);    % number of events in (t(0),T] or (0,T]

% find t(0)
if ind(1)==1  % all the events are in (0,T]
  t0=0;
else
  t0 = data(ind(1)-1);
end

% chop (t0,T] to (n+1) intervals for numerical integration
info1 = zeros(3,3,n+1); info2 = zeros(3,3,n+1);
grids = [t0;data(data>0);T]; delta = mean(diff(data))/g; % length of the grid

% integration in the i-th subinterval
for i = 1:n+1
    
    % construct intensity process
    time = [grids(i)+delta:delta:grids(i+1),grids(i+1)]; % row vector of grids
    ng   = length(time); % number of grids in this subinterval
    len  = diff([grids(i);time']); % length of the chopped interval
    
    % History points index
    index = repmat(data,1,ng)<time;    
    temp1 = zeros(3,3,ng); temp2 = zeros(3,3,ng);
    
    for k = 1:ng
        History = data(index(:,k));
       
        [l,lp]=Model_Intensity(time(k),History,param); % lambda and lambda_prime
        lp_param   = lp(1:3);
        lp_reparam = [lp(1);lp(4);lp(3)];
        
        % integration over the k-th grid
        temp1(:,:,k) = (lp_param * lp_param'/l)*len(k);  
        temp2(:,:,k) = (lp_reparam * lp_reparam'/l)*len(k);
    end
    info1(:,:,i) = sum(temp1,3);
    info2(:,:,i) = sum(temp2,3);
end
output1 = sum(info1,3);
output2 = sum(info2,3);
end
