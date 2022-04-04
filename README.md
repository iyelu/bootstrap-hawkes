##### SOURCE BEGIN #####
%% Asymptotic and Bootstrap Inference of Hawkes Process
% <https://sites.google.com/view/luye *Ye Lu*>*, April 2022*
% 
% This MATLAB  live script illustrates the asymptotic and boostrap inference 
% of a Hawkes point process discussed in <https://www.google.com/url?q=https%3A%2F%2Fprotect-au.mimecast.com%2Fs%2FVePMC1WLPxcpMZK08cp0GI9%3Fdomain%3Dauthors.elsevier.com&sa=D&sntz=1&usg=AOvVaw1eLCheLsspM5K9Iagr3Vty 
% Cavaliere, Lu, Rahbek et al. (2022): "Bootstrap inference for Hawkes and general 
% point processes">.
% 
% The m-file version of this script |main_Mfile.m| and the auxiliary functions 
% can be found in the same directory.

clear;clc;close all;
addpath(genpath('./'));     % add temporary search pathes
% 1. Simulation
% We first simulate a sample path of Hawkes self-exciting point process with 
% exponential kernel function over time span $(0,T]$ with a burn-in period $(-M,0]$.
% 
% The conditional intensity process is given as
% 
% $$\lambda(t;\mu,\alpha,\beta)=\mu+\sum_{t_i<t}\gamma(t-t_i),\quad \gamma(x)=\alpha 
% e^{-\beta x}.$$
% 
% The _branching ratio_ of the process is $a:=\alpha/\beta$.

M = -50;  % burn-in period: (M,0]
T = 100;  % time span in the simulation (0,T]
%% 
% We set the true parameters in the simulation as 
% 
% $\mu_0=1,\quad \alpha_0=3,\quad \beta_0=5,\quad a_0:=\alpha_0/\beta_0=0.6$.

mu0 =1; alp0=3; bet0 = 5; param0 = [mu0;alp0;bet0];
%% 
% We call functions *@simu_thinning* and |*@HawkesExp_Intensity*| to simulate 
% one sample path of data.

% simulation specifications
sim_options.StartTime = M;
sim_options.StopTime  = T;
sim_options.InitialHistory=[]; % assume no events before or at start time
sim_options.print='off';

S = simu_thinning(@HawkesExp_Intensity,param0,sim_options)
% 2. Asymptotic inference
% We estimate the paramters using MLE and conduct asymptotic inference.

siglv = 0.05;   % significance level in the inference
optim_options = optimset('Display','off','TolX',1e-4,'TolFun',1e-4); % optimizaition specification
%% 
% All the inference results are saved in a structure array variable returned 
% by function |*@myAsym*|.

AsymMLE = myAsym(S.Points,M,T,param0,siglv,optim_options)
%% 
% Display: MLEs, standard errors, and 95% confidence intervals of the parameters:

if sum(AsymMLE.SanityCheck)==0                % If sanity check is passed
    param_names = {'mu','alpha','beta','a'};  % Note: a is the branching ratio given as alpha/beta
    param_initial = [param0;alp0/bet0];

    disp(repmat('-',1,60))
    fprintf('The following estimation results are based on a sample of\n')
    fprintf('total %d event times in (0,%d] with burn-in period (%d,0]\n', S.NumPositivePoints, S.StopTime, S.StartTime)
    disp(repmat('-',1,60))
    citext = sprintf('%.0f%% Asym CI',(1-siglv)*100);
    fprintf('%8s%8s%8s%8s%16s\n','', 'TRUE', 'MLE','SE',citext)
    for k=1:length(AsymMLE.mle_urs.param)
        fprintf('%8s%8.1f%8.3f%8.3f%5s%5.3f,%5.3f%s\n',...
            param_names{k},param_initial(k),AsymMLE.mle_urs.param(k),...
            AsymMLE.mle_urs.se_hess(k),'[',AsymMLE.CI_hess(k,1),AsymMLE.CI_hess(k,2),']')
    end
    disp(repmat('-',1,60))
else
    disp('The estimation does not pass the sanity check')
end
%% 
% Display: Likelihood ratio (LR) tests for the following four null hypotheses:
%% 
% * Joint test: $H_0: \theta=\theta_0$ where $\theta=(\mu,\alpha,\beta)$
% * Single test: $H_0: \mu = \mu_0$ 
% * Single test: $H_0: \alpha = \alpha_0$ 
% * Single test: $H_0: \beta = \beta_0$

if sum(AsymMLE.SanityCheck)==0                % If sanity check is passed
    disp(repmat('-',1,60))
    fprintf('%.0f%% Asymptotic LR test:\n', siglv*100)
    disp(repmat('-',1,60))
    fprintf('%15s%10s%8s%8s%8s\n','H0', 'LR stat', 'CV','Rej','p-val')
    fprintf('%15s%10.3f%8.3f%8.0f%8.3f\n','theta=theta0',AsymMLE.LR_all.stat,AsymMLE.LR_all.cv,AsymMLE.LR_all.rej,AsymMLE.LR_all.pval)
    fprintf('%15s%10.3f%8.3f%8.0f%8.3f\n','mu=mu0',AsymMLE.LR_mu.stat,AsymMLE.LR_mu.cv,AsymMLE.LR_mu.rej,AsymMLE.LR_mu.pval)
    fprintf('%15s%10.3f%8.3f%8.0f%8.3f\n','alpha=alpha0',AsymMLE.LR_alp.stat,AsymMLE.LR_alp.cv,AsymMLE.LR_alp.rej,AsymMLE.LR_alp.pval)
    fprintf('%15s%10.3f%8.3f%8.0f%8.3f\n','beta=beta0',AsymMLE.LR_bet.stat,AsymMLE.LR_bet.cv,AsymMLE.LR_bet.rej,AsymMLE.LR_bet.pval)
    disp(repmat('-',1,60))
else
    disp('The estimation does not pass the sanity check')
end
% 3. Bootstrap inference
% Below we implement the 4 bootstrap schemes proposed in <https://www.google.com/url?q=https%3A%2F%2Fprotect-au.mimecast.com%2Fs%2FVePMC1WLPxcpMZK08cp0GI9%3Fdomain%3Dauthors.elsevier.com&sa=D&sntz=1&usg=AOvVaw1eLCheLsspM5K9Iagr3Vty 
% Cavaliere, Lu, Rahbek et al. (2022)> to obtain bootstrap MLE, bootstrap standard 
% errors, and bootstrap percentile intervals.
% 
% (1) PRFB: parametric fixed intensity boostrap
% 
% (2) PRRB: parametric recursive intensity bootstrap
% 
% (3) NPFB: nonparametric fixed intensity bootstrap
% 
% (4) NPRB: nonparametric recursive intensity bootstrap
% 
% Here we specify a small number (99) of bootstrap iterations to showcase the 
% idea. You may increase the number to allow more bootstrap iterations.

B  = 99;     % number of bootstrap iterations
NB = 4;      % number of bootstrap schemes: PRFB, NPFB, PRRB, PRFB
BStypeAll = {'PR','NP'}; BSintensityAll = {'FB','RB'};
%% 
% All the boostrap inference results are saved in a structure array variable 
% returned by function *@myBS*.

if sum(AsymMLE.SanityCheck)==0   % sanity check passed
    disp('Sanity check passed: Good to go with bootstrap')

    UBS = cell(B,1);                         % bootstrap results returned by @myBS
    param_star = AsymMLE.mle_urs.param(1:3); % bootstrap true value
    param_bs   = zeros(B,4,NB);              % MLE of each bootstrap sample    
    BSMLE = zeros(4,NB);                     % bootstrap MLE    
    BSSE = zeros(4,NB);                      % bootstrap standard erros
    PI   = zeros(4,2,NB);                    % bootstrap percentile interval

    i = 0;              % bootstrap scheme #i
    for i1 = 1:2        % fixed intensity (FB) and recursive intensity (RB)
        for i2 = 1:2    % parametric boostrap (PR) and nonparametric boostrap (NP)
            i=i+1;
            BStype = BStypeAll{i2}; BSintensity = BSintensityAll{i1};
            fprintf('Working on %s%s...\n',BStype,BSintensity)
            b = 0;
            
            % Simulate B bootstrap samples that all pass the sanity check
            while b < B
                b = b+1;
                temp = myBS(S,param_star,BStype,BSintensity,optim_options);
                if sum(temp.SanityCheck)==0
                    UBS{b} = temp;
                else  % Discard the sample if the sanity check is not passed
                    b = b-1;
                end
            end
            
            % Parameter estimates of each bootstrap sample
            for b = 1:B
                param_bs(b,:,i) = UBS{b}.mle.param';
            end
            
            % BS MLE
            BSMLE(:,i) = mean(param_bs(:,:,i))';
            
            % BS standard errors
            BSSE(:,i) = std(param_bs(:,:,i))';

            % BS percentile interval
            lower = 1; upper = 2;
            PI(:,lower,i) = quantile(param_bs(:,:,i),siglv/2)';
            PI(:,upper,i) = quantile(param_bs(:,:,i),1-siglv/2)';
        end
    end
    fprintf('\n')

    PRFB = 1; NPFB = 2; PRRB = 3; NPRB = 4;
%% 
% Display: Bootstrap MLE   

    disp(repmat('-',1,80))
    fprintf('Bootstrap MLEs based on %d bootstrap repetitions\n', B)
    disp(repmat('-',1,80))
    fprintf('%8s%8s%8s%8s%8s\n','', 'PRFB', 'NPFB','PRRB','NPRB')
    for k=1:length(AsymMLE.mle_urs.param)
        fprintf('%8s%8.3f%8.3f%8.3f%8.3f\n',param_names{k},BSMLE(k,PRFB),BSMLE(k,NPFB),BSMLE(k,PRRB),BSMLE(k,NPRB))
    end
%% 
% Display: Bootstrap standard errors

    disp(repmat('-',1,80))
    fprintf('Bootstrap standard errors based on %d bootstrap repetitions\n', B)
    disp(repmat('-',1,80))
    fprintf('%8s%8s%8s%8s%8s\n','', 'PRFB', 'NPFB','PRRB','NPRB')
    for k=1:length(AsymMLE.mle_urs.param)
        fprintf('%8s%8.3f%8.3f%8.3f%8.3f\n',param_names{k},BSSE(k,PRFB),BSSE(k,NPFB),BSSE(k,PRRB),BSSE(k,NPRB))
    end
%% 
% Display: Bootstrap percentile intervals

    disp(repmat('-',1,80))
    fprintf('%.0f%% bootstrap percentile intervals based on %d bootstrap repetitions\n', (1-siglv)*100, B)
    disp(repmat('-',1,80))
    fprintf('%8s%18s%18s%18s%18s\n','', 'PRFB', 'NPFB','PRRB','NPRB')
    for k=1:length(AsymMLE.mle_urs.param)
        fprintf('%8s%6s%5.2f,%5.2f%s%6s%5.2f,%5.2f%s%6s%5.2f,%5.2f%s%6s%5.2f,%5.2f%s\n',param_names{k},...
            '[',PI(k,1,PRFB),PI(k,2,PRFB),']','[',PI(k,1,NPFB),PI(k,2,NPFB),']',...
            '[',PI(k,1,PRRB),PI(k,2,PRRB),']','[',PI(k,1,NPRB),PI(k,2,NPRB),']')
    end
    disp(repmat('-',1,80))
else  
    disp('Not passing sanity check: Not good to go with bootstrap')
end
%%
rmpath(genpath('./'));  % remove temporary search pathes
##### SOURCE END #####
-->
</div></body></html>
