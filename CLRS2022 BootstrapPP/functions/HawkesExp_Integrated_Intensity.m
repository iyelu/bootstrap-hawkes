function [v,s] = HawkesExp_Integrated_Intensity(t,StartTime,param)
% -------------------------------------------------------------------------
% Purpose: Integrated intensities of Hawkes with exponential kernel
%
%          Given t(1), t(n) in (StartTime, StopTime], time changed
%          waiting times are integrated intensities over consecutive intervals
%             v(i) = Lambda(t(i-1),t(i); param)
%          and the time changed event times are integrated intensities 
%          over [-infty,t(i)]
%             s(i) = Lambda(-infty,t(i); param)
% -------------------------------------------------------------------------
% Input:
%   t      : event times of Hawkes process (n by 1 vector)
%   param:   [mu; alp; bet]
%            mu  : baseline intensity 
%            alp : jump size in the intensity
%            bet : memory parameter in the normalized kernel function
%            br  : branching ratio: alp/bet in this case
% -------------------------------------------------------------------------
% Reference:
%   Let w(i)=t(i)-t(i-1) be original waiting times  [w(1)=t(1)-T0]
%   v: transformed waiting times  (n by 1 vector)
%      E.g: Hawkes(infty) model
%        v(i) = mu*w(i) + a * v_h(i), for n>=1
%      where v_h is the integrated triggering kernel
%   s: transformed event times (n by 1 vector): s = Lambda(t;param)
% -------------------------------------------------------------------------
% Ye Lu, 2020-10-11
% ye.lu1@sydney.edu.au
% -------------------------------------------------------------------------

mu  = param(1);
alp = param(2);
bet = param(3);

% branching ratio
br  = alp/bet;

if isempty(t)  % no event occurrence
  v = []; s=[];
  fprintf('Warning: There is no events for time change!\n')

else           % at least one event occurrence
  n = numel(t);
  if ~iscolumn(t)
    t = t';
  end
  w  = [t(1)-StartTime; diff(t)]; % original waiting times

  % -- the triggering kernel part (see auxiliary function below) --
  [v_h,~] = Integrated_Kernel(t,StartTime,bet);

  % -- total part of the time changed waiting times--
  v = mu* w + br * v_h;

  % -- time changed event times --
  s = StartTime+cumsum(v);
end

%% Helper Function

function [v_h,H] = Integrated_Kernel(t,StartTime,bet)
% -------------------------------------------------------------------------
% Purpose: Integrated normalized exponential kernel for Hawkes(inftyty) 
%
%          Given t(1),..., t(n), time changed waiting times are integrated 
%          intensities over consecutive event intervals
%             v(i) = H(t(i-1),t(i); bet)
% -------------------------------------------------------------------------
% Input:
%   t      : event times of Hawkes process (n by 1 vector)
%   bet    : parameter in normalized exponential kernel h(x;bet)
% -------------------------------------------------------------------------
% Reference:
%   Let w(i)=t(i)-t(i-1) be original waiting times  [w(1)=t(1)-T0]
%   v_h: triggering kernel part of transformed waiting times  (n by 1 vector)
%      E.g: Hawkes(infty) model
%        v_h(1) = 0
%        v_h(i) = [1-exp(-bet*w(i)]*SS(i-1), for n>=2
%        where
%             SS(1) = 1;
%             SS(i) = exp(-bet*w(i))*SS(i-1)+1, for i>=2
%   H: integrated normalized triggering kernel part of the intensity
%        H = H(0,t(end);bet)
% -------------------------------------------------------------------------

if isempty(t)  % no event occurrence
  v_h = []; s_h=[];
  fprintf('Warning: There is no events for time change!\n')
else           % at least one event occurrence
  n = numel(t);
  if ~iscolumn(t)
    t = t';
  end
  w  = [t(1)-StartTime; diff(t)]; % original waiting times

  % -- transform to time changed waiting times v (normalized triggering kernel part) --
  v_h = zeros(n,1);
  SS  = zeros(n,1); SS(1) = 1;

  v_h(1) = 0;
  if n >=1
    for i = 2:n
      SS(i)  = exp(-bet*w(i))*SS(i-1)+1;
      v_h(i) = (1-exp(-bet*w(i)))*SS(i-1);
    end
  end

  % -- integrated triggering kernel part of the intensity over [0,t(end)]
  H = sum(v_h);
end


