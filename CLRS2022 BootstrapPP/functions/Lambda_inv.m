function t_star = Lambda_inv(s_star,t,s,param)
  % -----------------------
  % Purpose:
  %  FIB - transform s_star to bootstrap event times t_star 
  %        t_star = Lambda^(-1)(s_star; t,theta)
  % -----------------------
  % Note: 
  %  Only works for Hawkes exponential kernel
  % -----------------------
  % Ye Lu, 2020-11-03
  % ye.lu1@sydney.edu.au
  % -----------------------

  % make sure the original event times form column vector
  if ~iscolumn(t)
    t = t';
  end 
  if ~iscolumn(s)
    s = s';
  end 

  mu=param(1); alp=param(2); bet=param(3);

  if ~any(s>=s_star)  % all s < s_star
    v = s_star-s(end);
    base = t(end);
    data = t;
  % elseif ~any((s>=s_star)-1)  % s_star <= all s
  %   v = s_star;
  %   base = 0;
  else
    j = find(diff(s >= s_star)==1);
    v = s_star-s(j);
    base = t(j);
    data = t(1:j);
  end

  backtimes = data(end)-data;
  S = sum(exp(-bet*backtimes));

  fun = @(x)tc_fun(x,S,v,param);
  x0 = v/mu;
  sol = fzero(fun,x0);
  t_star=base+sol;
end

%% Helper Function
function output = tc_fun(x,S,v,para)
  % Input:
  %   x:       waiting time of Hawkes Point Process
  %   para:    parameters in Hawkes process with exponential kernel
  %   S:       S_k function which has iterative structure 
  %   v:       transformed waiting time 

  mu = para(1); alp = para(2); bet = para(3);
  output = mu * x + alp/bet * (1-exp(-bet*x))*S - v;
end
