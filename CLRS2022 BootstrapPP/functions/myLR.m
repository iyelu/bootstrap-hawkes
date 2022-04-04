function output = myLR(maxl_urs,maxl_res,chi2df,siglev)
  % Purpose: LR test 
  % -------
  % Inputs:
  %   maxl_urs: maximized unrestricted log-likelihood
  %   maxl_res: maximized restricted log-likelihood
  %   chi2df:   degree of freedome of limiting chi2 distribution 
  %   siglev:   level of significance of the test 
  % -------
  % Ye Lu, 2020-09-06
  % ye.lu1@sydney.edu.au
  % -------

  stat = 2*(maxl_urs - maxl_res);
  cv   = chi2inv(1-siglev, chi2df);
  rej  = stat > cv;     
  pval = 1-chi2cdf(stat,chi2df); % p-value

  output.sig  = siglev;
  output.stat = stat;
  output.cv   = cv;
  output.rej  = rej;
  output.pval = pval;
end

