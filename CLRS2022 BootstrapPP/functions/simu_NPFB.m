function output = simu_NPFB(Seq,param)
  % -------------------------------------------
  % Purpose: Simulate Bootstrap event time  
  %          Non-parametric (NP) fixed intensity (FB) bootstrap
  %          with mean scaling
  % -------------------------------------------
  % Input:
  %   Seq: structure variable for one data sample
  %          Seq.Param (parameter in the data DGP)
  %          Seq.StartTime (-M)
  %          Seq.StopTime (T)
  %          Seq.Points (simulated event times)
  %          Seq.NumPoints  
  %          Seq.NumPositivePoints  
  %   param: parameters in bootstrap DGP
  %          mu, alp, bet
  % -------------------------------------------
  % Ye Lu, 2020-11-02
  % ye.lu1@sydney.edu.au
  % -------------------------------------------

  Model_Int_Intensity = @HawkesExp_Integrated_Intensity;

  output = struct('Param', [], ...
    'StartTime', [], ...
    'StopTime', [], ...
    'Points', [], ...
    'NumPoints', [], ...
    'NumPositivePoints', [] ...
    );

  t = Seq.Points;  % data in (-M,T]
  T = Seq.StopTime; M = Seq.StartTime;
  InitialHistory = t(t<0); m = numel(InitialHistory);

  % time changed waiting times vi's for NP bootstrap
  % and all the transformed event times s
  [v, s] = Model_Int_Intensity(t,M,param);
  v    = v(m+1:end); % v_{m+1},v_{m+2},...,v_{m+n_T}
  v_sc = v/mean(v);  % scaled waiting times
  n    = numel(v_sc);

  % simulate bootstrap event times 
  s_star = s(m); t_star = t(m); t_star_all = [];
  while t_star <=T 
    v_star = v_sc(randi(n));
    s_star = s_star + v_star;
    t_star = Lambda_inv(s_star,t,s,param);

    if t_star <=T
        t_star_all = [t_star_all; t_star];
    end
  end 

  Points = [InitialHistory; t_star_all];

  %% Output %%
  output.Param     = param;
  output.StartTime = Seq.StartTime;
  output.StopTime  = Seq.StopTime;
  output.Points    = Points;
  output.NumPoints = numel(Points);
  output.NumPositivePoints = numel(Points(Points>0));
