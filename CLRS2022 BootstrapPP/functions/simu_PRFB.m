function output = simu_PRFB(Seq,param)
  % -------------------------------------------
  % Purpose: Simulate one bootstrap event time  
  %          Parametric (PR) fixed intensity (FB) bootstrap
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
  % Call function: 
  %     intensity/ModelHinfExp_Integrated_Intensity.m 
  %     simulation/Lambda_inv.m 
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

  % find all the transformed event times
  [~,s]= Model_Int_Intensity(t,M,param);
  s_star = s(m); t_star = t(m); t_star_all = [];

  % simulate bootstrap event times
  while t_star <=T 
    v_star = exprnd(1);
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

