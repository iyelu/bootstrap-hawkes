function output = simu_PRRB(Seq,param)
  % --------------------------------------------------
  % Purpose: Simulate one bootstrap sample for 
  %          Parametric (PR) recursive (RB) bootstrap
  % --------------------------------------------------
  % Input:
  %   Seq: structure variable for one data sample
  %          Seq.Param (parameter in the data DGP)
  %          Seq.StartTime (-M)
  %          Seq.StopTime (T)
  %          Seq.Points (simulated event times)
  %          Seq.NumPoints  
  %          Seq.NumPositivePoints  
  %   param: parameters in bootstrap DGP)
  %          mu, alp, bet
  % --------------------------------------------------
  % Call function:  simu_thinning.mlx
  % --------------------------------------------------
  % Ye Lu, 2020-10-24
  % ye.Lu1@sydney.edu.au
  % --------------------------------------------------

  Model_intensity = @HawkesExp_Intensity;

  output = struct('Param', [], ...
    'StartTime', [], ...
    'StopTime', [], ...
    'Points', [], ...
    'NumPoints', [], ...
    'NumPositivePoints', [] ...
    );

  % bootstrap simulation option
  t = Seq.Points; % MC sample
  options.InitialHistory = t(t<0); % original sample history 
  options.StartTime = 0; 
  options.StopTime  = Seq.StopTime;
  options.print='off';

  % simulate the bootstrap sample
  S = simu_thinning(Model_intensity,param,options);

  % concatenate the points in the data burn-in period and simulated points
  Points = [options.InitialHistory; S.Points];

  %% Output %%
  output.Param     = param;
  output.StartTime = Seq.StartTime;
  output.StopTime  = Seq.StopTime;
  output.Points    = Points;
  output.NumPoints = numel(Points);
  output.NumPositivePoints = numel(Points(Points>0));
