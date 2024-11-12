function [R2, ObsCARAVAN,ObsGRDC] = mergeOBS(Obs,DAteAI,Climate)
idx = find(DAteAI==Climate.date(1)); idy = find(DAteAI==Climate.date(end));
ObsGRDC = Obs(idx:idy);
ObsCARAVAN = Climate.streamflow;
R2 = ComputeR2(ObsGRDC,ObsCARAVAN);

end