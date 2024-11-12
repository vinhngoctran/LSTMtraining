function [MaskDat, ClimateDat,TargetDat] = formdata_EXP4_Glob(Climate,OBS,ExtendedFor,Qopt,TimeTestVal)
%%
TimeSeries = Climate.date;
idx = find(TimeSeries==TimeTestVal(1));idy = find(TimeSeries==TimeTestVal(end));

ClimateDat = [Climate.total_precipitation_sum, Climate.temperature_2m_mean,Climate.surface_net_solar_radiation_mean,...
    Climate.surface_net_thermal_radiation_mean,Climate.snow_depth_water_equivalent_mean,Climate.surface_pressure_mean]; % [dayl prcp srad tmax tmin vp]
TargetDat = OBS;

MaskDat(1:numel(TimeSeries),1) = 1;
MaskDat(idx:idy-1) = 2; % Validation set
MaskDat(idy:end) = 3; % Test set

ExtendedForcing = [];
ExtendedTarget = [];
ExtendedMask = [];
for i=1:numel(ExtendedFor)
    preMaskDat(1:size(ExtendedFor{i}.total_precipitation_sum,1),1) = 1;
    preMaskDat(1:366) = NaN;
    preClimateDat = [ExtendedFor{i}.total_precipitation_sum, ExtendedFor{i}.temperature_2m_mean,ExtendedFor{i}.surface_net_solar_radiation_mean,...
    ExtendedFor{i}.surface_net_thermal_radiation_mean,ExtendedFor{i}.snow_depth_water_equivalent_mean,ExtendedFor{i}.surface_pressure_mean];

    ExtendedForcing = [ExtendedForcing;preClimateDat];
    ExtendedTarget = [ExtendedTarget;Qopt(:,i)];
    ExtendedMask = [ExtendedMask;preMaskDat];
end
MaskDat = [ExtendedMask;MaskDat];
ClimateDat = [ExtendedForcing;ClimateDat];
TargetDat = [ExtendedTarget;TargetDat];

TargetDat(TargetDat<0)=NaN;
end
