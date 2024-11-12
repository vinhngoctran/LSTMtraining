function [MaskDat, ClimateDat,TargetDat] = formdata_EXP6_Glob(Climate,OBS,TimeTestVal)
%%
TimeSeries = Climate.date;
idx = find(TimeSeries==TimeTestVal(1));idy = find(TimeSeries==TimeTestVal(end));

ClimateDat = [Climate.total_precipitation_sum, Climate.temperature_2m_mean,Climate.surface_net_solar_radiation_mean,...
    Climate.surface_net_thermal_radiation_mean,Climate.snow_depth_water_equivalent_mean,Climate.surface_pressure_mean]; % [dayl prcp srad tmax tmin vp]
ClimateDat(2:end,7) = OBS(1:end-1);
TargetDat = OBS;

MaskDat(1:numel(TimeSeries),1) = 1;
MaskDat(idx:idy-1) = 2; % Validation set
MaskDat(idy:end) = 3; % Test set
TargetDat(TargetDat<0)=NaN;
end
