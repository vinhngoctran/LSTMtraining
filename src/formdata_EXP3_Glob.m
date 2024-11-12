function [MaskDat, ClimateDat,TargetDat] = formdata_EXP3_Glob(Climate,OBS,Qopt,TimeTestVal)
%%
TimeSeries = Climate.date;
idx = find(TimeSeries==TimeTestVal(1));idy = find(TimeSeries==TimeTestVal(end));

ClimateDat = [Climate.total_precipitation_sum, Climate.temperature_2m_mean,Climate.surface_net_solar_radiation_mean,...
    Climate.surface_net_thermal_radiation_mean,Climate.snow_depth_water_equivalent_mean,Climate.surface_pressure_mean,Qopt]; % [dayl prcp srad tmax tmin vp]
TargetDat = OBS;

MaskDat(1:numel(TimeSeries),1) = 1;
MaskDat(idx:idy-1) = 2; % Validation set
MaskDat(idy:end) = 3; % Test set
TargetDat(TargetDat<0)=NaN;
end
