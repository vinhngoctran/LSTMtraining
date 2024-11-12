function [MaskDat, ClimateDat,TargetDat] = formdata_EXP2_Glob(Climate,OBS,TimeTestVal)
%%
TimeSeries = Climate.date;
idx = find(TimeSeries==TimeTestVal(1));idy = find(TimeSeries==TimeTestVal(end));

preClimateDat = [Climate.total_precipitation_sum, Climate.temperature_2m_mean,Climate.surface_net_solar_radiation_mean,...
    Climate.surface_net_thermal_radiation_mean,Climate.snow_depth_water_equivalent_mean,Climate.surface_pressure_mean]; % [dayl prcp srad tmax tmin vp]
TargetDat = OBS;
ClimateDat = [];
for i=1:size(preClimateDat,2)
    DataWT = wavelet_decomposition(preClimateDat(:,i));
    ClimateDat = [ClimateDat, DataWT];
end
MaskDat(1:numel(TimeSeries),1) = 1;
MaskDat(idx:idy-1) = 2; % Validation set
MaskDat(idy:end) = 3; % Test set
TargetDat(TargetDat<0)=NaN;
end
