function [NSE, KGE, RMSE,PE, NSEevent] = computemetric(simulated, observed)
% Remove NaN values
valid = all(~isnan([observed, simulated]), 2);
observed = observed(valid);
simulated = simulated(valid, :);

% Preallocate arrays
[enKGE, enNSE, enRMSE] = deal(nan(size(simulated, 2), 1));

% Compute metrics for each simulation
for i = 1:size(simulated, 2)
    try
        enKGE(i) = computeKGE(observed, simulated(:, i));
        enNSE(i) = computeNash(observed, simulated(:, i));
        enRMSE(i) = computeRMSE(observed, simulated(:, i));
        enPE(i) = computeflooderror(observed, simulated(:, i));
    catch
        enKGE(i) = NaN;
        enNSE(i) = NaN;
        enRMSE(i) = NaN;
        enPE(i) = NaN;
    end
    [flood_events, event_peaks, event_dates] = find_flood_events(observed,95);
    for j=1:numel(flood_events)
        NSEvalue(j,i) = computeNash(observed(event_dates{j}),simulated(event_dates{j},i));
    end
end

NSE = mean(enNSE, 'omitnan');
KGE = mean(enKGE, 'omitnan');
RMSE = mean(enRMSE, 'omitnan');
try
NSEevent = mean(NSEvalue,'omitnan');
catch
    NSEevent=NaN;
end
try
    PE = mean(enPE, 'omitnan');
catch
    PE = NaN;
end
end

function KGE = computeKGE(observed, simulated)
sdSimulated = std(simulated);
sdObserved = std(observed);
meanSimulated = mean(simulated);
meanObserved = mean(observed);
r = corr(observed, simulated);
relvar = sdSimulated / sdObserved;
bias = meanSimulated / meanObserved;
KGE = 1 - sqrt(0.8 * ((r - 1)^2) + 0.2 * ((bias - 1)^2));
end

function NS = computeNash(observed, simulated)
meanObserved = mean(observed);
SSres = sum((observed - simulated).^2);
SStot = sum((observed - meanObserved).^2);
NS = 1 - SSres / SStot;
end

function RMSE = computeRMSE(observed, simulated)
N = numel(simulated);
RMSE = sqrt(sum((observed - simulated).^2) / N);
end

function [PE, T2P] = computeflooderror(Obs, Sim)
try
    idx = find(Obs==max(Obs));
    idy = find(Sim(idx(1)-10:idx(1)+10)==max(Sim(idx(1)-10:idx(1)+10)));
    PE = (max(Sim(idx(1)-10:idx(1)+10))-max(Obs))/max(Obs)*100; % Percentage
    T2P = idy-11; % day
catch
    PE = NaN;T2P = NaN;
end
end