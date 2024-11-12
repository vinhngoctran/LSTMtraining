function PE = computemetric_PE(simulated, observed)
% Remove NaN values
% valid = all(~isnan([observed, simulated]), 2);
% observed = observed(valid);
% simulated = simulated(valid, :);

% Preallocate arrays

% Compute metrics for each simulation
for i = 1:size(simulated, 2)
    [flood_events, event_peaks, event_dates] = find_flood_events(observed,99);
    for j=1:numel(flood_events)
        try
        PE(j,i) = computeflooderror(observed(event_dates{j}),simulated(event_dates{j},i));
        catch
            PE(j,i) = NaN;
        end
    end
end


end

function [PE, T2P] = computeflooderror(Obs, Sim)
% try
%     idx = find(Obs==max(Obs));
%     idy = find(Sim(idx(1)-10:idx(1)+10)==max(Sim(idx(1)-10:idx(1)+10)));
%     PE = (max(Sim(idx(1)-10:idx(1)+10))-max(Obs))/max(Obs)*100; % Percentage
%     T2P = idy-11; % day
% catch
%     PE = NaN;T2P = NaN;
% end

try
    idx = find(Obs==max(Obs));
    idy = find(Sim(idx(1)-10:idx(1)+10)==max(Sim(idx(1)-10:idx(1)+10)));
    PE = (max(Sim(idx(1)))-max(Obs))/max(Obs)*100; % Percentage
    T2P = idy-11; % day
catch
    PE = NaN;T2P = NaN;
end

end