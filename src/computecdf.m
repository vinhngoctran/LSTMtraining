function valuecdf = computecdf(xxx,observation)
observation = observation(~isnan(observation));
for i=1:numel(xxx)
    valuecdf(i,1) = numel(find(observation<=xxx(i)))/numel(observation);
end
end