function R2 = ComputeR2(observed, predicted)
    % Remove NaN values from both arrays
    validIndices = ~isnan(observed) & ~isnan(predicted);
    observed = observed(validIndices);
    predicted = predicted(validIndices);
    
    % Check if there are enough valid data points
    if length(observed) < 2
        % warning('Not enough valid data points to compute R²');
        R2 = NaN;
        return;
    end
    
    % Compute R²
    SSres = sum((observed - predicted).^2);
    SStot = sum((observed - mean(observed)).^2);
    
    R2 = 1 - SSres / SStot;
    
    % Check for invalid R² (can happen with very poor fits)
    if R2 < 0
        % warning('Negative R² computed. This indicates a very poor fit.');
    end
end