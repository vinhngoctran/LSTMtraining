function Dataout = wavelet_decomposition(DataIn)
    % Ensure precipitation is a column vector
    DataIn = DataIn(:);
    
    % wavelet (e.g., 'db5' for Daubechies 5)
    wname = 'db5';
    WTlevel = 3;
    
    precipitation_interp = DataIn;
    nan_indices = isnan(precipitation_interp);
    precipitation_interp(nan_indices) = interp1(find(~nan_indices), precipitation_interp(~nan_indices), find(nan_indices), 'linear', 'extrap');
    
    % Perform wavelet decomposition on interpolated data
    [C, L] = wavedec(precipitation_interp, WTlevel, wname);

    A3 = appcoef(C, L, wname, 3);
    [D1, D2, D3] = detcoef(C, L, [1 2 3]);

    A3 = wrcoef('a', C, L, wname, 3);
    D1 = wrcoef('d', C, L, wname, 1);
    D2 = wrcoef('d', C, L, wname, 2);
    D3 = wrcoef('d', C, L, wname, 3);
    
    A3(nan_indices) = NaN;
    D1(nan_indices) = NaN;
    D2(nan_indices) = NaN;
    D3(nan_indices) = NaN;

    Dataout = [DataIn, A3, D1, D2, D3];
    
    % Find columns with all the same value or all NaN
    same_value_cols = all(Dataout == Dataout(1,:), 1) | all(isnan(Dataout), 1);
    Dataout(:, same_value_cols) = [];
    if isempty(Dataout)
        warning('All columns were removed. Returning original precipitation data.');
        Dataout = DataIn;
    end
end