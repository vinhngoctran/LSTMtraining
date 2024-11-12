function [MaskDat2] = selecthighflow(MaskDat,TargetDat)
    % Calculate the 90th percentile threshold
    ThresholdFlow = prctile(TargetDat, 90);
    MaskDat2 = zeros(size(MaskDat));
    % Initialize variables
    n = size(TargetDat, 1);
    event_length = 30;
    peak_day = 15;
    
    % Iterate through the data
    for i = peak_day:(n - event_length + peak_day)
        % Check if the flow at the peak day exceeds the threshold
        if TargetDat(i) > ThresholdFlow
            % Check if it's the maximum in the 30-day window
            window = TargetDat(i-peak_day+1 : i-peak_day+event_length);
            if TargetDat(i) == max(window)
                % Mark the entire 30-day event in MaskDat
                MaskDat2(i-peak_day+1 : i-peak_day+event_length,1) = 1;
                
                % Ensure no overlapping events
                if i + 1 <= n
                    i = i + event_length - peak_day;
                end
            end
        end
    end
    for i=1:n
        if MaskDat2(i)==1 && MaskDat(i)==2
            MaskDat2(i)=2;
        elseif MaskDat(i)==3
            MaskDat2(i)=3;
        end
    end

end