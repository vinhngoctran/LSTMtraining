clear all; close all; clc
addpath(genpath('Function'));

%%
clear all; close all; clc;
warning off
addpath(genpath(pwd));


% Datasetname
DATNAME = ls('1.Data/attributes');
DATNAME = DATNAME(3:end,:);

% Parameter bound and initial states
LB=[10,   1,  50,    1,  0.01,   0.01, 0.001, 0.01, 0.01 -2.0  0.1];
UB=[400,  4,  1000,  5,  1.00,   1.00, 0.30, 0.99,  0.99  4.0  7.0];
InStates=[0 0 0];   % The initial states of the SIMHYD-Snow model
k=0;

% Optimize parameters using SCE-U and EnKF
for id = 1:size(DATNAME,1)
    Locate = erase(DATNAME(id,:),' ');
    SUBID{id}.attributes = readtable(['1.Data/attributes/',Locate,'/','attributes_caravan_',Locate,'.csv']);
    SUBID{id}.hydro = readtable(['1.Data/attributes/',Locate,'/','attributes_hydroatlas_',Locate,'.csv']);
    SUBID{id}.info = readtable(['1.Data/attributes/',Locate,'/','attributes_other_',Locate,'.csv']);
    SUBID{id}.shapef = shaperead(['1.Data/shapefiles/',Locate,'/',Locate,'_basin_shapes.shp']);
    j=0;
    ErrorW{id}{1,1} = [];
    for sub = 1:size(SUBID{id}.attributes,1)
        SubName = SUBID{id}.attributes.gauge_id{sub};
        if exist(['3.Results/SCE/',SubName,'.mat'])==0
            try
                k=k+1
                Climate = readtable(['1.Data/csv/',Locate,'/',SubName,'.csv']);
                ModelInput = [Climate.total_precipitation_sum, Climate.potential_evaporation_sum,Climate.temperature_2m_max,Climate.temperature_2m_min,Climate.streamflow];
                [OptPars,Qopt,Metrics_opt]   = sceoptimization(ModelInput,InStates,LB,UB);      % Optimization with SCE-UA

                save(['3.Results/SCE/',SubName,'.mat'],'Climate','Metrics_opt',"ModelInput","OptPars","Qopt");
            catch
                j=j+1;
                ErrorW{id}{j,1} = SubName;
            end
        end
    end
end
save('Results_Gobal/Infor.mat','SUBID',"ErrorW");

%% Select overlap stations
load('E:\NWM21\Results8\R3_Nearing_sim_gauged.mat')
load('Results_Global\Infor.mat')
load('E:\NWM21\Results8\R2_Nearing_obs.mat');
NearingObs(NearingObs<0)=NaN;

DAteAI = [datetime(1980,1,1)]+days([1:1:size(NearingObs,1)]-1)';
k=0;
for i=1:numel(SUBID)
    for j=1:size(SUBID{i}.attributes,1)
        if exist(['E:\PUB\3.Results/SCE/',SUBID{i}.attributes.gauge_id{j},'.mat'])
            [idx_overlap, temp_distance(i,1)] = findclosestation_utm([SUBID{i}.info.gauge_lat(j),SUBID{i}.info.gauge_lon(j)],[BasinNearing.latitude, BasinNearing.longitude],1000);
            if ~isnan(idx_overlap) && max(NearingObs(:,idx_overlap))>0
                filename = ['E:\PUB\3.Results/SCE/',SUBID{i}.attributes.gauge_id{j},'.mat'];
                load(filename,'Climate');
                Obs = NearingObs(:,idx_overlap)/(SUBID{i}.info.area(j)*10^6/1000/86400);
                [R2, ObsCARAVAN,ObsGRDC] = mergeOBS(Obs,DAteAI,Climate);
                % pause
                if  R2 > 0.99
                    k=k+1
                    overlap_global(k,:) = [SUBID{i}.info.gauge_lat(j),SUBID{i}.info.gauge_lon(j), BasinNearing.calculated_drain_area(idx_overlap,1), idx_overlap, temp_distance(i,1), i];
                    Q_AI(:,k) = NearingSim{idx_overlap}(:,1);
                    Q_obs(:,k) = mean([ObsCARAVAN,ObsGRDC],2,'omitnan');
                    idname(k,:) = string(SUBID{i}.attributes.gauge_id{j});
                end
            end
        end
    end
end

save('Results_Global\R1_meta_AI.mat',"idname","Q_AI","overlap_global","DAteAI",'Q_obs');


%% Run SIMHYD-Snow with other forcing data
load('Results_Global\R1_meta_AI.mat',"idname");
nsize = 3;
InStates=[0 0 0];
for i=1:size(idname,1)
    i
    nidx = RandomForcing(i,size(idname,1),nsize);
    filename = ['E:\PUB\3.Results/SCE/',idname{i},'.mat'];
    load(filename,'OptPars');
    for j=1:nsize
        filename = ['E:\PUB\3.Results/SCE/',idname{nidx(j)},'.mat'];
        load(filename,'ModelInput','Climate');
        [SynQ(:,j),~,Inv]=SIMHYD_Snow(ModelInput,OptPars,InStates);
        ExtendedFor{j} = Climate;
    end
    filename = ['Results_Global/SYN/',num2str(i),'.mat'];
    save(filename,"SynQ",'nidx',"ExtendedFor");
    % end
end


%% EXP 1: Train normal (baseline) model
TimeTestVal = [datetime(2005,1,1),datetime(2014,1,1)];
for i=1:size(idname,1)
    i
    filename = ['E:\PUB\3.Results/SCE/',idname{i},'.mat'];
    load(filename,'Climate');

    [MaskDat, ClimateDat,TargetDat] = formdata_EXP1_Glob(Climate,Q_obs(:,i),TimeTestVal);
    Filename = ['Results_Global/EXP_1/Input/',num2str(i),'.mat'];
    save(Filename,"MaskDat","ClimateDat","TargetDat");
    % end
end

%% EXP 2: Train model with WT
for i=1:size(idname,1)
    i
    filename = ['E:\PUB\3.Results/SCE/',idname{i},'.mat'];
    load(filename,'Climate');

    [MaskDat, ClimateDat,TargetDat] = formdata_EXP2_Glob(Climate,Q_obs(:,i),TimeTestVal);
    Filename = ['Results_Global/EXP_2/Input/',num2str(i),'.mat'];
    save(Filename,"MaskDat","ClimateDat","TargetDat");
    % end
end


%%
% EXP 3: Train model with SimHYD output as extended input
for i=1:size(idname,1)
    i
    filename = ['E:\PUB\3.Results/SCE/',idname{i},'.mat'];
    load(filename,'Climate','Qopt');
    [MaskDat, ClimateDat,TargetDat] = formdata_EXP3_Glob(Climate,Q_obs(:,i),Qopt,TimeTestVal);
    Filename = ['Results_Global/EXP_3/Input/',num2str(i),'.mat'];
    save(Filename,"MaskDat","ClimateDat","TargetDat");
    % end
end


%%
% EXP 4: Train model with SimHYD output as extended output with random forcings from other basins and fine-tune
for i=1:size(idname,1)
    i
    filename = ['E:\PUB\3.Results/SCE/',idname{i},'.mat'];
    load(filename,'Climate');
    load(['Results_Global/SYN/',num2str(i),'.mat'])
    [MaskDat, ClimateDat,TargetDat] = formdata_EXP4_Glob(Climate,Q_obs(:,i),ExtendedFor,SynQ,TimeTestVal);
    Filename = ['Results_Global/EXP_4/Input/',num2str(i),'.mat'];
    save(Filename,"MaskDat","ClimateDat","TargetDat");
    % end
end


%%
% EXP 5: Train model with extreme events
for i=1:size(idname,1)
    i
    filename = ['E:\PUB\3.Results/SCE/',idname{i},'.mat'];
    load(filename,'Climate');

    [MaskDat, ClimateDat,TargetDat] = formdata_EXP5_Glob(Climate,Q_obs(:,i),TimeTestVal);
    Filename = ['Results_Global/EXP_5/Input/',num2str(i),'.mat'];
    save(Filename,"MaskDat","ClimateDat","TargetDat");
    % end
end



%%
% EXP 6: Train model with data intergration
for i=1:size(idname,1)
    i
    filename = ['E:\PUB\3.Results/SCE/',idname{i},'.mat'];
    load(filename,'Climate');

    [MaskDat, ClimateDat,TargetDat] = formdata_EXP6_Glob(Climate,Q_obs(:,i),TimeTestVal);
    Filename = ['Results_Global/EXP_6/Input/',num2str(i),'.mat'];
    save(Filename,"MaskDat","ClimateDat","TargetDat");
    % end
end



%% Analysis ================================================================================
% 1: Compute NSE, KGE, RMSE, NSEevent, KGEevent, RMSEevent, PE, T2P
clear all; clc
load('Results_Global\R1_meta_AI.mat',"idname","Q_AI","overlap_global","DAteAI",'Q_obs');
TimeTestVal = [datetime(2005,1,1),datetime(2014,1,1)];
idx = find(DAteAI==TimeTestVal(2));
filename = ['E:\PUB\3.Results/SCE/',idname{1},'.mat'];
load(filename,'Climate');
idx1 = find(Climate.date==TimeTestVal(2));
idy = find(DAteAI==Climate.date(end));

for i=1:size(Q_obs,2)
    i
    filename = ['E:\PUB\3.Results/SCE/',idname{i},'.mat'];load(filename,'Qopt')
    [NSE(i,1),KGE(i,1),RMSE(i,1),PE(i,1),NSE_event(i,1)] = computemetric(Qopt(idx1:end),Q_obs(idx1:end,i));
    [NSE(i,2),KGE(i,2),RMSE(i,2),PE(i,2),NSE_event(i,2)] = computemetric(Q_AI(idx:idy,i),Q_obs(idx1:end,i));    
    for j=1:6
        clearvars y_pred y_true
        try
        Filename = ['Results_Global/EXP_',num2str(j),'/Results/',num2str(i),'.mat'];
        if exist(Filename)
        load(Filename)
        [NSE(i,2+j),KGE(i,2+j),RMSE(i,2+j),PE(i,2+j),NSE_event(i,2+j)] = computemetric(y_pred,y_true);
        Smodel{i,1}(:,j) = y_pred;
        else
            KGE(i,2+j)=NaN;RMSE(i,2+j)=NaN;PE(i,2+j)=NaN;NSE(i,2+j) = NaN;
        end
        catch 
            KGE(i,2+j)=NaN;RMSE(i,2+j)=NaN;PE(i,2+j)=NaN;NSE(i,2+j) = NaN;
        end
    end
end
save('Results_Global\R3_Comparison.mat',"PE","RMSE","KGE","NSE",'NSE_event')
save('Results_Global\Smodel.mat','Smodel')
%% Compute flood peak error
clear all; clc
load('Results_Global\R1_meta_AI.mat',"idname","Q_AI","overlap_global","DAteAI",'Q_obs');
TimeTestVal = [datetime(2005,1,1),datetime(2014,1,1)];
idx = find(DAteAI==TimeTestVal(2));
filename = ['E:\PUB\3.Results/SCE/',idname{1},'.mat'];
load(filename,'Climate');
idx1 = find(Climate.date==TimeTestVal(2));
idy = find(DAteAI==Climate.date(end));

for i=1:size(Q_obs,2)
    i
    filename = ['E:\PUB\3.Results/SCE/',idname{i},'.mat'];load(filename,'Qopt')   
    try
    PE{i,1}(:,1) = computemetric_PE(Qopt(idx1:end),Q_obs(idx1:end,i));
    PE{i,1}(:,2) = computemetric_PE(Q_AI(idx:idy,i),Q_obs(idx1:end,i));
    
    
    for j=1:6
        try
        Filename = ['Results_Global/EXP_',num2str(j),'/Results/',num2str(i),'.mat'];
        load(Filename)
        PE{i,1}(:,2+j) = computemetric_PE(y_pred,y_true);
        catch
            PE{i,1}(1:end,2+j)=NaN;
        end
    end
    catch

    end
end
save('Results_Global\R4_Comparison_PE.mat',"PE")


%% Compare Global vs Regional: Ungauged basins
% https://hess.copernicus.org/articles/23/5089/2019/
% https://hess.copernicus.org/articles/27/139/2023/#section7
% https://hess.copernicus.org/articles/25/5517/2021/#section6
clear; clc
load('E:\PUB_realistic/RESULTS2\R3_Nearing_sim.mat',"BasinNearing","NearingSim","StartTime");
load('E:\PUB_realistic/RESULTS2\R2_Nearing_obs.mat', 'NearingObs')

% Read Thomas data: CAMELS-GB
Qsim = ncread('E:\PUB_realistic/Data/RegionalPaper/Thomas/preds.nc','LSTM');
Qobs = ncread('E:\PUB_realistic/Data/RegionalPaper/Thomas/preds.nc','obs');
Time = ncread('E:\PUB_realistic/Data/RegionalPaper/Thomas/preds.nc','time');
TimeDATE = datetime(1988,1,1)+days(Time);
StationID = ncread('E:\PUB_realistic/Data/RegionalPaper/Thomas/preds.nc','station_id');
InfoGB = xlsread('E:\PUB_realistic/Data\RegionalPaper\Thomas\Info\data\CAMELS_GB_topographic_attributes.csv');

for i=1:numel(StationID)
    idx = find(InfoGB==StationID(i));
    InforSelec(i,:)=InfoGB(idx,:);
end
% save('RESULTS2\R8_Regional_GB.mat','Qobs','Qsim',"TimeDATE","StationID",'InforSelec');

k=0;
for i=1:size(StationID,1)
     [overlapdata(i,1),Data, AREA] = findoverlapdata(NearingObs,BasinNearing,NearingSim,StartTime(1),Qsim(:,i),Qobs(:,i),TimeDATE,[InforSelec(i,3:4),InforSelec(i,8)]);
     if ~isnan(overlapdata(i,1))
        k=k+1
        DataAll{k,1} = Data;
        OverlapInfo(k,:) = InforSelec(i,:);
        AREA_all(k,:) = AREA;
     end
end
save('Results_Global\R2_Regional_GB_ungauged_.mat','Qobs','Qsim',"TimeDATE","StationID",'InforSelec','DataAll','OverlapInfo','AREA_all','overlapdata');

% Read Kratzert data: CAMELS-US - Ungauged basins: lstm_seed111
% Run 'Read_KratzertData2.py' to load data
clear; clc
load('E:\PUB_realistic/RESULTS2\R3_Nearing_sim.mat',"BasinNearing","NearingSim","StartTime");
load('E:\PUB_realistic/RESULTS2\R2_Nearing_obs.mat', 'NearingObs')
load('E:\PUB_realistic/RESULTS2\R8_Regional_US.mat')  
InfoUS = readtable('E:\PUB_realistic\Data\camels\camels_attributes_v2.0/camels_topo.txt');
InfoUS = table2array(InfoUS);
Basin = load('E:\PUB_realistic/Data/single-basin-vs-regional-model/basin_lists/531_basin_list.txt');
Time = datetime(Time,'InputFormat','yyyy/MM/dd');
for i=1:numel(Basin)
    idx = find(InfoUS(:,1)==Basin(i));
    InforSelec(i,:)=InfoUS(idx,:);
end
k=0;
for i=1:size(InforSelec,1)
     [overlapdata_US(i,1),Data, AREA] = findoverlapdata(NearingObs,BasinNearing,NearingSim,StartTime(1),all_qsim(i,:),all_qobs(i,:),Time,[InforSelec(i,2:3), InforSelec(i,6)]);
     if ~isnan(overlapdata_US(i,1))
        k=k+1;
        DataAll{k,1} = Data;
        OverlapInfo(k,:) = InforSelec(i,:);
        AREA_all(k,:) = AREA;
     end
end
save('Results_Global\R2_Regional_US_ungauged.mat','all_qsim','all_qobs','InforSelec','Time','DataAll','OverlapInfo','AREA_all','overlapdata_US');  


% CONUS - Gauged basins
clear; clc
load('E:\PUB_realistic/RESULTS2\R3_Nearing_sim_gauged.mat',"BasinNearing","NearingSim","StartTime");
load('E:\PUB_realistic/RESULTS2\R2_Nearing_obs.mat', 'NearingObs')
load('Data/R1_Krat_data.mat')
load('Data/R1_Krat_meta.mat');
[AlllCamel,camelinfo] = loadcamel();
for i=1:size(Basins,1)
    
    idx = find(camelinfo(:,1)==str2num(Basins(i,:)));
    InforSelec(i,:)=camelinfo(idx,:);
end
Time = [datetime(1989,10,1):days(1):datetime(1999,9,30)]';
%
k=0;
for i=1:size(InforSelec,1)
     [overlapdata_US(i,1),Data_M, AREA] = findoverlapdata(NearingObs,BasinNearing,NearingSim,StartTime(1),mean(Kratz_M_SIM(:,:,i),2),Kratz_M_OBS(:,1,i),Time,[InforSelec(i,2:3), InforSelec(i,6)]);
     [overlapdata_US(i,1),Data_S, AREA] = findoverlapdata(NearingObs,BasinNearing,NearingSim,StartTime(1),mean(Kratz_S_SIM(:,:,i),2),Kratz_M_OBS(:,1,i),Time,[InforSelec(i,2:3), InforSelec(i,6)]);
     
     if ~isnan(overlapdata_US(i,1))
        k=k+1
        DataAll{k,1} = Data_S;
        DataAll{k,2} = Data_M;
        OverlapInfo(k,:) = InforSelec(i,:);
        AREA_all(k,:) = AREA;
     end
end

save('Results_Global\R2_Regional_US_gauged.mat','InforSelec','Time','DataAll','OverlapInfo','AREA_all','overlapdata_US');  


%% FIGURE
plot_figure_2
plot_figure_3
plot_figure_4
