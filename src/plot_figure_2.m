function plot_figure_2()
%%
clc;  clear all

close all;

% CONUS-Gauged
load('Results_Global\R2_Regional_US_gauged.mat','InforSelec','Time','DataAll','OverlapInfo','AREA_all','overlapdata_US');
for i=1:size(OverlapInfo,1)
    NSEvalue{1,1}(i,1) = Nash(DataAll{i,1}(:,2),DataAll{i,1}(:,1));
    NSEvalue{1,1}(i,2) = Nash(DataAll{i,1}(:,4),DataAll{i,1}(:,3));

    NSEvalue{2,1}(i,1) = Nash(DataAll{i,2}(:,2),DataAll{i,2}(:,1));
    NSEvalue{2,1}(i,2) = Nash(DataAll{i,2}(:,4),DataAll{i,2}(:,3));

    [flood_events, event_peaks, event_dates] = find_flood_events(DataAll{i,1}(:,2));
    for j=1:numel(flood_events)
        NSEvalue{1,2}{i}(j,1) = Nash(DataAll{i,1}(event_dates{j},2),DataAll{i,1}(event_dates{j},1));
        NSEvalue{1,2}{i}(j,2) = Nash(DataAll{i,1}(event_dates{j},4),DataAll{i,1}(event_dates{j},3));

        NSEvalue{2,2}{i}(j,1) = Nash(DataAll{i,2}(event_dates{j},2),DataAll{i,2}(event_dates{j},1));
        NSEvalue{2,2}{i}(j,2) = Nash(DataAll{i,2}(event_dates{j},4),DataAll{i,2}(event_dates{j},3));
    end

    NSEvalue{1,3}(i,1) = mean(NSEvalue{1,2}{i}(:,1),'omitnan');
    NSEvalue{1,3}(i,2) = mean(NSEvalue{1,2}{i}(:,2),'omitnan');

    NSEvalue{2,3}(i,1) = mean(NSEvalue{2,2}{i}(:,1),'omitnan');
    NSEvalue{2,3}(i,2) = mean(NSEvalue{2,2}{i}(:,2),'omitnan');
end
LocationDat{1} = OverlapInfo;
LocationDat{2} = OverlapInfo;



% CONUS-Ungauged
load('Results_Global\R2_Regional_US_ungauged.mat','all_qsim','all_qobs','InforSelec','Time','DataAll','OverlapInfo','AREA_all','overlapdata_US');
for i=1:size(OverlapInfo,1)
    NSEvalue{3,1}(i,1) = Nash(DataAll{i,1}(:,2),DataAll{i,1}(:,1));
    NSEvalue{3,1}(i,2) = Nash(DataAll{i,1}(:,4),DataAll{i,1}(:,3));

    [flood_events, event_peaks, event_dates] = find_flood_events(DataAll{i,1}(:,2));
    for j=1:numel(flood_events)
        NSEvalue{3,2}{i}(j,1) = Nash(DataAll{i,1}(event_dates{j},2),DataAll{i,1}(event_dates{j},1));
        NSEvalue{3,2}{i}(j,2) = Nash(DataAll{i,1}(event_dates{j},4),DataAll{i,1}(event_dates{j},3));
    end

    NSEvalue{3,3}(i,1) = mean(NSEvalue{3,2}{i}(:,1),'omitnan');
    NSEvalue{3,3}(i,2) = mean(NSEvalue{3,2}{i}(:,2),'omitnan');
end
LocationDat{3} = OverlapInfo;


% CONUS-GB Ungauged
load('Results_Global\R2_Regional_GB_ungauged_.mat','Qobs','Qsim',"TimeDATE","StationID",'InforSelec','DataAll','OverlapInfo','AREA_all','overlapdata');
for i=1:size(OverlapInfo,1)
    NSEvalue{4,1}(i,1) = Nash(DataAll{i,1}(:,2),DataAll{i,1}(:,1));
    NSEvalue{4,1}(i,2) = Nash(DataAll{i,1}(:,4),DataAll{i,1}(:,3));

    [flood_events, event_peaks, event_dates] = find_flood_events(DataAll{i,1}(:,2));
    for j=1:numel(flood_events)
        NSEvalue{4,2}{i}(j,1) = Nash(DataAll{i,1}(event_dates{j},2),DataAll{i,1}(event_dates{j},1));
        NSEvalue{4,2}{i}(j,2) = Nash(DataAll{i,1}(event_dates{j},4),DataAll{i,1}(event_dates{j},3));
    end

    NSEvalue{4,3}(i,1) = mean(NSEvalue{4,2}{i}(:,1),'omitnan');
    NSEvalue{4,3}(i,2) = mean(NSEvalue{4,2}{i}(:,2),'omitnan');
end
LocationDat{4} = OverlapInfo;


%
load Col.mat % Color setup
LevelNSE = [0.5, 0.65, 0.75, 1];
LevelName = ["Unsatisfactory","Acceptable","Good","Very good"];

close all
figure1 = figure('OuterPosition',[300 10 1000 900]);
axes1 = axes('Parent',figure1,...
    'Position',[0.1 0.5 0.7 0.48]);hold on;
% Load geographic data
load coastlines;
mapshow(coastlon,coastlat, 'DisplayType', 'polygon', 'FaceColor', [0.8 0.8 0.8],'EdgeColor','none');hold on
scatter(LocationDat{1}(:,3), LocationDat{1}(:,2), 10, 'filled', 'MarkerEdgeColor', 'none','MarkerFaceColor','k','Marker','o');

scatter(LocationDat{4}(:,4), LocationDat{4}(:,3), 10, 'filled', 'MarkerEdgeColor', 'none','MarkerFaceColor','k','Marker','o');
set(axes1,'Color','none','Linewidth',1.5,'FontSize',13,'Xcolor','none','Ycolor','none');
% title("a",'FontSize',16,'VerticalAlignment','top');axes1.TitleHorizontalAlignment = 'left';



axes1 = axes('Parent',figure1,...
    'Position',[0.5 0.6 0.171747967479672 0.331474597273854]);hold on; box on
% Load geographic data
load coastlines;
mapshow(coastlon,coastlat, 'DisplayType', 'polygon', 'FaceColor', [0.8 0.8 0.8],'EdgeColor','none');hold on
scatter(LocationDat{1}(:,3), LocationDat{1}(:,2), 10, 'filled', 'MarkerEdgeColor', 'none','MarkerFaceColor','k','Marker','o');

scatter(LocationDat{4}(:,4), LocationDat{4}(:,3), 10, 'filled', 'MarkerEdgeColor', 'none','MarkerFaceColor','k','Marker','o');
% Add colorbar
xlim([min(LocationDat{4}(:,4))-1 max(LocationDat{4}(:,4))+1]);ylim([min(LocationDat{4}(:,3))-1 max(LocationDat{4}(:,3))+1])
set(axes1,'Linewidth',1,'FontSize',13,'XTickLabel',{},'YTickLabel',{},'XColor','k','YColor','k','Layer','top');


annotation(figure1,'arrow',[0.46849593495935 0.5],...
    [0.870127633209418 0.870127633209418],'Color','k','LineWidth',1);


% annotation(figure1,'textbox',...
%     [0.223560975609756 0.809169764560099 0.0284715447154472 0.0384138785625775],...
%     'FitBoxToText','off','LineWidth',1,...
%     'EdgeColor',Col(5,:));

annotation(figure1,'textbox',...
    [0.217463414634147 0.799256505576208 0.124 0.0755885997521686],...
    'LineWidth',1,...
    'FitBoxToText','off');

annotation(figure1,'textbox',...
    [0.421731707317076 0.848822800495663 0.0437154471544696 0.0495662949194547],...
    'Color',[0 0.447058823529412 0.741176470588235],...
    'LineWidth',1,...
    'FitBoxToText','off',...
    'EdgeColor','k');
%
annotation(figure1,'textbox',...
    [0.217479674796749 0.762081785014423 0.0569105677124931 0.0334572484428259],...
    'String',{'US32'},...
    'LineStyle','none','FontSize',13);
annotation(figure1,'textbox',...
    [0.6 0.890954151805006 0.0569105677124931 0.0334572484428259],...
    'String',{'GB139'},...
    'LineStyle','none','FontSize',13);

TTT = ["a US32-Gauged","b US32-Gauged","c US32-Ungauged","d GB139-Ungauged"];


for i=1:4
    axes1 = axes('Parent',figure1,...
        'Position',[0.07+(i-1)*0.24 0.34 0.18 0.2]); hold on; box on
    scatter(NSEvalue{i,1}(:,1),NSEvalue{i,1}(:,2),'MarkerFaceColor',[0.5 0.5 0.5],'MarkerFaceAlpha',0.5,'MarkerEdgeColor','k')
    % scatter(-4,median(NSEvalue{3,1}(:,3,i)),100,'Marker','square','MarkerFaceColor',[0.5 0.5 0.5],'MarkerEdgeColor','k')
    % scatter(median(NSEvalue{3,1}(:,2,i)),-4,100,'Marker','^','MarkerFaceColor',[0.5 0.5 0.5],'MarkerEdgeColor','k')

    scatter(NSEvalue{i,3}(:,1),NSEvalue{i,3}(:,2),100,'Marker','+','MarkerFaceColor','b','MarkerFaceAlpha',0.5,'MarkerEdgeColor','b')
    % scatter(-4,median(NSEvalue{3,3}(:,3,i)),100,'Marker','square','MarkerFaceColor','r','MarkerEdgeColor','k')
    % scatter(median(NSEvalue{3,3}(:,2,i)),-4,100,'Marker','^','MarkerFaceColor','r','MarkerEdgeColor','k')
    % median(NSEvalue{3,3}(:,2,i))
    plot([0.5 0.5],[-4 1],'LineStyle','--','Color','r','LineWidth',1.5)
    plot([-4 1],[0.5 0.5],'LineStyle','--','Color','r','LineWidth',1.5)
    plot([-4 1],[-4 1],'Color','r','LineWidth',1.5)
    xlabel('NSE_{G} [-]');xlim([0 1]);ylim([0 1])
    if i==1
        ylabel('NSE_{S} [-]');
    else
        ylabel('NSE_{R} [-]');
    end 
    set(axes1,'Linewidth',1.5,'FontSize',11,...
        'Xtick',[-4:0.5:1],'Ytick',[-4:0.5:1])
    % if i>1
    %     axes1.YTickLabel = {};
    % end
    % if i==1
        title(TTT(i),'FontSize',13,'VerticalAlignment','baseline');axes1.TitleHorizontalAlignment = 'left'

end


exportgraphics(figure1,"Figures/F2.jpg",'Resolution',600)
exportgraphics(figure1, "Figures/F2.pdf", 'ContentType', 'vector');

end