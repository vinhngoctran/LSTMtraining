function plot_figure_2()
%%
clc;  clear all

close all;
load('Results_Global\R3_Comparison.mat',"PE","RMSE","KGE","NSE",'NSE_event')
load('Results_Global\R1_meta_AI.mat',"idname","Q_AI","overlap_global","DAteAI",'Q_obs');
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
scatter(overlap_global(:,2), overlap_global(:,1), 5, 'filled', 'MarkerEdgeColor', 'none','MarkerFaceColor','k','Marker','o');
set(axes1,'Color','none','Linewidth',1.5,'FontSize',13,'Xcolor','none','Ycolor','none');
% title("a",'FontSize',16,'VerticalAlignment','top');axes1.TitleHorizontalAlignment = 'left';


k=0;
for i=1:2
    for j=1:4
        k=k+1;
        POSS(k,:) = [0.08+(j-1)*0.24 0.38-(i-1)*0.3 0.18 0.2];
    end
end

TTT = ["a","b","c","d","e","f","g","h"];
for i=1:6
    axes1 = axes('Parent',figure1,...
        'Position',POSS(i,:)); hold on; box on
    scatter(NSE(:,2),NSE(:,i+2),'MarkerFaceColor',[0.5 0.5 0.5],'MarkerFaceAlpha',0.5,'MarkerEdgeColor','k','SizeData',10)
    scatter(NSE_event(:,2),NSE_event(:,i+2),100,'Marker','+','MarkerFaceColor','b','MarkerFaceAlpha',0.5,'MarkerEdgeColor','b')
    
    plot([0.5 0.5],[-4 1],'LineStyle','--','Color','r','LineWidth',1.5)
    plot([-4 1],[0.5 0.5],'LineStyle','--','Color','r','LineWidth',1.5)
    plot([-4 1],[-4 1],'Color','r','LineWidth',1.5)
    xlabel('NSE_{G} [-]');xlim([0 1]);ylim([0 1])
    ylabel(['NSE_{S-',num2str(i),'} [-]']);
    set(axes1,'Linewidth',1.5,'FontSize',11,...
        'Xtick',[-4:0.5:1],'Ytick',[-4:0.5:1]);
    title(TTT(i),'FontSize',13,'VerticalAlignment','baseline');axes1.TitleHorizontalAlignment = 'left';
end

axes1 = axes('Parent',figure1,...
        'Position',POSS(7,:)); hold on; box on
% axes1 = axes('Parent',figure1,...
%         'Position',POSS(7,:)+[0 0 0.24 0]); hold on; box on
for i=1:7
    xxx = [0:0.01:1];
    NSE_cdf = computecdf(xxx,NSE(:,i+1));
    if i==1
       ll{i}= plot(xxx,NSE_cdf,'LineWidth',1.5,'DisplayName','G','Color','k');
    else
       ll{i}= plot(xxx,NSE_cdf,'LineWidth',1.5,'DisplayName',['S-',num2str(i-1)]);
    end
end
for j=1:4
    plot([LevelNSE(j) LevelNSE(j)],[0 1],'LineStyle','--','Color',[0.5 0.5 0.5],'LineWidth',1)
    % if i==4
        text(LevelNSE(j)-0.06,0.05,LevelName(j),'Rotation',90,'Color',[0.5 0.5 0.5])
    % end
end

ylabel('CDF [-]');
% legend([ll{1},ll{2},ll{3},ll{4},ll{5},ll{6},ll{7}],'Box','off','Position',[0.567365809607838 0.178730635610814 0.146341462142584 0.091078064403274],...
%     'NumColumns',2)
xlim([0 1]);ylim([0 1]);xlabel('NSE [-]')
set(axes1,'Linewidth',1.5,'FontSize',11);
    title(TTT(7),'FontSize',13,'VerticalAlignment','baseline');axes1.TitleHorizontalAlignment = 'left';

    axes1 = axes('Parent',figure1,...
        'Position',POSS(8,:)); hold on; box on
% axes1 = axes('Parent',figure1,...
%         'Position',POSS(7,:)+[0 0 0.24 0]); hold on; box on
for i=1:7
    xxx = [0:0.01:1];
    NSE_cdf = computecdf(xxx,NSE_event(:,i+1));
    if i==1
       ll{i}= plot(xxx,NSE_cdf,'LineWidth',1.5,'DisplayName','G','Color','k');
    else
       ll{i}= plot(xxx,NSE_cdf,'LineWidth',1.5,'DisplayName',['S-',num2str(i-1)]);
    end
end
for j=1:4
    plot([LevelNSE(j) LevelNSE(j)],[0 1],'LineStyle','--','Color',[0.5 0.5 0.5],'LineWidth',1)
    % if i==4
        % text(LevelNSE(j)-0.06,0.05,LevelName(j),'Rotation',90,'Color',[0.5 0.5 0.5])
    % end
end

ylabel('CDF [-]');
legend([ll{1},ll{2},ll{3},ll{4},ll{5},ll{6},ll{7}],'Box','off','Position',[0.486064998038362 0.00184091967530107 0.502032516175896 0.0260223042049076],...
    'Orientation','horizontal')
xlim([0 1]);ylim([0 1]);xlabel('NSE [-]')
set(axes1,'Linewidth',1.5,'FontSize',11);
    title(TTT(8),'FontSize',13,'VerticalAlignment','baseline');axes1.TitleHorizontalAlignment = 'left';

exportgraphics(figure1,"Figures/F3.jpg",'Resolution',600)
exportgraphics(figure1, "Figures/F3.pdf", 'ContentType', 'vector');
%% Analysis
clc
for i=1:size(NSE,1)
    DeltaNSE(i,:) = NSE(i,:) - NSE(i,2);
end
for i=1:6
disp(['N_{NSE} > G: ',num2str(numel(find(DeltaNSE(:,i+2)>0 ))/size(removeNaNRows(DeltaNSE(:,i+2)),1)*100)]);
end

for i=1:size(NSE,1)
    DeltaNSEe(i,:) = NSE_event(i,:) - NSE_event(i,2);
end
for i=1:6
disp(['Event_N_{NSE} > G: ',num2str(numel(find(DeltaNSEe(:,i+2)>0 ))/size(removeNaNRows(DeltaNSEe(:,i+2)),1)*100)]);
end

for i=1:8
disp(['M_G_{NSE} >= 0.5: ',num2str(numel(find(NSE(:,i)>=0.5 ))/size(removeNaNRows(NSE(:,i)),1)*100)]);
end

for i=1:8
disp(['M_G_{NSE} < 0.5: ',num2str(numel(find(NSE(:,i)<0.5 ))/size(removeNaNRows(NSE(:,i)),1)*100)]);
end

MaxNSE = max(NSE(:,3:end)');
disp(['N_{NSE} < 0.5: ',num2str(numel(find(MaxNSE<0.5)))]);

for i=1:8
    disp(['Nmodel: ',num2str(numel(removeNaNRows(NSE(:,i))))]);
end
end