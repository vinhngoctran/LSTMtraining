function plot_figure_4()
%%
clc;  clear all

close all;
load('Results_Global\R4_Comparison_PE.mat',"PE")

PE_all = [];
for i=1:size(PE,1)
    PE_all = [PE_all;PE{i,1}];
end
PEabs = abs(PE_all);

close all
figure1 = figure('OuterPosition',[300 10 1000 900]);


k=0;
for i=1:2
    for j=1:3
        k=k+1;
        POSS(k,:) = [0.08+(j-1)*0.32 0.38-(i-1)*0.3 0.25 0.2];
    end
end

TTT = ["a","b","c","d","e","f","g","h"];
for i=1:6
    axes1 = axes('Parent',figure1,...
        'Position',POSS(i,:)); hold on; box on
    % scatter(PEabs(:,2),PEabs(:,i+2),'MarkerFaceColor',[0.5 0.5 0.5],'MarkerFaceAlpha',0.5,'MarkerEdgeColor','k','SizeData',10)
    
    x= PE_all(:,2);
    y= PE_all(:,i+2);
    if i==3
        cb=1;
    else
        cb=1;
    end
    LIMS = [0 2];
    plot_heatscater(x,y,50,cb,LIMS)
    % [X, Y] = meshgrid(linspace(min(x), max(x), 50), linspace(min(y), max(y), 50));
    % [density, ~] = ksdensity([x y], [X(:) Y(:)]);
    % density = reshape(density, size(X));
    % 
    % scatter(x, y, 10, interp2(X, Y, density, x, y), 'filled');
    % cmap = flipud(hot);
    % colormap(cmap);
    % if i==3
    %     colorbar
    %     caxis([min(density(:)), max(density(:))]);
    % end
    plot([-100 100],[-100 100],'Color','r','LineWidth',1.5)
    xlabel('PE_{G} [-]');xlim([-100 100]);ylim([-100 100])
    ylabel(['PE_{S-',num2str(i),'} [-]']);
    set(axes1,'Linewidth',1.5,'FontSize',11,...
        'Xtick',[-100:50:100],'Ytick',[-100:50:100],'Layer','top');
    title(TTT(i),'FontSize',13,'VerticalAlignment','baseline');axes1.TitleHorizontalAlignment = 'left';
end
exportgraphics(figure1,"Figures/F4.jpg",'Resolution',600)
exportgraphics(figure1, "Figures/F4.pdf", 'ContentType', 'vector');

for i=1:8
    disp(['Median PE: ',num2str(median(PE_all(:,i),'omitnan'))])
end
end