function plot_heatscater(x,y,nbins,cb,LIMS,VarName)
% Generate sample data
% n = 1000;
% x = randn(n, 1);
% y = randn(n, 1);
if nargin < 6
    VarName = 'NA';
end
% % Calculate 2D histogram
% nbins = 50;
[N, edges_x, edges_y] = histcounts2(x, y, nbins);
xcenters = (edges_x(1:end-1) + edges_x(2:end)) / 2;
ycenters = (edges_y(1:end-1) + edges_y(2:end)) / 2;
% [X, Y] = meshgrid(xcenters, ycenters);
[X, Y] = meshgrid(linspace(min(x), max(x), nbins), linspace(min(y), max(y), nbins));
dx = diff(edges_x(1:2));
dy = diff(edges_y(1:2));
area = dx * dy;
density = N' / area;
density_interp = interp2(X, Y, density, x, y, 'linear');
scatter(x, y, 5, density_interp, 'filled','DisplayName',VarName);
% cmap = flipud(parula); % Invert parula colormap
colormap("parula");
if cb == 1
c = colorbar;
end
% c.Label.String = 'Density (points per unit area)';
caxis(LIMS);
end

