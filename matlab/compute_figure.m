function [data,param] = compute_figure(data,param)
% compute stability figures for given parameterization

data.rKtheta = zeros(param.aux.ni,param.aux.nj);
data.EZ.p1bound = zeros(param.aux.ni,param.aux.nj);

for i = 1 : param.aux.ni
    for j = 1 : param.aux.nj

        p = param;
        p.(data.labeli) = data.veci(i);
        p.(data.labelj) = data.vecj(j);
        
        output = param.aux.simulation_routine(p);
        data.rKtheta(i,j) = p.BETA*(output.RC)^(1-1/p.PSI);
        data.EZ.p1bound(i,j) = p.BETA*(output.MC)^(1-1/p.PSI);
    end
    fprintf('i = %d out of %d done.\n',i,param.aux.ni);
end

% -------------------------------------------------------------------------
% plot figure and store data for later plotting

if (param.aux.loadpreviousvalues)
    load (sprintf('%s_data.mat',param.aux.filename));
end

fig = figure; hold on;
contour(data.vecj,data.veci,data.EZ.p1bound,param.aux.EZ_contour_levels,'ShowText','on');
xlabel(data.labelj);
ylabel(data.labeli);
plot(param.(data.labelj),param.(data.labeli),'k.','MarkerSize',16); 
text(param.(data.labelj),param.(data.labeli),sprintf(' \\leftarrow %s',param.aux.modelname))
title('Probability one test value \beta M_C^{1-1/\psi}');

savefig(fig,sprintf('%sA.fig',param.aux.filename));
saveas(fig,sprintf('%sA.pdf',param.aux.filename));

fig = figure; hold on;
contour(data.vecj,data.veci,data.rKtheta,param.aux.contour_levels,'ShowText','on');
xlabel(data.labelj);
ylabel(data.labeli);
plot(param.(data.labelj),param.(data.labeli),'k.','MarkerSize',16); 
text(param.(data.labelj),param.(data.labeli),sprintf(' \\leftarrow %s',param.aux.modelname))
title('Spectral radius test value r(K)^{1/\theta}');

savefig(fig,sprintf('%sB.fig',param.aux.filename));
saveas(fig,sprintf('%sB.pdf',param.aux.filename));

save(sprintf('%s_data.mat',param.aux.filename),'param','data');

end