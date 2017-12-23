% Recreates the plots in the paper for the Schorfheide, Song and Yaron (2017) model.
% Figures 2a, 2b, 3a, 3b

% Parameters set to produce quick results. For higher quality, refine
% parameters param.aux.M, param.aux.T,
% param.aux.Tdrop, param.aux.ni, param.aux.nj

tic;

% benchmark parameterization
param.GAMMA = 8.89;
param.BETA = 0.999;
param.PSI = 1.97;
param.MUC = 0.0016;
param.RHO = 0.987;
param.PHIZ = 0.215;
param.SIGMABAR = 0.0032;
param.PHIC = 1;
param.RHOHZ = 0.992;
param.SIGMAHZ = 0.0039^0.5;
param.RHOHC = 0.991;
param.SIGMAHC = 0.0096^0.5;

% auxiliary parameters - for the Monte Carlo simulation
param.aux.M = 10000;     % number of simultaneous draws
param.aux.T = 1000;      % number of periods
param.aux.Tdrop = 500;  % number of burn-in periods for the Monte Carlo

% -------------------------------------------------------------------------
% FIGURE 2
% auxiliary graph grid parameters
param.aux.contour_levels = [0.9992 0.9996 1 1.0004 1.0008 1.0012 1.0016];

% probability-one bound (EZ1989/MM2010) parameters
param.EZ.percentile = 0.95;
param.aux.EZ_contour_levels = [1 1.001 1.002 1.003 1.004 1.005 1.006];

% graph grid content - figure 2
data.labeli = 'MUC';
data.labelj = 'PSI';
param.aux.ni = 10;
param.aux.nj = 10;
data.veci = linspace(0.0012,0.005,param.aux.ni);
data.vecj = linspace(1.05,2.5,param.aux.nj);

% solution routine and output
param.aux.simulation_routine = str2func('compute_RC_SSY_model');
param.aux.filename = 'data_SSY_model_figure_2';
param.aux.modelname = 'Schorfheide, Song and Yaron';
param.aux.loadpreviousvalues = 0;

disp('Computing Figure 2');
compute_figure(data,param);

% -------------------------------------------------------------------------
% FIGURE 3
% auxiliary graph grid parameters
param.aux.contour_levels = [0.9975 0.998 0.9985 0.999 0.9995 1 1.0005];

% probability-one bound (EZ1989/MM2010) parameters
param.EZ.percentile = 0.95;
param.aux.EZ_contour_levels = [0.999 1 1.001 1.002 1.003 1.004 1.005 1.006];

% graph grid content - figure 3
data.labeli = 'PSI';
data.labelj = 'BETA';
param.aux.ni = 15;
param.aux.nj = 15;
data.veci = linspace(1.25,3.5,param.aux.ni);
data.vecj = linspace(0.997,1,param.aux.nj);

% solution routine and output
param.aux.simulation_routine = str2func('compute_RC_SSY_model');
param.aux.filename = 'data_SSY_model_figure_2';
param.aux.modelname = 'Schorfheide, Song and Yaron';
param.aux.loadpreviousvalues = 0;

% disp('Computing Figure 3');
% compute_figure(data,param);

fprintf('Finished in %f seconds.\n',toc);
