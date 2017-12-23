% Recreates the plots in the paper for the Bansal and Yaron (2004) model.
% Figures 1a, 1b

% Parameters set to produce quick results. For higher quality, refine
% parameters param.aux.M, param.aux.T,
% param.aux.Tdrop, param.aux.ni, param.aux.nj

tic;

% benchmark model parameterization
param.GAMMA = 10;
param.BETA = 0.998;
param.PSI = 1.5;
param.MUC = 0.0015;
param.RHO = 0.979;
param.PHIZ = 0.044;
param.V = 0.987;
param.D = 7.9092*10^(-7);
param.PHISIGMA = 2.3*10^(-6);

% auxiliary parameters - for the Monte Carlo simulation
param.aux.M = 50000;     % number of simultaneous draws
param.aux.T = 100;      % number of periods
param.aux.Tdrop = 100;  % number of burn-in periods for the Monte Carlo

% -------------------------------------------------------------------------
% FIGURE 2
% auxiliary graph grid parameters
param.aux.contour_levels = [0.9984 0.9988 0.9992 0.9996 1 1.0004];

% probability-one bound (EZ1989/MM2010) parameters
param.EZ.percentile = 0.95;
param.aux.EZ_contour_levels = [0.998 1 1.002 1.004 1.006 1.008 1.01];

% graph grid content - figure 1
data.labeli = 'MUC';
data.labelj = 'PSI';
param.aux.ni = 15;
param.aux.nj = 15;
data.veci = linspace(0.0012,0.0045,param.aux.ni);
data.vecj = linspace(1.05,4,param.aux.nj);

% solution routine and output
param.aux.simulation_routine = str2func('compute_RC_BY_model');
param.aux.filename = 'data_BY_model_figure_1';
param.aux.modelname = 'Bansal and Yaron';
param.aux.loadpreviousvalues = 0;

disp('Computing Figure 1');
compute_figure(data,param);

fprintf('Finished in %f seconds.\n',toc);

% =========================================================================

