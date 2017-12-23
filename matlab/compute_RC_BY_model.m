function output = compute_RC_BY_model(p)
% simulate and compute RC and MC for the Bansal, Yaron (2004) model

M = p.aux.M;
T = p.aux.T;
Tdrop = p.aux.Tdrop;

% initialization of z and sigma2 draws
z = zeros(M,1);
sigma2 = zeros(M,1) + p.D/(1-p.V);
gn = zeros(M,1);

% simulate consumption growth
for t = 1 : T + Tdrop
    
    eta = randn(M,3); %c, z, sigma
%     eta = etadraws(:,[(t-1)*3+1:t*3]);
    gc = p.MUC + z + (sigma2.^0.5) .* eta(:,1);
    
    z = p.RHO*z + p.PHIZ*(sigma2.^0.5).*eta(:,2);
    sigma2 = max(p.V*sigma2 + p.D + p.PHISIGMA*eta(:,3),0);
    
    if t > Tdrop
        gn = gn + gc;
    end
end

output.RC = (mean(exp((1-p.GAMMA)*gn))^(1/T))^(1/(1-p.GAMMA));

% EZ/MM probability one criterion
gc_sorted = sort(gc,1,'ascend');
output.MC = exp(gc_sorted(round(p.EZ.percentile*M),1));

end
