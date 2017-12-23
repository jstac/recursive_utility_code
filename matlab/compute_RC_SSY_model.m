function output = compute_RC_SSY_model(p)
% simulate and compute RC and MC for the Schorfheide, Song, Yaron (2017) model

M = p.aux.M;
T = p.aux.T;
Tdrop = p.aux.Tdrop;

% initialization of z and sigma2 draws
z = zeros(M,1);
hc = zeros(M,1);
hz = zeros(M,1);
gn = zeros(M,1);

% etadraws = randn(M,3*(T+Tdrop));

for t = 1 : T + Tdrop
    
    eta = randn(M,4); % c, z, hc, hz
    
    sigmac = p.PHIC*p.SIGMABAR*exp(hc);
    sigmaz = p.PHIZ*p.SIGMABAR*exp(hz);

    gc = p.MUC + z + sigmac .* eta(:,1);
    
    z = p.RHO*z + (1-p.RHO^2)^0.5 * sigmaz .* eta(:,2);
    
    hc = p.RHOHC * hc + p.SIGMAHC*eta(:,3);
    hz = p.RHOHZ * hz + p.SIGMAHZ*eta(:,4);
    
    if t > Tdrop
        gn = gn + gc;
    end
end

output.RC = (mean(exp((1-p.GAMMA)*gn))^(1/T))^(1/(1-p.GAMMA));

% EZ/MM probability one criterion
gc_sorted = sort(gc,1,'ascend');
output.MC = exp(gc_sorted(round(p.EZ.percentile*M),1));

end
