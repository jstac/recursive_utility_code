ssy_model;

function ssy_prob_one_test_val = compute_ssy_prob_one_test_val(ssy)
    ssy = ssy_model
    eta_hat = 3.0
    
    beta = ssy.beta
    gamma = ssy.gamma
    psi = ssy.psi
    
    mu_c = ssy.mu_c
    rho = ssy.rho
    phi_z = ssy.phi_z
    sigma_bar = ssy.sigma_bar
    phi_c = ssy.phi_c
    
    rho_hz = ssy.rho_hz
    sigma_hz = ssy.sigma_hz
    rho_hc = ssy.rho_hc
    sigma_hc = ssy.sigma_hc
    
    hz_hat = sigma_hz * eta_hat/(1 - rho_hz)
    hc_hat = sigma_hc * eta_hat/(1 - rho_hc)
    
    z_hat = sqrt(1 - rho^2) * sigma_z_hat * eta_hat/(1 - rho)
    
    B_c = exp(mu_c + z_hat + sigma_c_hat * eta_hat)
    
    ssy_prob_one_test_val = beta * B_c^(1 - 1/psiss
end