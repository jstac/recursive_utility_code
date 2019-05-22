ssy_model;

function truncated_randn = compute_truncated_randn(truncation_val)
    y = randn
    if y > truncation_val
        truncated_randn = y;
    elseif y < -truncation_val
        truncated_randn = -truncation_value;
    else
        truncated_randn = y;
    end
end
    
function update_state = compute_update_state(x, c_params, trunc_val)
    mu_c = c_params(1)
    rho = c_params(2)
    phi_z = c_params(3)
    sigma_bar = c_params(4)
    phi_c = c_params(5)
    rho_hz = c_params(6)
    sigma_hz = c_params(7)
    rho_hc = c_params(8)
    sigma_hc = c_params(9)
    
    z = x
    h_z = x
    h_c = x
    
    sigma_z = phi_z * sigma_bar * exp(h_z)
    sigma_c = phi_c * sigma_bar * exp(h_c)
    
    z = rho * z + sqrt(1 - rho^2) * sigma_z * compute_truncated_randn(trunc_val)
    h_z = rho_hz * h_z + simga_hz * compute_truncated_randn(trunc_val)
    h_c = rho_hc * h_c + sigma_hc * compute_truncated_randn(trunc_val)
    
    update_state = [z, h_z, h_c]
end

function eval_kappa = compute_eval_kappa(x, y, c_params, trunc_val)
    mu_c = c_params(1)
    rho = c_params(2)
    phi_z = c_params(3)
    sigma_bar = c_params(4)
    phi_c = c_params(5)
    rho_hz = c_params(6)
    sigma_hz = c_params(7)
    rho_hc = c_params(8)
    sigma_hc = c_params(9)

    z = x
    h_z = x
    h_c = x
    
    simga_c = phi_c * sigma_bar * exp(h_c)
    
    eval_kappa = mu_c + z + sigma_c * randn

end

function ssy_stat_mc = ssy_function_factory(ssy)
    ssy = ssy_model
    
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
    
    function exp_log_lm = ssy_compute_stat_mc()
        initial_state = zeros(3)
        n = 1000
        m = 1000
        seed = 1234
        trunc_val = 25
        burn_in = 500
        
        rng(seed)
        
        c_params = [mu_c, rho, phi_z, sigma_bar, phi_c, rho_hz, sigma_hz, rho_hc, sigma_hc]
        
        x = initial_state
        
        for t = 0:burn_in
            x = update_state(x, c_params, trunc_val)
        end
        
        yn_vals = empty(m)
        theta = (1 - gamma)/(1 - 1/psi)
        
        for i = 0:m
            kappa_sum = 0.0
            x = initial_state
            
            for t = 0:n
                y = update_state(x, cparams, trunc_val)
                kappa_sum = kappa_sum + eval_kappa(x, y, c_params, trunc_val)
                x = y
            end
            
            yn_vals(i) = exp((1-gamma) * kappa_sum)
            
        end
        
        mean_yns = mean(yn_vals)
        log_Lm = log(beta) + (1/(n * theta)) * log(mean_yns)
        
        exp_log_lm = exp(log_Lm)
        
    end

end










