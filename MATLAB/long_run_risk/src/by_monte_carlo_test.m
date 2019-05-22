by_model;

function update_state = compute_update_state(x, c_params)
    mu_c = c_params(1)
    rho = c_params(2)
    phi_z = c_params(3)
    v = c_params(4)
    d = c_params(5)
    phi_sigma = c_params(6)
    
    z = x
    sigma = x
    
    sigma2 = v * sigma^2 + d + phi_sigma * randn
    sigma = sqrt(max(sigma2, 0))
    z = rho * z + phi_z * sigma * randn
    
    update_state = [z, sigma]
end

function eval_kappa = compute_eval_kappa(x, y, c_params)
    mu_c = c_params(1)
    rho = c_params(2)
    phi_z = c_params(3)
    v = c_params(4)
    d = c_params(5)
    phi_sigma = c_params(6)
    
    z = x
    sigma = x
    
    eval_kappa = mu_c + z + sigma * randn
end

function by_compute_stat_mc = by_function_factory(by)
    by = by_model

    beta = by.beta
    gamma = by.gamma
    psi = by.psi
    
    mu_c = by.mu_c
    rho = by.rho
    phi_z = by.phi_z
    v = by.v
    d = by.d
    phi_sigma = by.phi_sigma
    
    z = 0.0
    sigma = d/(1 - v)
    initial_state = [z, sigma]
    
    function Lm = by_compute_mc_stat()
        initial_state = initial_state
        n = 1000
        m = 1000
        seed = 1234
        burn_in = 500
        
        rng(seed)
        
        c_params = [mu_c, rho, phi_z, v, d, phi_sigma]
        
        x = initial_state
        for t = 0:burn_in
            x = update_state(x, c_params)
        end
        
        yn_vals = empty(m)
        theta = (1 - gamma)/(1 - 1/psi)
        
        for i = 0:m
            kappa_sum = 0.0
            
            for t = 0:n
                y = update_state(x, c_params)
                kappa_sum = kappa_sum + eval_kappa(x, y, c_params)
                x = y
            end  
            
            yn_vals(i) = exp((1 - gamma) * kappa_sum)
        end
    
        mean_yns = mean(yn_vals)
        Lm = beta * mean_yns^(1/(n * theta))
    end

end







