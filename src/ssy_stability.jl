#=

Compute the stability test values for the SSY model.

=#


include("ssy_model.jl")


"""
Compute the spectral radius by iteration.

"""
function compute_spec_rad_coeff(ez::EpsteinZin,
                                ssy::SSYconsumption;
                                tol=1e-6, 
                                max_iter=5000) 

    scm = SSYComputableModel(ez, ssy)

    error = tol + 1
    r = 1
    i = 1

    while error > tol && i < max_iter
        s = K_interp!(scm)
        new_r = s^(1/i)
        error = abs(new_r - r)
        i += 1
        r = new_r
        copy!(scm.g, scm.g_next)
    end

    return r^scm.θ
    
end


"""
Compute Marinacci and Montrucchio's stability coefficient 

    exp(b) β^(ψ / (1 - ψ))

for the SSY model.
"""
function compute_mm_coef(ez::EpsteinZin, ssy::SSYconsumption)
                            
    out = compute_bounds(ssy)
    z_min, z_max, h_z_min, h_z_max, h_c_min, h_c_max, c_max = out
    β, ψ = ez.β, ez.ψ

    return exp(c_max) * β^(ψ / (ψ - 1))

end

