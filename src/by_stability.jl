#=

Compute the stability test values for the BY model.

=#


include("by_model.jl")


"""
Compute the spectral radius by iteration.

"""
function compute_spec_rad_coeff(ez::EpsteinZin,
                                by::BYconsumption;
                                tol=1e-6, 
                                max_iter=5000) 

    bcm = BYComputableModel(ez, by)

    error = tol + 1
    r = 1
    i = 1

    while error > tol && i < max_iter
        s = K_interp!(bcm)
        new_r = s^(1/i)
        error = abs(new_r - r)
        i += 1
        r = new_r
    end

    return r^bcm.θ
end



"""
Compute Marinacci and Montrucchio's stability coefficient 

    exp(b) β^(ψ / (1 - ψ))

for the BY model.
"""
function compute_mm_coef(ez::EpsteinZin, by::BYconsumption)
                            
    z_min, z_max, σ_min, σ_max, c_max = compute_bounds(by)
    β, ψ = ez.β, ez.ψ

    return exp(c_max) * β^(ψ / (ψ - 1))

end
