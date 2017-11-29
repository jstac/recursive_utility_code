#=

Compute the stability test values for the BY model.

=#


include("by_model.jl")

"""
Compute the spectral radius by simulation.

"""
function compute_spec_rad_coef(ez::EpsteinZin, bc::BYConsumption; 
                               z0=0.0, σ0=0.0, M=1000, N=2000)
    
    # Unpack
    θ, β, γ = ez.θ, ez.β, ez.γ
    
    sum_obs = 0.0
    
    for m in 1:M
        c, z, σ = simulate(bc, 
                           z0=z0, 
                           σ0=σ0, 
                           seed=m, 
                           ts_length=N+1);
        sum_obs +=  exp((1 - γ) * sum(c))
    end

    rK = β^θ * (sum_obs / M)^(1/N)
    return rK^θ  
end



"""
Compute Marinacci and Montrucchio's stability coefficient 

    exp(b) β^(ψ / (1 - ψ))

for the BY model.
"""
function compute_mm_coef(ez::EpsteinZin, by::BYConsumption)
                            
    z_min, z_max, σ_min, σ_max, c_max = compute_bounds(by)
    β, ψ = ez.β, ez.ψ

    return exp(c_max) * β^(ψ / (ψ - 1))

end
