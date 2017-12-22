#=

Compute the stability test values for the BY model.

=#


using StatsBase

include("ez_parameters.jl")
include("consumption_models.jl")



"""
Compute the spectral radius by simulation.

"""
function compute_spec_rad_coef(ez::EpsteinZin, 
                               cp::ConsumptionProcess; 
                               M=1000, N=2000)
    
    # Unpack
    θ, β, γ = ez.θ, ez.β, ez.γ
    
    sum_obs = 0.0
    
    for m in 1:M
        c_growth = simulate(cp, seed=m, ts_length=N+1)
        sum_obs +=  exp((1 - γ) * sum(c_growth))
    end

    rK = β^θ * (sum_obs / M)^(1/N)
    return rK^(1/θ)
end



"""
Compute EZ's stability coefficient for the BY model.
"""
function compute_ez_coef(ez::EpsteinZin, 
                         cp::ConsumptionProcess;
                         q=0.95)
                            
    c_growth = simulate(cp)

    c_max = quantile(c_growth, q)
    β, ψ = ez.β, ez.ψ

    return exp(c_max)^(1 - (1/ψ)) * β

end
