#=

Compute the stability test values for the SSY model.

=#
include("ssy_model.jl")

"""
Compute the spectral radius by simulation.

"""
function compute_spec_rad_coef(ez, sc; z0=0.0, h_z0=0.0, h_c0=0.0, M=1000, N=2000)
    
    # Unpack
    θ, β, γ = ez.θ, ez.β, ez.γ
    
    prod_sum = 0.0
    
    for m in 1:M
        c, z, hz, hc = simulate(sc, 
                               z0=z0, 
                               h_z0=h_z0, 
                               h_c0=h_c0, 
                               seed=m, 
                               ts_length=N+1);
        prod_sum += prod(β^θ .* exp.( (1 - γ) .* c ))
    end

    rK = (prod_sum / M)^(1/N)
    return rK^θ
end



"""
Compute Marinacci and Montrucchio's stability coefficient 

    exp(b) β^(ψ / (1 - ψ))

for the SSY model.
"""
function compute_mm_coef(ez::EpsteinZin, sc::SSYConsumption, p=0.05) 

    # Compute the upper bound for C_t
    c_vals, z_vals, h_z_vals, h_c_vals = simulate(sc, seed=1234)
    c_max = quantile(c_vals, 1 - p)

    β, ψ = ez.β, ez.ψ

    return exp(c_max) * β^(ψ / (ψ - 1))

end

