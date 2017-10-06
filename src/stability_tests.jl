#=

Code for the Bansal-Yaron and SSY stability tests

=#

include("compute_kmatrix.jl")


"""
Function to compute spectral radius of a matrix.

"""
compute_spec_rad(Q::Matrix{Float64}) = maximum(abs, eigvals(Q))


"""
Compute the spectral radius for the BY model.

"""
function compute_spec_rad(ez::EpsteinZin, byd::BYconsumptionDiscretized)

    K_matrix = compute_K(ez, byd) 
    return compute_spec_rad(K_matrix)

end


"""
Compute the spectral radius for the SSY model.

"""
function compute_spec_rad(ez::EpsteinZin,
                          ssyd::SSYconsumptionDiscretized)

    K_matrix = compute_K(ez, ssyd)
    return compute_spec_rad(K_matrix)

end




"""
Compute a probability one upper bound on the growth rate of consumption for
the BY model.

"""
function compute_growth_upper_bound(byd::BYconsumptionDiscretized) 
                            
    z_upper = maximum(byd.x_states[1, :])
    σ_upper = maximum(byd.x_states[2, :])

    return byd.by.μ + z_upper + σ_upper * 1.96

end



"""
Compute a probability one upper bound on the growth rate of consumption for
the SSY model.

"""
function compute_growth_upper_bound(ssyd::SSYconsumptionDiscretized) 
                            
    z_upper = maximum(ssyd.z_states)
    σ_upper = maximum(ssyd.σ_c_states)

    return ssyd.ssy.μ + z_upper + σ_upper * 1.96

end



"""
Compute Marinacci and Montrucchio's stability coefficient 

    a_w β^(ψ / (1 - ψ))

for the BY model.
"""
function compute_mm_coef(ez::EpsteinZin, byd::BYconsumptionDiscretized)
                            
    a = compute_growth_upper_bound(byd)
    β, ψ = ez.β, ez.ψ

    return exp(a) * β^(ψ / (ψ - 1))

end





"""
Compute Marinacci and Montrucchio's stability coefficient 

    a_w β^(ψ / (1 - ψ))

for the SSY model.
"""
function compute_mm_coef(ez::EpsteinZin, ssyd::SSYconsumptionDiscretized)
                            
    a = compute_growth_upper_bound(ssyd)
    β, ψ = ez.β, ez.ψ

    return exp(a) * β^(ψ / (ψ - 1))

end





