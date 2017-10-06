
include("ez_model.jl")
include("by_consumption.jl")
include("ssy_consumption.jl")



"""
Compute K for the BY model.

"""
function compute_K(ez::EpsteinZin, byd::BYconsumptionDiscretized)

    # Unpack parameters, allocate memory
    ψ, γ, β = ez.ψ, ez.γ, ez.β
    θ = (1 - γ) / (1 - 1/ψ)
    μ = byd.by.μ

    M = byd.I * byd.J
    K = Array{Float64}(M, M)

    x, Q = byd.x_states, byd.Q

    for m in 1:M
        for mp in 1:M
            z, σ = x[1, m], x[2, m] 
            a = exp((1 - γ) * (μ + z) + (1 - γ)^2 * σ^2 / 2)
            K[m, mp] =  a * Q[m, mp]
        end
    end
    
    return β^θ * K

end


"""
Compute K in the BY model, convenience function.

"""
function compute_K(ez::EpsteinZin, 
                   by::BYconsumption;
                   I=6,   
                   J=6)  

    byd = discretize(by, I, J)
    return compute_K(ez, byd)
end



"""
Compute K in the SSY model.

"""
function compute_K(ez::EpsteinZin, ssyd::SSYconsumptionDiscretized)

    # Unpack parameters, allocate memory
    ψ, γ, β = ez.ψ, ez.γ, ez.β
    θ = (1 - γ) / (1 - 1/ψ)

    μ = ssyd.ssy.μ

    K, I, J = ssyd.K, ssyd.I, ssyd.J
    σ_c_states, σ_z_states = ssyd.σ_c_states, ssyd.σ_z_states
    z_states, Q = ssyd.z_states, ssyd.Q

    M = I * J * K
    K_matrix = Array{Float64}(M, M)


    for m in 1:M
        for mp in 1:M
            k, i, j = single_to_multi(m, I, J)
            σ_c, σ_z, z = σ_c_states[k], σ_z_states[i], z_states[i, j]
            a = exp((1 - γ) * (μ + z) + (1 - γ)^2 * σ_c^2 / 2)
            K_matrix[m, mp] =  a * Q[m, mp]
        end
    end
    
    return β^θ * K_matrix
end




"""
Compute K in the SSY model, convenience function.

"""
function compute_K(ez::EpsteinZin, 
                   ssy::SSYconsumption;
                   K=6,   # discretization in σ_c
                   I=6,   # discretization in σ_z
                   J=6)   # discretization in z for each σ_z

    ssyd = discretize(ssy, K, I, J)
    return compute_K(ez, ssyd)

end


