#=

Code for the Bansal-Yaron and SSY stability tests

=#

include("ez_model.jl")
include("consumption.jl")



"""
Compute K for the BY model.

"""
function compute_K(ez::EpsteinZin, 
                   by::BYconsumption;
                   I=10,   # discretization in σ
                   J=10)   # discretization in z for each σ

    # Unpack parameters, allocate memory
    ψ, γ, β = ez.ψ, ez.γ, ez.β
    θ = (1 - γ) / (1 - 1/ψ)
    μ = by.μ

    M = I * J
    K = Array{Float64}(M, M)

    # Discretize SV process 
    x, Q = discretize_by(by, I, J)

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
Compute K in the SSY model.

"""
function compute_K(ez::EpsteinZin, 
                   ssy::SSYconsumption;
                   K=6,   # discretization in σ_c
                   I=6,   # discretization in σ_z
                   J=6,   # discretization in z for each σ_z
                   return_states=false)   

    # Unpack parameters, allocate memory
    ψ, γ, β = ez.ψ, ez.γ, ez.β
    θ = (1 - γ) / (1 - 1/ψ)
    μ = ssy.μ

    M = I * J * K
    K_matrix = Array{Float64}(M, M)

    # Discretize SV process 
    σ_c_states, σ_z_states, z_states, Q = discretize_ssy_process(ssy, K, I, J)

    for m in 1:M
        for mp in 1:M
            k, i, j = single_to_multi(m, I, J)
            σ_c, σ_z, z = σ_c_states[k], σ_z_states[i], z_states[i, j]
            a = exp((1 - γ) * (μ + z) + (1 - γ)^2 * σ_c^2 / 2)
            K_matrix[m, mp] =  a * Q[m, mp]
        end
    end
    
    if return_states
        return σ_c_states, σ_z_states, z_states, β^θ * K_matrix
    else
        return β^θ * K_matrix
    end

end



"""
Function to compute spectral radius of a matrix.

"""
compute_spec_rad(Q::Matrix) = maximum(abs, eigvals(Q))


"""
Compute the spectral radius for the BY model.

"""
function compute_spec_rad(ez::EpsteinZin,
                          by::BYconsumption;
                          I=10,   # discretization in σ_z
                          J=10)   # discretization in z for each σ_z

    K_matrix = compute_K(ez, by, I=I, J=J) 

    r = compute_spec_rad(K_matrix)
    return r

end

function compute_spec_rad(ez::EpsteinZin,
                          by::BYconsumption,
                          int_vec)

    I, J = int_vec
    return compute_spec_rad(ez, by, I=I, J=J)

end



"""
Compute the spectral radius for the SSY model.

"""
function compute_spec_rad(ez::EpsteinZin,
                          ssy::SSYconsumption;
                          K=6,   # discretization in σ_c
                          I=6,   # discretization in σ_z
                          J=6)   # discretization in z for each σ_z

    K_matrix = compute_K(ez, 
                         ssy,
                         K=K,
                         I=I,
                         J=J, 
                         return_states=false)                      

    r = compute_spec_rad(K_matrix)
    return r

end


function compute_spec_rad(ez::EpsteinZin,
                          ssy::SSYconsumption,
                          int_vec)

    K, I, J = int_vec
    return compute_spec_rad(ez, ssy, K=K, I=I, J=J)


end




"""
Compute a probability one upper bound on the growth rate of consumption for
the BY model.

"""
function compute_growth_upper_bound(by::BYconsumption, I, J) 
                            
    x_states, Q = discretize_by(by, I, J)

    z_upper = maximum(x_states[1, :])
    σ_upper = maximum(x_states[2, :])

    return by.μ + z_upper + σ_upper * 1.96

end



"""
Compute a probability one upper bound on the growth rate of consumption for
the SSY model.

"""
function compute_growth_upper_bound(ssy::SSYconsumption, K, I, J) 
                            
    σ_c_states, σ_z_states, z_states, Q = discretize_ssy_process(ssy, K, I, J)

    z_upper = maximum(z_states)
    σ_upper = maximum(σ_c_states)

    return ssy.μ + z_upper + σ_upper * 1.96

end



"""
Compute Marinacci and Montrucchio's stability coefficient 

    a_w β^(ψ / (1 - ψ))

"""
function compute_mm_coef(ez::EpsteinZin,
                         by::BYconsumption; 
                         I=10,
                         J=10)
                            
    a = compute_growth_upper_bound(by, I, J)

    β, ψ = ez.β, ez.ψ

    return exp(a) * β^(ψ / (ψ - 1))

end


function compute_mm_coef(ez::EpsteinZin,
                         by::BYconsumption, 
                         int_vec)

    I, J = int_vec
    return compute_mm_coef(ez, by, I=I, J=J)

end




"""
Compute Marinacci and Montrucchio's stability coefficient 

    a_w β^(ψ / (1 - ψ))

"""
function compute_mm_coef(ez::EpsteinZin,
                         ssy::SSYconsumption; 
                         K=6,
                         I=6,
                         J=6)
                            
    a = compute_growth_upper_bound(ssy::SSYconsumption, K, I, J)

    β, ψ = ez.β, ez.ψ

    return exp(a) * β^(ψ / (ψ - 1))

end


function compute_mm_coef(ez::EpsteinZin,
                         ssy::SSYconsumption, 
                         int_vec)

    K, I, J = int_vec
    return compute_mm_coef(ez, ssy, K=K,  I=I, J=J)

end




