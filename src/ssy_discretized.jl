#=

Code for SSY stability and dynamics.

=#

include("utilities.jl")
include("ez_model.jl")
include("ssy_state_process.jl")


function compute_K_ssy(ez::EpsteinZin, 
                       ssy::SSY;
                       μ=0.0016, 
                       K=6,   # discretization in σ_c
                       I=6,   # discretization in σ_z
                       J=6,
                       return_states=false)   # discretization in z for each σ_z

    # Unpack parameters, allocate memory
    ψ, γ, β, ζ, θ = ez.ψ, ez.γ, ez.β, ez.ζ, ez.θ
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
Iterate to convergence on the Koopmans operator. Unpack the final result into
(σ_c[k], σ_z[i], z[i,j]) values and an array of corresponding values w[m],
where

    m = (k-1) * (I * J) + (i - 1) * J + j

"""
function compute_fp_ssy(ez::EpsteinZin, 
                        ssy::SSY;
                        μ=0.0016, 
                        K=6,  
                        I=6,  
                        J=6, 
                        tol=1e-5, 
                        max_iter=10000)

    # Unpack and set up parameters EpsteinZin parameters
    ψ, γ, β, ζ, θ = ez.ψ, ez.γ, ez.β, ez.ζ, ez.θ

    σ_c_states, σ_z_states, z_states, K_matrix = 
        compute_K_ssy(ez, ssy, μ=μ, K=K, I=I, J=J, return_states=true)

    r = compute_spec_rad(K_matrix)

    println("Spec rad = $r and θ = $θ")
    println("Beginning iteration\n\n")

    M = I * J * K
    w = ones(M)
    iter = 0
    error = tol + 1

    update(w) = ζ^(1 - (1 / ψ)) .+ (K_matrix * (w.^θ)).^(1 / θ)

    while error > tol && iter < max_iter
        Tw = update(w)
        error = maximum(abs, w .- Tw)
        iter += 1
        w = Tw
        if iter % 100 == 0
            println("iter, error = $iter, $error")
        end
    end

    println("Iteration converged after $iter iterations") 

    # Return results, reversing the transform to recover v/c
    return σ_c_states, σ_z_states, z_states, w.^(1 / (1 - (1 / ψ)) )  
end

