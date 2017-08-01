#=

Code for the Bansal-Yaron and BKY growth paths.  The growth process for
consumption is

    ln (C' / C) = μ + z - σ η'

where X = (z, σ) is generated via a SV process and {η} is iid N(0, 1).

The default value for μ is 0.0015.

=#

include("stochastic_volatility.jl")



# SV specifications correspoinding to BY and BKY

function StochasticVolatilityBY(; ρ=0.979, 
                                  s_z=0.044,     # ϕ_x in PSW
                                  v=0.987,       # ν (nu) in PSW, vee here
                                  σ_bar=0.0078,  # same in PSW
                                  s_σ=2.3e-6)    # ϕ_σ in PSW

    d = σ_bar^2 * (1 - v)
    return StochasticVolatility(ρ, s_z, v, d, s_σ)
end


function StochasticVolatilityBKY(; ρ=0.975, 
                                   s_z=0.038,     # ϕ_x in PSW
                                   v=0.999,       # ν (nu) in PSW, vee here
                                   σ_bar=0.0072,  # same in PSW
                                   s_σ=2.8e-6)    # ϕ_σ in PSW
    d = σ_bar^2 * (1 - v)
    return StochasticVolatility(ρ, s_z, v, d, s_σ)
end


function compute_K_bansal_yaron(ez::EpsteinZin, 
                                sv::StochasticVolatility;
                                μ=0.0015, 
                                I=10,   # discretization in σ
                                J=10)   # discretization in z for each σ

    # Unpack parameters, allocate memory
    ψ, γ, β, ζ, θ = ez.ψ, ez.γ, ez.β, ez.ζ, ez.θ
    M = I * J
    K = Array{Float64}(M, M)

    # Discretize SV process 
    x, Q = discretize_sv(sv, I, J)

    for m in 1:M
        for mp in 1:M
            i, j = multi_from_single(m, J)
            z, σ = x[1, m], x[2, m] 
            a = exp((1 - γ) * (μ + z) + (1 - γ)^2 * σ^2 / 2)
            K[m, mp] =  a * Q[m, mp]
        end
    end
    
    return β^θ * K

end



"""
Iterate to convergence on the Koopmans operator. Unpack the final result into
an IxJ matrix X of (z_{ij}, σ_i) pairs, and a matrix of corresponding values 
W = w_{ij}

"""
function compute_fp_by(ez::EpsteinZin, 
                       sv::StochasticVolatility;
                       μ=0.0015, 
                       I=10,  
                       J=10, 
                       L=20,
                       tol=1e-5, 
                       max_iter=8000)

    # Unpack and set up parameters EpsteinZin parameters
    ψ, γ, β, ζ, θ = ez.ψ, ez.γ, ez.β, ez.ζ, ez.θ
    theta_inv, psi_inv = 1 / θ, 1 / ψ

    # Obtain the states associated with the SV process
    x, Q, Z, σ_vals = discretize_sv(sv, I, J, verbose=true)

    K = compute_K_bansal_yaron(ez, sv, μ=μ, I=I, J=J)
    M = I * J
    w = ones(M)
    iter = 0
    error = tol + 1

    while error > tol && iter < max_iter
        Tw = ζ^(1 - psi_inv) .+ (K * (w.^θ)).^theta_inv
        error = maximum(abs, w .- Tw)
        w = Tw
        iter += 1
    end

    println("Iteration converged after $iter iterations") 

    # To plot contours or in 3d we need to evaluate the fixed point at a
    # consistent set of z values.

    W = Matrix{Float64}(I, J)
    for m in 1:M
        i, j = multi_from_single(m, J)
        W[i, j] = w[m]
    end

    # Make a grid of z values that runs from smallest observed to largest
    z_max = maximum(Z)
    z_min = minimum(Z)
    z_vals = linspace(z_min, z_max, L)

    # Prepare final W matrix
    W_plot = Array{Float64}(I, L)

    for (i, σ) in enumerate(σ_vals)
        for (j, z) in enumerate(z_vals)
            # Interpolate from z values in i-th row 
            wf = LinInterp(Z[i,:], W[i,:])
            W_plot[i, :] = wf.(z_vals)
        end
    end

    return σ_vals, z_vals, W_plot.^(1 / (1 - 1 / ψ))
end


