#=

Compute the fixed point of T by iteration.

=#


include("stability_tests.jl")


"""
Iterate to convergence on the Koopmans operator assocaited with the BY model.
Unpack the result into an IxJ matrix X of (z_{ij}, σ_i) pairs, and a matrix of
corresponding values W = w_{ij}

"""
function compute_fp(ez::EpsteinZin, 
                    by::BYconsumption;
                    I=10,  
                    J=10, 
                    L=20,
                    tol=1e-5, 
                    max_iter=8000)

    # Unpack and set up parameters EpsteinZin parameters
    ψ, γ, β = ez.ψ, ez.γ, ez.β
    theta_inv, psi_inv = 1 / θ, 1 / ψ

    # Obtain the states associated with the SV process
    x, Q, Z, σ_vals = discretize_by(by, I, J)

    K = compute_K_by(ez, by, I=I, J=J)
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



"""
Iterate to convergence on the Koopmans operator. Unpack the final result into
(σ_c[k], σ_z[i], z[i,j]) values and an array of corresponding values w[m],
where

    m = (k-1) * (I * J) + (i - 1) * J + j

"""
function compute_fp_ssy(ez::EpsteinZin, 
                        ssy::SSYconsumption;
                        K=6,  
                        I=6,  
                        J=6, 
                        tol=1e-5, 
                        max_iter=10000)

    # Unpack and set up parameters EpsteinZin parameters
    ψ, γ, β = ez.ψ, ez.γ, ez.β

    σ_c_states, σ_z_states, z_states, K_matrix = 
        compute_K_ssy(ez, ssy, K=K, I=I, J=J, return_states=true)

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

