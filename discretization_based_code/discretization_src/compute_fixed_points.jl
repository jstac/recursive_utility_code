#=

Compute the fixed point of T by iteration.

=#


include("stability_tests.jl")


"""
Iterate to convergence on the Koopmans operator assocaited with the BY model.

"""
function compute_fp(ez::EpsteinZin, 
                    cpd::DiscretizedConsumptionProcess;
                    tol=1e-6, 
                    init_val=1, 
                    max_iter=20000)

    # Unpack and set up parameters EpsteinZin parameters
    ψ, γ, β = ez.ψ, ez.γ, ez.β
    θ = (1 - γ) / (1 - 1/ψ)
    ζ = 1 - β
    theta_inv, psi_inv = 1 / θ, 1 / ψ

    K_matrix = compute_K(ez, cpd)
    _, M = size(cpd.x_states)  # M is number of columns
    w = ones(M) * init_val
    iter = 0
    error = tol + 1

    r = compute_spec_rad(K_matrix)
    println("Spec rad = $r and θ = $θ")
    println("Beginning iteration\n\n")


    while error > tol && iter < max_iter
        Tw = ζ^(1 - psi_inv) .+ (K_matrix * (w.^θ)).^theta_inv
        error = maximum(abs, w .- Tw)
        w = Tw
        iter += 1
    end

    println("Iteration converged after $iter iterations") 

    return w.^(1 / (1 - 1 / ψ))
end
