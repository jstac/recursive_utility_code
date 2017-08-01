#=

Code for the linear state space (Tallarini) growth path.  The growth process
for consumption is

    ln (C' / C) = \mu + X' - X

and {X} is a finite state Markov chain.

=#

include("ez_model.jl")


"""
Compute the K matrix corresponding to the linear time trend.

"""
function compute_K_ltt(ez::EpsteinZin, 
                       mc::MarkovChain, 
                       μ::AbstractFloat)
    x = mc.state_values
    c1 = ez.β^ez.θ 
    c2 = 1 - ez.γ
    M = length(x)
    K = Array{Float64}(M, M)
    for i in 1:M
        for j in 1:M
            K[i, j] = c1 * exp(c2 * (μ + x[j] - x[i])) * mc.p[i, j]
        end
    end
    return K
end


"""
Compute the K matrix corresponding to the linear time trend 
growth process for consumption when {X} is an AR1 process (which is 
then converted to a finite state Markov chain.)

"""
function compute_K_ltt(ez::EpsteinZin, 
                       ar1::AR1, 
                       μ::AbstractFloat, 
                       M::Integer)
    mc = ar1_to_mc(ar1, M)
    return compute_K_ltt(ez, mc, μ)
end



#=

Code for the Bansal-Yaron and BKY growth paths.  The growth process for
consumption is

    ln (C' / C) = μ + z - σ η'

where X = (z, σ) is generated via a SV process and {η} is iid N(0, 1).

The default value for μ is 0.0015.

=#



"""
Two convenience functions to switch between a single index m that 
points to elements of a matrix A_{ij} and the multi-index (ij).  The
rule taking us from ij to m is that we start at row 1 and keep counting,
starting at the first row and working down.  Thus,

    m = (i - 1) * J + j 

"""
multi_from_single(m, J) = div(m - 1, J) + 1, rem(m - 1, J) + 1
single_from_multi(i, j, J) = (i - 1) * J + j


"""
Struct for parameters of the SV model

    z' = ρ z + s_z σ e'

    (σ^2)' = v σ^2 + d + s_σ w'

where {e} and {w} are IID and N(0, 1).  
"""
mutable struct StochasticVolatility{T <: AbstractFloat}
    ρ::T
    s_z::T
    v::T
    d::T
    s_σ::T
end


# Specifications correspoinding to BY and BKY

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


"""
Discretize the SV model defined above.  Returns a (2, M) matrix 
x_states, each element x of which is a pair (z, σ) stacked vertically, 
and a transition matrix Q such that 

    Q[m, mp] = probability of transition x_states[m] -> x_states[mp]

The strategy is to 

1. Discretize the σ process to produce state values σ_1, ..., σ_I

2. For each σ_i, 

    * discretize the z process to get z_{i1}, ... z_{iJ}

In each case, discretization uses Rouwenhorst's method 

The final states are constructed as 

    x_m = (z_{ij}, σ_i), where m = (i - 1) * J + j.
    
Each x_m vector is stacked as a column of x_states.  The transition
probability Q[m, n] from x_m to x_n is computed from the transition matrices
arising from the discretization of σ and z discussed above.

"""
function discretize_sv(sv::StochasticVolatility, I, J; 
                       fail_with_neg_σ=false, 
                       verbose=false) 

    # Unpack names
    ρ, s_z, v, d, s_σ = sv.ρ, sv.s_z, sv.v, sv.d, sv.s_σ

    # Discretize σ first
    mc = rouwenhorst(I, v, s_σ, d)
    sig_Q, sig2 = mc.p, collect(mc.state_values)

    # This gives σ^2 values so now we take the square root
    σ_states = similar(sig2)
    if fail_with_neg_σ == true
        @assert all(sig2 .>= 0) "Discretization failed: negative σ values."
    else
        for i in 1:I
            σ_states[i] = sig2[i] < 0 ? 1e-8 : sqrt(sig2[i])
        end
    end

    # Allocate memory
    M = I * J
    z_states = Array{Float64}(I, J)
    q = Array{Float64}(I, J, J)
    x_states = Array{Float64}(2, M)
    Q = Array{Float64}(M, M)
    
    # Discretize z at each σ_i and record state values for z in z_states.
    # Also, record transition probability from z_states[i, j] to 
    # z_states[i, jp] when σ = σ_i.  Store it as q[i, j, jp].
    for (i, σ) in enumerate(σ_states)
        mc_z = rouwenhorst(J, ρ, s_z * σ, 0.0) 
        for j in 1:J
            z_states[i, j] = mc_z.state_values[j]
            for jp in 1:J
                q[i, j, jp] = mc_z.p[j, jp]  
            end
        end
    end

    # Compute x_states and Q
    for m in 1:M
        i, j = multi_from_single(m, J)
        x_states[:, m] = [z_states[i, j], σ_states[i]]
        for mp in 1:M
            ip, jp = multi_from_single(mp, J)
            Q[m, mp] = sig_Q[i, ip] * q[i, j, jp]
        end
    end

    if verbose == true
        return x_states, Q, z_states, σ_states
    else
        return x_states, Q
    end
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


