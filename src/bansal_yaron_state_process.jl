#=

Discretize the stochastic volatility model 

    z' = ρ z + ϕ_z σ e'

    (σ^2)' = v σ^2 + d + ϕ_σ w'

We assume that {e} and {w} are IID and N(0, 1).  The discretization method
uses two iterations of Rouwenhorst.  The discretized version is a
representation of a Markov chain with finitely many states x = (z, σ) and
stochastic matrix giving transition probabilitites between them.

=#

import QuantEcon: rouwenhorst, MarkovChain



"""
Struct for parameters of the SV model as described in the header.

"""
mutable struct StochasticVolatility{T <: AbstractFloat}
    ρ::T
    ϕ_z::T
    v::T
    d::T
    ϕ_σ::T
end


"""
Parameters for the BY paper.

"""
function StochasticVolatilityBY()
    return StochasticVolatility(0.979, 0.044, 0.987, 7.9092e-7, 2.3e-6)
end

"""
Parameters for the BKY paper.

"""
function StochasticVolatilityBKY()
    return StochasticVolatility(0.975, 0.038, 0.999, 5.184e-8, 2.8e-6)
end




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
function discretize_sv(sv::StochasticVolatility, 
                       I::Integer, 
                       J::Integer; 
                       fail_with_neg_σ=false) 

    # Unpack names
    ρ, ϕ_z, v, d, ϕ_σ = sv.ρ, sv.ϕ_z, sv.v, sv.d, sv.ϕ_σ

    # Discretize σ first
    mc = rouwenhorst(I, v, ϕ_σ, d)
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
        mc_z = rouwenhorst(J, ρ, ϕ_z * σ, 0.0) 
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

    return x_states, Q
end



function sim_consumption(sv::StochasticVolatility; 
                            I=10,
                            J=10,
                            μ=0.0015, 
                            ts_length=1000)

    x, Q = discretize_sv(sv, I, J)
    M = I * J

    mc = MarkovChain(Q, [x[:, m] for m in 1:M])

    c_growth = Vector{Float64}(ts_length)
    z_vals = Vector{Float64}(ts_length)
    σ_vals = Vector{Float64}(ts_length)

    ts = simulate(mc, ts_length)
    for t in 1:ts_length
        z, σ = ts[t]
        z_vals[t] = z
        σ_vals[t] = σ
        c_growth[t] = μ + z + σ * randn()
    end
    
    return σ_vals, z_vals, c_growth
end
