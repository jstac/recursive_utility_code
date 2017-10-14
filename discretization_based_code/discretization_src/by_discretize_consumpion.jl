#=

We provide operations to discretize the Bansal-Yaron consumption process.  The
discretization method uses two iterations of Rouwenhorst.  The discretized
version is a representation of a Markov chain with finitely many states x =
(z, σ) and stochastic matrix giving transition probabilitites between them.
The σ process is truncated at zero.

More specifically, discretization produces a (2, M) matrix x_states, each
element x of which is a pair (z, σ) stacked vertically, and a transition
matrix Q such that 

    Q[m, mp] = probability of transition x_states[m] -> x_states[mp]

The strategy is to 

1. Discretize the σ process to produce state values σ_1, ..., σ_I

2. For each σ_i, 

    * discretize the z process to get z_{i1}, ... z_{iJ}

In each case, discretization uses Rouwenhorst's method 

The final states are constructed as 

    x_m = (z_{ij}, σ_i), where m = (i - 1) * J + j
    
Each x_m vector is stacked as a column of x_states.  The transition
probability Q[m, n] from x_m to x_n is computed from the transition matrices
arising from the discretization of σ and z discussed above.

=#

include("consumption.jl")
using QuantEcon
using StatsBase


"""
A struct to store parameters and the discretized version of the consumption
process.

"""
struct BYconsumptionDiscretized <: DiscretizedConsumptionProcess
    by::BYconsumption
    I::Int64
    J::Int64
    z_states::Array{Float64}
    σ_states::Array{Float64}
    x_states::Array{Float64}
    Q::Array{Float64}
end


"""
Two convenience functions to switch between a single index m that 
points to elements of a matrix A_{ij} and the multi-index (ij).  The
rule taking us from ij to m is that we start at row 1 and keep counting,
traversing the matrix from left to right and top to bottow.  Thus,

    m = (i - 1) * J + j 

"""
function single_to_multi(m, J) 
    i, j = div(m - 1, J) + 1, rem(m - 1, J) + 1
    return i, j
end

multi_to_single(i, j, J) = (i - 1) * J + j

"""
Generate a discretized version of the process.  Returns an instance of
BYconsumptionDiscretized.

"""
function discretize(by::BYconsumption, 
                    I::Integer, 
                    J::Integer; 
                    fail_with_neg_σ=false) 
 
    # Unpack names
    ρ, ϕ_z, v, d, ϕ_σ = by.ρ, by.ϕ_z, by.v, by.d, by.ϕ_σ

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
        i, j = single_to_multi(m, J)
        x_states[:, m] = [z_states[i, j], σ_states[i]]
        for mp in 1:M
            ip, jp = single_to_multi(mp, J)
            Q[m, mp] = sig_Q[i, ip] * q[i, j, jp]
        end
    end

    byd = BYconsumptionDiscretized(by,
                                   I, 
                                   J, 
                                   z_states, 
                                   σ_states, 
                                   x_states, 
                                   Q)
    return byd
end

"""
Convenience function.

"""
function discretize(by::BYconsumption, int_vec)
    I, J = int_vec
    return discretize(by, I, J)
end



function sim_consumption(byd::BYconsumptionDiscretized;
                         ts_length=1000)

    M = byd.I * byd.J
    mc = MarkovChain(byd.Q, [byd.x_states[:, m] for m in 1:M])

    c_growth = Vector{Float64}(ts_length)
    z_vals = Vector{Float64}(ts_length)
    σ_vals = Vector{Float64}(ts_length)

    ts = simulate(mc, ts_length)
    for t in 1:ts_length
        z, σ = ts[t]
        z_vals[t] = z
        σ_vals[t] = σ
        c_growth[t] = byd.by.μ + z + σ * randn()
    end
    
    return σ_vals, z_vals, c_growth
end


