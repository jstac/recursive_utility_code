#=

Assorted consumption processes.

=#


import QuantEcon: rouwenhorst, MarkovChain


abstract type 
    ConsumptionProcess 
end

#=

Discretize the Bansal Yaron consumption process

    g = μ + z + σ η

    z' = ρ z + ϕ_z σ e'

    (σ^2)' = v σ^2 + d + ϕ_σ w'

We assume that {e} and {w} are IID and N(0, 1).  The discretization method
uses two iterations of Rouwenhorst.  The discretized version is a
representation of a Markov chain with finitely many states x = (z, σ) and
stochastic matrix giving transition probabilitites between them.

=#


"""
Struct for parameters of the SV model as described above.

"""
mutable struct BYconsumption{T <: Real} <: ConsumptionProcess
    μ::T
    ρ::T
    ϕ_z::T
    v::T
    d::T
    ϕ_σ::T
end


"""
Parameters for the BY paper.

"""
function BYconsumption()
    return BYconsumption(0.0015,     # μ
                         0.979,      # ρ
                         0.044,      # ϕ_z 
                         0.987,      # v
                         7.9092e-7,  # d 
                         2.3e-6)     # ϕ_σ
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
function discretize_by(by::BYconsumption, 
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
        i, j = multi_from_single(m, J)
        x_states[:, m] = [z_states[i, j], σ_states[i]]
        for mp in 1:M
            ip, jp = multi_from_single(mp, J)
            Q[m, mp] = sig_Q[i, ip] * q[i, j, jp]
        end
    end

    return x_states, Q
end



function sim_consumption(by::BYconsumption; 
                         I=10,
                         J=10,
                         μ=0.0015, 
                         ts_length=1000)

    x, Q = discretize_by(by, I, J)
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






#=

Discretize the model consumption process in Schorfheide, Song and Yaron, where
log consumption growth g is given by

    g = μ + z + σ_c η'

    z' = ρ z + sqrt(1 - ρ^2) σ_z e'

    σ_z = ϕ_z σ_bar exp(h_z)

    σ_c = ϕ_c σ_bar exp(h_c)

    h_z' = ρ_hz h_z + σ_hz u'

    h_c' = ρ_hc h_c + σ_hc w'

Here {e}, {u} and {w} are IID and N(0, 1).  

The discretization method uses iterations of Rouwenhorst.  The indices are

    σ_c[k] for k in 1:K
    σ_z[i] for i in 1:I
    z[i, :] is all z states when σ_z = σ_z[i]

The discretized version is a representation of a Markov chain with finitely
many states 

    x = (σ_c, σ_z, z)

and stochastic matrix Q giving transition probabilitites between them.

The set of states x_states is computed as a 3 x M matrix with each column
corresponding to values of (σ_c, σ_z, z)'

Discretize the SSY state process builds the discretized state values for the
    states (σ_c, σ_z, z) and a transition matrix Q such that 

    Q[m, mp] = probability of transition x[m] -> x[mp]

where

    x[m] := (σ_c[k], σ_z[i], z[i,j])
    
The rule for the index is

    m = (k-1) * (I * J) + (i - 1) * J + j



=#



"""
Struct for parameters of SSY model

"""
mutable struct SSYconsumption{T <: Real} <: ConsumptionProcess
    μ::T 
    ρ::T 
    ϕ_z::T 
    σ_bar::T
    ϕ_c::T
    ρ_hz::T
    σ_hz::T
    ρ_hc::T 
    σ_hc::T 
end


"""
Default values from May 2017 version of Schorfheide, Song and Yaron. 
See p. 28.

"""

function SSYconsumption()

    return SSYconsumption(0.0016,              # μ
               0.987,               # ρ
               0.215,               # ϕ_z
               0.0032,              # σ_bar
               1.0,                 # ϕ_c
               0.992,               # ρ_hz
               sqrt(0.0039),        # σ_hz
               0.991,               # ρ_hc
               sqrt(0.0096))        # σ_hc
end



"""
A utility function for the multi-index.
"""
split_index(i, M) = div(i - 1, M) + 1, rem(i - 1, M) + 1


function single_to_multi(m, I, J)
    k, temp = split_index(m, I * J)
    i, j = split_index(temp, J)
    return k, i, j
end

multi_to_single(k, i, j, I, J) = (k-1) * (I * J) + (i - 1) * J + j


"""
And here's the actual discretization process.  

"""
function discretize_ssy_process(ssy::SSYconsumption, 
                       K::Integer, 
                       I::Integer, 
                       J::Integer) 

    # Unpack
    ρ, ϕ_z, σ_bar, ϕ_c, ρ_hz, σ_hz, ρ_hc, σ_hc = 
        ssy.ρ, ssy.ϕ_z, ssy.σ_bar, ssy.ϕ_c, ssy.ρ_hz, ssy.σ_hz, ssy.ρ_hc, ssy.σ_hc

    ## Discretize by applying rouwenhorst to h_c and h_z
    hc_mc = rouwenhorst(K, ρ_hc, σ_hc)
    hz_mc = rouwenhorst(I, ρ_hz, σ_hz)

    σ_c_states  = ϕ_c * σ_bar * exp.(collect(hc_mc.state_values))
    σ_z_states  = ϕ_z * σ_bar * exp.(collect(hz_mc.state_values))

    # Allocate memory
    M = I * J * K
    z_states = Array{Float64}(I, J)
    q = Array{Float64}(I, J, J)

    Q = Array{Float64}(M, M)
    
    # Discretize z at each σ_z[i] and record state values for z in z_states.
    # Also, record transition probability from z_states[i, j] to 
    # z_states[i, jp] when σ_z = σ_z[i].  Store it as q[i, j, jp].
    for (i, σ_z) in enumerate(σ_z_states)
        mc_z = rouwenhorst(J, ρ, sqrt(1 - ρ^2) * σ_z, 0.0) 
        for j in 1:J
            z_states[i, j] = mc_z.state_values[j]
            for jp in 1:J
                q[i, j, jp] = mc_z.p[j, jp]  
            end
        end
    end

    # Compute Q
    for m in 1:M
        k, i, j = single_to_multi(m, I, J)
        for mp in 1:M
            kp, ip, jp = single_to_multi(mp, I, J)
            Q[m, mp] = hc_mc.p[k, kp] * hz_mc.p[i, ip] * q[i, j, jp]
        end
    end

    return σ_c_states, σ_z_states, z_states, Q
end




function sim_consumption(ssy::SSYconsumption;
                         K=6,
                         I=6,
                         J=6,
                         ts_length=1000)

    σ_c_states, σ_z_states, z_states, Q = discretize_ssy_process(ssy, K, I, J)
    M = I * J * K

    x_states = []
    for m in 1:M
        k, i, j = single_to_multi(m, I, J)
        push!(x_states, (σ_c_states[k], σ_z_states[i], z_states[i, j]))
    end

    mc = MarkovChain(Q, x_states)

    c_growth = Vector{Float64}(ts_length)
    z_vals = Vector{Float64}(ts_length)
    σ_c_vals = Vector{Float64}(ts_length)
    σ_z_vals = Vector{Float64}(ts_length)

    ts = simulate(mc, ts_length)
    for t in 1:ts_length
        σ_c, σ_z, z = ts[t]
        σ_c_vals[t] = σ_c
        σ_z_vals[t] = σ_z
        z_vals[t] = z
        c_growth[t] = ssy.μ + z + σ_c * randn()
    end
    
    return σ_c_vals, σ_z_vals, z_vals, c_growth
end


