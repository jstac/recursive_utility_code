#=

Code for the linear state space (Tallarini) growth path.  The growth process
for consumption is

    ln (C' / C) = \mu + X' - X

and {X} is a finite state Markov chain.

=#
using QuantEcon

include("ez_model.jl")
include("utilities.jl")


"""
Compute the K matrix corresponding to the linear time trend.

"""
function compute_K_ltt(ez::EpsteinZin, 
                       mc::MarkovChain, 
                       μ::AbstractFloat)
    x = mc.state_values
    c = 1 - ez.γ
    M = length(x)
    K = Array{Float64}(M, M)
    for i in 1:M
        for j in 1:M
            K[i, j] = exp(c * (μ + x[j] - x[i])) * mc.p[i, j]
        end
    end
    return ez.β^ez.θ * K
end


function compute_spec_rad_ltt(;ψ=1.5,
                               γ=2.5,
                               β=0.99,
                               ρ=0.91,
                               b=0.0,
                               σ=0.0343,
                               μ=0.02,
                               M=10)
    ez = EpsteinZin(ψ, γ, β)
    ar1 = AR1(ρ=ρ, b=b, σ=σ)
    mc = ar1_to_mc(ar1, M)
    K = compute_K_ltt(ez, mc, μ)
    return compute_spec_rad(K), ez.θ
end


