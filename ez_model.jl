#=
Code for studying stability and computing solutions to the recursion that
defines lifetime utility in Epstein-Zin preference models, with nonstationary
consumption.

=#

import QuantEcon: rouwenhorst, MarkovChain, LinInterp


"""
Function to compute spectral radius of a matrix.

"""
compute_spec_rad(Q::Matrix) = maximum(abs, eigvals(Q))


#=

Epstein-Zin utility specification.

=#


"""
Struct to store parameters of Epstein-Zin model.

"""
mutable struct EpsteinZin{T <: AbstractFloat}
    ψ::T   # Elasticity of intertemporal substitution
    γ::T   # Risk aversion parameter
    β::T   # Time discount factor
    ζ::T   # Preference factor, current consumption
    θ::T   # Derived parameter
end


"""
Simple EpsteinZin constructor.

"""
function EpsteinZin(ψ, γ, β)
    ζ = 1 - β
    θ = (1 - γ) / (1 - 1/ψ)
    return EpsteinZin(ψ, γ, β, ζ, θ) 
end



"""
EpsteinZin constructor for BY model.

"""
function EpsteinZinBY(; ψ=1.5, γ=10.0, β=0.998)
    return EpsteinZin(ψ, γ, β)
end

"""
EpsteinZin constructor for BKY model.

"""
function EpsteinZinBKY(; ψ=1.5, γ=10.0, β=0.9989)
    return EpsteinZin(ψ, γ, β)
end


#= 

Convenience functions for AR1 models

=#

"""
Struct to store parameters of AR1 model

    X' = ρ X + b + σ W

"""
mutable struct AR1{T <: AbstractFloat}
    ρ::T   # Correlation coefficient
    b::T   # Intercept
    σ::T   # Volatility
end


"""
A constructor for AR1 with default values from Pohl, Schmedders and Wilms.

"""
function AR1(; ρ=0.91, b=0.0, σ=0.0343)
    return AR1(ρ, b, σ)
end


"""
Convert a Gaussian AR1 process to a Markov Chain via Rouwenhorst's method.

"""
function ar1_to_mc(ar1::AR1, M::Integer)
    return rouwenhorst(M, ar1.ρ, ar1.σ, ar1.b)
end


