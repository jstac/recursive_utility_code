#= 

Some utilities, including convenience functions for AR1 models

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


