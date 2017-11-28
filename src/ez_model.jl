#=

Epstein-Zin utility specification, plus default constructors for
some standard models.

Author: @jstac

=#


"""
Struct to store parameters of the Epstein-Zin model.

"""
struct EpsteinZin{T <: Real}
    ψ::T   # Elasticity of intertemporal substitution
    γ::T   # Risk aversion parameter
    β::T   # Time discount factor
    ζ::T   # Typically 1 - β 
    θ::T   # Derived parameter (1 - γ) / (1 - 1/ψ)
end


"""
EpsteinZin constructor for Bansal-Yaron model.

"""
function EpsteinZinBY(; ψ=1.5, γ=10.0, β=0.998)
    θ = (1 - γ) / (1 - 1/ψ)
    ζ = 1 - β
    return EpsteinZin(ψ, γ, β, ζ, θ)
end


"""
EpsteinZin constructor for SSY model.  See p. 28 of the May 2017 version
of Schorfheide, Song and Yaron.

"""
function EpsteinZinSSY(; ψ=1.97, γ=8.89, β=0.999, ζ=1-β)
    θ = (1 - γ) / (1 - 1/ψ)
    return EpsteinZin(ψ, γ, β, ζ, θ)
end


