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

"""
EpsteinZin constructor for SSY model.

"""
function EpsteinZinSSY(; ψ=1.93, γ=8.6, β=0.999)
    return EpsteinZin(ψ, γ, β)
end



