#=

Epstein-Zin utility specification, plus default constructors for
some standard models.

=#


"""
Struct to store parameters of the Epstein-Zin model.

"""
mutable struct EpsteinZin{T <: Real}
    ψ::T   # Elasticity of intertemporal substitution
    γ::T   # Risk aversion parameter
    β::T   # Time discount factor
end


"""
EpsteinZin constructor for BY model.

"""
function EpsteinZinBY(; ψ=1.5, γ=10.0, β=0.998)
    return EpsteinZin(ψ, γ, β)
end


"""
EpsteinZin constructor for SSY model.  See p. 28 of the May 2017 version
of Schorfheide, Song and Yaron.

"""
function EpsteinZinSSY(; ψ=1.97, γ=8.89, β=0.999)
    return EpsteinZin(ψ, γ, β)
end



