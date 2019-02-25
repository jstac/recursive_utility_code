#=

Epstein-Zin utility specification and the Schorfheide, Song and Yaron model
consumption process. 
    
Log consumption growth g_c is given by

    g_c = μ_c + z + σ_c η'

    z' = ρ z + sqrt(1 - ρ^2) σ_z e'

    σ_z = ϕ_z σ_bar exp(h_z)

    σ_c = ϕ_c σ_bar exp(h_c)

    h_z' = ρ_hz h_z + σ_hz u'

    h_c' = ρ_hc h_c + σ_hc w'

Here {e}, {u} and {w} are IID and N(0, 1).  


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
EpsteinZin constructor for SSY model.  See p. 28 of the May 2017 version
of Schorfheide, Song and Yaron.

"""
function EpsteinZinSSY(; ψ=1.97, γ=8.89, β=0.999, ζ=1.0)
    θ = (1 - γ) / (1 - 1/ψ)
    return EpsteinZin(ψ, γ, β, ζ, θ)
end




"""
Consumption process parameters of SSY model


"""
struct SSYConsumption{T <: Real}  
    μ_c::T 
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
Default consumption process values from May 2017 version of Schorfheide, Song
and Yaron.  See p. 28.

"""
function SSYConsumption(;μ_c=0.0016,
                         ρ=0.987,
                         ϕ_z=0.215,
                         σ_bar=0.0032,
                         ϕ_c=1.0,
                         ρ_hz=0.992,
                         σ_hz=sqrt(0.0039),
                         ρ_hc=0.991,
                         σ_hc=sqrt(0.0096))  
                       
    return SSYConsumption(μ_c,
                          ρ,
                          ϕ_z,
                          σ_bar,
                          ϕ_c,
                          ρ_hz,
                          σ_hz,
                          ρ_hc,
                          σ_hc)
end


