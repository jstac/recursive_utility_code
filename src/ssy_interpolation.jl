#=

Code for solving the Schorfheide, Song and Yaron mode, where
log consumption growth g is given by

    g = μ + z + σ_c η'

    z' = ρ z + sqrt(1 - ρ^2) σ_z e'

    σ_z = ϕ_z σ_bar exp(h_z)

    σ_c = ϕ_c σ_bar exp(h_c)

    h_z' = ρ_hz h_z + σ_hz u'

    h_c' = ρ_hc h_c + σ_hc w'

Here {e}, {u} and {w} are IID and N(0, 1).  

=#

include("ez_model.jl")
include("consumption.jl")

using QuantEcon
using StatsBase
using Interpolations



"""
Struct for parameters of SSY model

"""
mutable struct SSYconsumption{T <: Real}  <: ConsumptionProcess
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
Compute upper and lower bounds for the state variables and consumption,
in order to truncate.  The function simulates and then picks high and low
quanttiles.

"""
function compute_bounds(ssy::SSYconsumption; 
                        q=0.025,        # quantitle for state
                        p=0.05,         # quantile for consumption 
                        ts_length=1000000)

    # Set the seed to minimize variation
    srand(1234)


    # Unpack
    μ, ρ, ϕ_z, σ_bar, ϕ_c = ssy.μ, ssy.ρ, ssy.ϕ_z, ssy.σ_bar, ssy.ϕ_c
    ρ_hz, σ_hz, ρ_hc, σ_hc = ssy.ρ_hz, ssy.σ_hz, ssy.ρ_hc, ssy.σ_hc

    tz(h_z) = ϕ_z * σ_bar * exp(h_z)
    tc(h_c) = ϕ_c * σ_bar * exp(h_c)

    # Allocate memory with initial conditions at zero
    z_vals = zeros(ts_length)
    c_vals = zeros(ts_length)
    h_z_vals = zeros(ts_length)
    h_c_vals = zeros(ts_length)

    for t in 1:(ts_length-1)
        h_z, h_c = h_z_vals[t], h_c_vals[t]
        σ_z = tz(h_z)
        σ_c = tc(h_c)
        c_vals[t+1] = μ + z_vals[t] + σ_c * randn()
        z_vals[t+1] = ρ * z_vals[t] + sqrt(1 - ρ^2) * σ_z * randn()
        h_z_vals[t+1] = ρ_hz * h_z + σ_hz * randn()
        h_c_vals[t+1] = ρ_hc * h_c + σ_hc * randn()
    end

    c_max = quantile(c_vals, 1 - p)
    z_max = quantile(z_vals, 1 - q)
    h_z_max = quantile(h_z_vals, 1 - q)
    h_c_max = quantile(h_c_vals, 1 - q)

    z_min = quantile(z_vals, q)
    h_z_min = quantile(h_z_vals, q)
    h_c_min = quantile(h_c_vals, q)

    return z_min, z_max, h_z_min, h_z_max, h_c_min, h_c_max, c_max

end


"""
Apply the operator K induced by the SSY model to a function g.
Populate the vector Kg and return the supremum

    || Kg || := sup_x | Kg(x) |

Uses Gaussian quadrature for numerical integration.

The order for arguments of g is g(z, h_z, h_c)

"""
function K_interp!(ez::EpsteinZin,
                   ssy::SSYconsumption,
                   g_in::Array{Float64},   
                   g_out::Array{Float64},
                   z_grid::AbstractVector,
                   h_z_grid::AbstractVector,
                   h_c_grid::AbstractVector;
                   std_range::Int64=2,
                   shock_state_size::Int64=8)

    msg = "Dimention of input arrays must match grid"
    gs_z, gs_h_z, gs_h_c = length(z_grid), length(h_z_grid), length(h_c_grid)
    for g in (g_in, g_out)
        @assert size(g) == (gs_z, gs_h_z, gs_h_c) msg
    end

    # Unpack parameters
    β, γ, ψ = ez.β, ez.γ, ez.ψ
    μ, ρ, ϕ_z, σ_bar, ϕ_c = ssy.μ, ssy.ρ, ssy.ϕ_z, ssy.σ_bar, ssy.ϕ_c
    ρ_hz, σ_hz, ρ_hc, σ_hc = ssy.ρ_hz, ssy.σ_hz, ssy.ρ_hc, ssy.σ_hc

    # Some useful constants
    θ = (1 - γ) / (1 - 1/ψ)
    δ = sqrt(1 - ρ^2)
    τ_z = ϕ_z * σ_bar
    τ_c = ϕ_c * σ_bar

    # Extract state and probs for N(0, 1) shocks
    mc = tauchen(shock_state_size, 0, 1, 0, std_range)
    s_vec = mc.state_values
    p_vec = mc.p[1, :]  # Any row, all columns

    # Interpolate g and allocate memory for new g
    g_func = interpolate((z_grid, h_z_grid, h_c_grid), g_in, Gridded(Linear()))

    # Apply the operator K to g, computing Kg and || Kg ||
    sup = -Inf
    for (i, z) in enumerate(z_grid)
        for (k, h_c) in enumerate(h_c_grid)
            σ_c = τ_c * exp(h_c)
            mf = exp((1 - γ) * (μ + z) + (1 - γ)^2 * σ_c^2 / 2)
            for (j, h_z) in enumerate(h_z_grid)
                g_exp = 0.0
                for (p, η) in zip(p_vec, s_vec)
                    for (q, ω) in zip(p_vec, s_vec)
                        for (r, ϵ) in zip(p_vec, s_vec)
                            # Update state variables given shocks
                            σ_z = τ_z * exp(h_z)
                            zp = ρ * z + δ * σ_z * η
                            h_cp = ρ_hc * h_c + σ_hc * ω
                            h_zp = ρ_hz * h_z + σ_hz * ϵ
                            g_exp += g_func[zp, h_zp, h_cp] * p * q * r
                        end
                    end
                end
                result = β^θ * mf * g_exp 
                if result > sup
                    sup = result
                end
                g_out[i, j, k] = result
            end
        end
    end

    return sup
end



"""
Compute the spectral radius by iteration.

"""
function compute_spec_rad_interp!(ez::EpsteinZin, 
                                 ssy::SSYconsumption,
                                 g_in,
                                 z_grid,
                                 h_z_grid,
                                 h_c_grid;
                                 tol=1e-6, 
                                 max_iter=5000) 
    
    g_out = similar(g_in)

    error = tol + 1
    r = 1
    i = 1

    while error > tol && i < max_iter
        s = K_interp!(ez, ssy, g_in, g_out, z_grid, h_z_grid, h_c_grid)
        new_r = s^(1/i)
        error = abs(new_r - r)
        copy!(g_in, g_out)
        i += 1
        r = new_r
    end

    return r
end

function compute_spec_rad_interp(ez::EpsteinZin, 
                                ssy::SSYconsumption;
                                gs_z=8, 
                                gs_h_z=4, 
                                gs_h_c=4, 
                                tol=1e-6, 
                                max_iter=5000) 

    g_in = ones(gs_z, gs_h_z, gs_h_c)
    g_out = similar(g_in)

    z_min, z_max, h_z_min, h_z_max, h_c_min, h_c_max, c_max = compute_bounds(ssy)

    z_grid = linspace(z_min, z_max, gs_z)
    h_z_grid = linspace(h_z_min, h_z_max, gs_h_z)
    h_c_grid = linspace(h_c_min, h_c_max, gs_h_c)

    return compute_spec_rad_interp!(ez,
                                    ssy,
                                    g_in,
                                    z_grid,
                                    h_z_grid,
                                    h_c_grid,
                                    tol=tol, 
                                    max_iter=max_iter) 
end


"""
Compute Marinacci and Montrucchio's stability coefficient 

    exp(b) β^(ψ / (1 - ψ))

for the SSY model.
"""
function compute_mm_coef_interp(ez::EpsteinZin, ssy::SSYconsumption)
                            
    z_min, z_max, h_z_min, h_z_max, h_c_min, h_c_max, c_max = compute_bounds(ssy)
    β, ψ = ez.β, ez.ψ

    return exp(c_max) * β^(ψ / (ψ - 1))

end
