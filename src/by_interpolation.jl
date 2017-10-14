#=

Code for solving the Bansal-Yaron model based around Monte Carlo and
interpolation.

The Bansal Yaron consumption process is

    g = μ + z + σ η

    z' = ρ z + ϕ_z σ e'

    (σ^2)' = v σ^2 + d + ϕ_σ w'

where {e} and {w} are IID and N(0, 1). 


=#

using Interpolations
using QuantEcon
using StatsBase

include("ez_model.jl")
include("consumption.jl")


"""
Struct for parameters of the BY model as described above.

"""
mutable struct BYconsumption{T <: Real}  <: ConsumptionProcess
    μ::T
    ρ::T
    ϕ_z::T
    v::T
    d::T
    ϕ_σ::T
end


"""
A constructor using parameters from the BY paper.

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
Compute upper and lower bounds for the state variables and consumption,
in order to truncate.  The function simulates and then picks high and low
quanttiles.

"""
function compute_bounds(by::BYconsumption; 
                        q=0.025,        # quantitle for state
                        p=0.05,         # quantile for consumption 
                        ts_length=1000000)

    # Set the seed to minimize variation
    srand(1234)

    ρ, ϕ_z, v, d, ϕ_σ = by.ρ, by.ϕ_z, by.v, by.d, by.ϕ_σ
    μ = by.μ

    z_vals = Vector{Float64}(ts_length)
    σ_vals = Vector{Float64}(ts_length)
    c_vals = Vector{Float64}(ts_length)

    z_vals[1] = 0
    c_vals[1] = 0
    σ_vals[1] = d / (1 - v)

    for t in 1:(ts_length-1)
        σ2 = v * σ_vals[t]^2 + d + ϕ_σ * randn()
        σ = sqrt(max(σ2, 0))
        c_vals[t+1] = μ + z_vals[t] + σ * randn()
        σ_vals[t+1] = σ
        z_vals[t+1] = ρ * z_vals[t] + ϕ_z * σ * randn()
    end

    z_max = quantile(z_vals, 1 - q)
    σ_max = quantile(σ_vals, 1 - q)

    z_min = quantile(z_vals, q)
    σ_min = quantile(σ_vals, q)

    c_max = quantile(c_vals, 1 - p)

    return z_min, z_max, σ_min, σ_max, c_max

end





""" 
Apply the operator K induced by the Bansal-Yaron model to a function g.
Populate the vector Kg and return the supremum

    || Kg || := sup_x | Kg(x) |

This function uses Monte Carlo.   Only for test purposes.
"""
function K_interp!(ez::EpsteinZin,
                  by::BYconsumption,
                  g_in::Matrix{Float64},     # Input vector g
                  g_out::Matrix{Float64},    # Output vector Kg
                  z_grid::AbstractVector,
                  σ_grid::AbstractVector,
                  η_vec::Vector{Float64},
                  ω_vec::Vector{Float64})

    # Unpack parameters
    β, γ, ψ = ez.β, ez.γ, ez.ψ
    θ = (1 - γ) / (1 - 1/ψ)
    ρ, v, d, ϕ_z, ϕ_σ = by.ρ, by.v, by.d, by.ϕ_z, by.ϕ_σ
    μ = by.μ

    η_shock_size = length(η_vec)
    ω_shock_size = length(ω_vec)
    n = η_shock_size * ω_shock_size

    # Interpolate w and allocate memory for new w
    g_func = interpolate((z_grid, σ_grid), g_in, Gridded(Linear()))

    # Apply the operator K to g, computing Kg and || Kg ||
    sup = -Inf
    for (i, z) in enumerate(z_grid)
        for (j, σ) in enumerate(σ_grid)
            mf = exp((1 - γ) * (μ + z) + (1 - γ)^2 * σ^2 / 2)
            s = 0.0
            for η in η_vec
                for ω in ω_vec
                    zp = ρ * z + ϕ_z * σ * η
                    σp2 = v * σ^2 + d + ϕ_σ * ω
                    σp = σp2 < 0 ? 1e-8 : sqrt(σp2)
                    s += g_func[zp, σp]
                end
            end
            result = β^θ * mf * s / n
            if result > sup
                sup = result
            end
            g_out[i, j] = result
        end
    end

    return sup
end


"""
Same thing but this function uses Gaussian quadrature.

"""
function K_interp!(ez::EpsteinZin,
                  by::BYconsumption,
                  g_in::Matrix{Float64},
                  g_out::Matrix{Float64},
                  z_grid::AbstractVector,
                  σ_grid::AbstractVector;
                  std_range::Int64=2,
                  shock_state_size::Int64=12)

    msg = "Dimention of input arrays must match grid"
    for g in (g_in, g_out)
        @assert size(g) == (length(z_grid), length(σ_grid)) msg
    end

    # Unpack parameters
    β, γ, ψ = ez.β, ez.γ, ez.ψ
    θ = (1 - γ) / (1 - 1/ψ)
    ρ, v, d, ϕ_z, ϕ_σ = by.ρ, by.v, by.d, by.ϕ_z, by.ϕ_σ
    μ = by.μ

    # Extract state and probs for N(0, 1) shocks
    mc = tauchen(shock_state_size, 0, 1, 0, std_range)
    s_vec = mc.state_values
    p_vec = mc.p[1, :]  # Any row, all columns

    # Interpolate w and allocate memory for new w
    g_func = interpolate((z_grid, σ_grid), g_in, Gridded(Linear()))

    # Apply the operator K to g, computing Kg and || Kg ||
    sup = -Inf
    for (i, z) in enumerate(z_grid)
        for (j, σ) in enumerate(σ_grid)
            mf = exp((1 - γ) * (μ + z) + (1 - γ)^2 * σ^2 / 2)
            g_exp = 0.0
            for (p, η) in zip(p_vec, s_vec)
                for (q, ω) in zip(p_vec, s_vec)
                    zp = ρ * z + ϕ_z * σ * η
                    σp2 = v * σ^2 + d + ϕ_σ * ω
                    σp = σp2 < 0 ? 1e-8 : sqrt(σp2)
                    g_exp += g_func[zp, σp] * p * q
                end
            end
            result = β^θ * mf * g_exp 
            if result > sup
                sup = result
            end
            g_out[i, j] = result
        end
    end

    return sup
end

"""
Compute the spectral radius by iteration.

"""
function compute_spec_rad_interp!(ez::EpsteinZin, 
                                 by::BYconsumption,
                                 g_in,
                                 z_grid,
                                 σ_grid;
                                 tol=1e-6, 
                                 max_iter=5000) 
    
    g_out = similar(g_in)

    error = tol + 1
    r = 1
    i = 1

    while error > tol && i < max_iter
        s = K_interp!(ez, by, g_in, g_out, z_grid, σ_grid)
        new_r = s^(1/i)
        error = abs(new_r - r)
        copy!(g_in, g_out)
        i += 1
        r = new_r
    end

    return r
end



function compute_spec_rad_interp(ez::EpsteinZin, 
                                by::BYconsumption;
                                gs_z=12, 
                                gs_σ=12, 
                                tol=1e-6, 
                                state_quantile=0.025,
                                max_iter=5000) 

    g_in = ones(gs_z, gs_σ)
    g_out = similar(g_in)

    z_min, z_max, σ_min, σ_max, c_max = compute_bounds(by, q=state_quantile)
    σ_grid = linspace(σ_min, σ_max, gs_z)
    z_grid = linspace(z_min, z_max, gs_σ)

    return compute_spec_rad_interp!(ez,
                                    by,
                                    g_in,
                                    z_grid,
                                    σ_grid,
                                    tol=tol, 
                                    max_iter=max_iter) 
end


"""
Compute Marinacci and Montrucchio's stability coefficient 

    exp(b) β^(ψ / (1 - ψ))

for the BY model.
"""
function compute_mm_coef_interp(ez::EpsteinZin, by::BYconsumption)
                            
    z_min, z_max, σ_min, σ_max, c_max = compute_bounds(by)
    β, ψ = ez.β, ez.ψ

    return exp(c_max) * β^(ψ / (ψ - 1))

end
