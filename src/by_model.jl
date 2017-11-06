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
quantiles.

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
The model constructed for computation.

"""
struct BYComputableModel{T <: Real}

    ez::EpsteinZin{T}
    by::BYconsumption{T}
    z_grid::Vector{Float64}
    σ_grid::Vector{Float64}
    s_vec::Vector{Float64}  # N(0, 1) nodes for integration
    p_vec::Vector{Float64}  # N(0, 1) probs on those nodes
    θ::Float64
    g::Matrix{Float64}    # Stores a function for K to iterate on
    Kg::Matrix{Float64}   # Extra storage

end


function BYComputableModel(ez::EpsteinZin, 
                           by::BYconsumption;
                           std_range::Int64=2,
                           shock_state_size::Int64=12,
                           gs_z=8,       # z grid size
                           gs_σ=8)       # σ grid size

    # Build the grids
    z_min, z_max, σ_min, σ_max, c_max = compute_bounds(by)
    z_grid = linspace(z_min, z_max, gs_σ)
    σ_grid = linspace(σ_min, σ_max, gs_z)
    z_grid = collect(z_grid)
    σ_grid = collect(σ_grid)

    # Extract state and probs for N(0, 1) shocks
    mc = tauchen(shock_state_size, 0, 1, 0, std_range)
    s_vec = mc.state_values
    p_vec = mc.p[1, :]  # Any row, all columns

    # A function g (represented as an array) for K to iterate on
    # plus a second array Kg for extra storage
    g = ones(gs_z, gs_σ)
    Kg = similar(g)

    # Unpack, define parameters
    θ = (1 - ez.γ) / (1 - 1/ez.ψ)

    return BYComputableModel(ez, by, z_grid, σ_grid, s_vec, p_vec, θ, g, Kg)
end


""" 
Apply the operator K induced by the Bansal-Yaron model to a function g.
Populate the vector Kg and return the supremum

    || Kg || := sup_x | Kg(x) |

Uses Gaussian quadrature.  Updates g in BYComputableModel.


"""
function K_interp!(bcm::BYComputableModel)      

    # Unpack parameters
    ez, by = bcm.ez, bcm.by
    β, γ, ψ = ez.β, ez.γ, ez.ψ
    ρ, v, d, ϕ_z, ϕ_σ = by.ρ, by.v, by.d, by.ϕ_z, by.ϕ_σ
    μ = by.μ
    θ = bcm.θ
    p_vec, s_vec = bcm.p_vec, bcm.s_vec
    z_grid, σ_grid = bcm.z_grid, bcm.σ_grid

    # Interpolate g and allocate memory for new g
    g_func = interpolate((z_grid, σ_grid), bcm.g, Gridded(Linear()))

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
            bcm.Kg[i, j] = result
        end
    end

    copy!(bcm.g, bcm.Kg)

    return sup
end


