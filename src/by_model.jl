#=

Code for solving the Bansal-Yaron model based around Monte Carlo and
interpolation.

The Bansal Yaron consumption process is

    g = μ_c + z + σ η

    z' = ρ z + ϕ_z σ e'

    (σ^2)' = v σ^2 + d + ϕ_σ w'

where {e} and {w} are IID and N(0, 1). 


=#

using StatsBase

include("ez_model.jl")


"""
Struct for parameters of the BY model as described above.

"""
struct BYConsumption{T <: Real}  
    μ_c::T
    ρ::T
    ϕ_z::T
    v::T
    d::T
    ϕ_σ::T
end


"""
A constructor using parameters from the BY paper.

"""
function BYConsumption(;μ_c=0.0015,
                        ρ=0.979,
                        ϕ_z=0.044,
                        v=0.987,
                        d=7.9092e-7,
                        ϕ_σ= 2.3e-6)

    return BYConsumption(μ_c,
                        ρ,
                        ϕ_z,
                        v,
                        d,
                        ϕ_σ)     
end

"""
Simulate the state process and consumption for the BY model.  

Returns

    * gc[2], ..., X[ts_length]
    * X[1], ..., X[ts_length]

where gc[t] = ln(C[t]) - ln(C[t-1])


"""
function simulate(by::BYConsumption; 
                  z0=0.0, 
                  σ0=0.0,
                  ts_length=1000000, 
                  seed=nothing)

    if seed != nothing
        srand(seed)
    end

    # Unpack
    ρ, ϕ_z, v, d, ϕ_σ = by.ρ, by.ϕ_z, by.v, by.d, by.ϕ_σ
    μ_c = by.μ_c

    z_vals = Array{Float64, 1}(ts_length)
    σ_vals = Array{Float64, 1}(ts_length)
    c_vals = Array{Float64, 1}(ts_length)

    z_vals[1] = z0
    σ_vals[1] = σ0

    for t in 1:(ts_length-1)
        σ2 = v * σ_vals[t]^2 + d + ϕ_σ * randn()
        σ = sqrt(max(σ2, 0))
        c_vals[t+1] = μ_c + z_vals[t] + σ * randn()
        σ_vals[t+1] = σ
        z_vals[t+1] = ρ * z_vals[t] + ϕ_z * σ * randn()
    end

    return c_vals[2:end], z_vals, σ_vals
end


"""
Compute upper and lower bounds for the state variables and consumption,
in order to truncate.  The function simulates and then picks high and low
quantiles.

"""
function compute_bounds(by::BYConsumption; 
                        q=0.025,        # quantitle for state
                        p=0.05)         # quantile for consumption 

    # Set the seed to minimize variation
    srand(1234)

    c_vals, z_vals, σ_vals = simulate(by)

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
    by::BYConsumption{T}
    z_grid::Vector{Float64}
    σ_grid::Vector{Float64}
    s_vec::Vector{Float64}  # N(0, 1) nodes for integration
    p_vec::Vector{Float64}  # N(0, 1) probs on those nodes

end


function BYComputableModel(ez::EpsteinZin, 
                           by::BYConsumption;
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

    # A function (represented as an array) that stores the fixed point of T
    w_star = Array{Float64, 1}(gs_z, gs_σ)

    return BYComputableModel(ez, by, z_grid, σ_grid, s_vec, p_vec, w_star)
end

