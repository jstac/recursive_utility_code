#=

Code for solving the Schorfheide, Song and Yaron model. 
    
=#

using StatsBase

include("ez_model.jl")


"""
Consumption process parameters of SSY model

Log consumption growth g_c is given by

    g_c = μ_c + z + σ_c η'

    z' = ρ z + sqrt(1 - ρ^2) σ_z e'

    σ_z = ϕ_z σ_bar exp(h_z)

    σ_c = ϕ_c σ_bar exp(h_c)

    h_z' = ρ_hz h_z + σ_hz u'

    h_c' = ρ_hc h_c + σ_hc w'

Here {e}, {u} and {w} are IID and N(0, 1).  



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




"""
Simulate the state process and consumption for the SSY model.  

Returns

    * gc[2], ..., X[ts_length]
    * X[1], ..., X[ts_length]

where gc[t] = ln(C[t]) - ln(C[t-1])


"""
function simulate(sc::SSYConsumption; 
                  z0=0.0, 
                  h_z0=0.0,
                  h_c0=0.0, 
                  h_d0=0.0,
                  ts_length=1000000, 
                  seed=nothing)

    if seed != nothing
        srand(seed)
    end

    # Unpack
    μ_c, ρ, ϕ_z, σ_bar, ϕ_c = sc.μ_c, sc.ρ, sc.ϕ_z, sc.σ_bar, sc.ϕ_c
    ρ_hz, σ_hz, ρ_hc, σ_hc = sc.ρ_hz, sc.σ_hz, sc.ρ_hc, sc.σ_hc

    # Map h to σ
    tz(h_z) = ϕ_z * σ_bar * exp(h_z)
    tc(h_c) = ϕ_c * σ_bar * exp(h_c)

    # Allocate memory for states 
    z_vals = Array{Float64, 1}(ts_length)
    h_z_vals = Array{Float64, 1}(ts_length)
    h_c_vals = Array{Float64, 1}(ts_length)

    # Initialize
    z_vals[1],  h_z_vals[1], h_c_vals[1] = z0, h_z0, h_c0 

    # Allocate memory consumption and dividends
    c_vals = Array{Float64, 1}(ts_length)

    # Simulate all stochastic processes 
    for t in 1:(ts_length-1)
        # Simplify names
        h_z, h_c = h_z_vals[t], h_c_vals[t]
        σ_z, σ_c = tz(h_z), tc(h_c)
        # Evaluate consumption 
        c_vals[t+1] = μ_c + z_vals[t] + σ_c * randn()
        # Update states
        z_vals[t+1] = ρ * z_vals[t] + sqrt(1 - ρ^2) * σ_z * randn()
        h_z_vals[t+1] = ρ_hz * h_z + σ_hz * randn()
        h_c_vals[t+1] = ρ_hc * h_c + σ_hc * randn()
    end

    return c_vals[2:end], z_vals, h_z_vals, h_c_vals
end



"""
The model constructed for computation, including parameters, derived
parameters, grids, etc.

The order for arguments of g is g(z, h_z, h_c)

"""
struct SSYComputableModel{T <: Real}

    ez::EpsteinZin{T}
    sc::SSYConsumption{T}

    z_grid::Vector{Float64}
    h_z_grid::Vector{Float64}
    h_c_grid::Vector{Float64}

    w_star::Array{Float64}      # Stores fixed point of T once computed
    shocks::Array{Array{Float64, 1}, 1}    # Array of Gaussian shocks for MC
end


"""
And its constructor.

"""
function SSYComputableModel(ez::EpsteinZin, 
                            sc::SSYConsumption,
                            n_shocks::Int64=1000,    # for MC integrals
                            q=0.01,                  # quantitle for states
                            gs_z=8,                  # z grid size
                            gs_h_z=4,                # h_z grid size
                            gs_h_c=4)                # h_c grid size
                            

    c_vals, z_vals, h_z_vals, h_c_vals = simulate(sc)

    # And for states
    z_max = quantile(z_vals, 1 - q)
    h_z_max = quantile(h_z_vals, 1 - q)
    h_c_max = quantile(h_c_vals, 1 - q)

    z_min = quantile(z_vals, q)
    h_z_min = quantile(h_z_vals, q)
    h_c_min = quantile(h_c_vals, q)

    # Now build the grids
    z_grid = collect(linspace(z_min, z_max, gs_z))
    h_z_grid = collect(linspace(h_z_min, h_z_max, gs_h_z))
    h_c_grid = collect(linspace(h_c_min, h_c_max, gs_h_c))

    # A three-dimensional array to store an estimate of the fixed point of T
    w_star = Array{Float64}(gs_z, gs_h_z, gs_h_c)

    shocks = [randn(3) for n in 1:n_shocks]

    return SSYComputableModel(ez, 
                              sc, 
                              z_grid, 
                              h_z_grid, 
                              h_c_grid,
                              w_star,
                              shocks) 
end


"""
If called with nothing, use all defaults.

"""
function SSYComputableModel()
    ez = EpsteinZinSSY()
    sc = SSYConsumption()
    return SSYComputableModel(ez, sc) 
end


