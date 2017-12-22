#=

Representations of Bansal--Yaron and Schorfheide, Song & Yaron consumption
processes

=#

abstract type ConsumptionProcess
end


#=

The Bansal Yaron consumption process is (see p. 1487)

    z' = ρ z + ϕ_z σ e'                 # z, ϕ_z here is x, ϕ_e in BY

    g = μ_c + z + σ η                   # consumption growth, μ_c is μ

    (σ^2)' = v σ^2 + d + ϕ_σ w'         # v, d, ϕ_σ is v_1, σ^2(1-v_1), σ_w

where {e} and {w} are IID and N(0, 1). 

=#



"""
Struct for parameters of the BY model as described above.

"""
struct BYConsumption{T <: Real} <: ConsumptionProcess
    μ_c::T
    ρ::T
    ϕ_z::T
    v::T
    d::T
    ϕ_σ::T
end


"""
A constructor using parameters from the BY paper.  See table IV on page 1489.

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





#=

The Schorfheide, Song and Yaron model consumption process. 
    
Log consumption growth g_c is given by

    g_c = μ_c + z + σ_c η'

    z' = ρ z + sqrt(1 - ρ^2) σ_z e'

    σ_z = ϕ_z σ_bar exp(h_z)

    σ_c = ϕ_c σ_bar exp(h_c)

    h_z' = ρ_hz h_z + σ_hz u'

    h_c' = ρ_hc h_c + σ_hc w'

Here {e}, {u} and {w} are IID and N(0, 1).  


=#



"""
Consumption process parameters of SSY model


"""
struct SSYConsumption{T <: Real}  <: ConsumptionProcess
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


#=

 ============ Simulation functions =============

=# 


"""
Simulate the BY state process and consumption

Returns

    * X[1], ..., X[ts_length]
    * gc[2], ..., X[ts_length]
    * gd[2], ..., X[ts_length]

where

    gc[t] = ln(C[t]) - ln(C[t-1])


"""
function simulate(cp::BYConsumption; ts_length=1000000, seed=1234)

    srand(seed)

    # Unpack
    ρ, ϕ_z, v, d, ϕ_σ = cp.ρ, cp.ϕ_z, cp.v, cp.d, cp.ϕ_σ
    μ_c = cp.μ_c

    # Allocate memory
    c_growth = zeros(ts_length)

    z = 0
    σ = d / (1 - v)

    for t in 1:(ts_length-1)
        # Evaluate consumption and dividends
        c_growth[t+1] = μ_c + z + σ * randn()

        # Update state
        σ2 = v * σ^2 + d + ϕ_σ * randn()
        σ = sqrt(max(σ2, 0))
        z = ρ * z + ϕ_z * σ * randn()
    end

    return c_growth[2:end]
end



"""
Simulate the state process and consumption for the SSY model.  

Returns

    * X[1], ..., X[ts_length]
    * gc[2], ..., gc[ts_length]

where

    gc[t] = ln(C[t]) - ln(C[t-1])


"""
function simulate(cp::SSYConsumption; ts_length=1000000, seed=1234)

    srand(seed)

    # Unpack
    μ_c, ρ, ϕ_z, σ_bar, ϕ_c = cp.μ_c, cp.ρ, cp.ϕ_z, cp.σ_bar, cp.ϕ_c
    ρ_hz, σ_hz, ρ_hc, σ_hc = cp.ρ_hz, cp.σ_hz, cp.ρ_hc, cp.σ_hc

    # Map h to σ
    tz(h_z) = ϕ_z * σ_bar * exp(h_z)
    tc(h_c) = ϕ_c * σ_bar * exp(h_c)

    # Allocate memory for states with initial conditions at the stationary
    # mean, which is zero
    z, h_z, h_c = 0.0, 0.0, 0.0

    # Allocate memory consumption 
    c_growth = zeros(ts_length)

    # Simulate all stochastic processes 
    for t in 1:(ts_length-1)
        # Simplify names
        σ_z, σ_c = tz(h_z), tc(h_c) 
        
        # Evaluate consumption and dividends
        c_growth[t+1] = μ_c + z + σ_c * randn()

        # Update states
        z = ρ * z + sqrt(1 - ρ^2) * σ_z * randn()
        h_z = ρ_hz * h_z + σ_hz * randn()
        h_c = ρ_hc * h_c + σ_hc * randn()
    end

    return c_growth[2:end]
end


