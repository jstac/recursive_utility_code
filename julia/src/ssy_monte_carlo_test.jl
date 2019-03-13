include("ssy_model.jl")
using QuantEcon
using LinearAlgebra
using Random
using Statistics

function update_state(x::Float64, c_params::Array{Float64})
    μ_c, ρ, ϕ_z, σ_bar, ϕ_c, ρ_hz, σ_hz, ρ_hc, σ_hc = c_params
    z, h_z, h_c = x

    σ_z = ϕ_z * σ_bar * exp(h_z)
    σ_c = ϕ_c * σ_bar * exp(h_c)

    z = ρ * z + sqrt(1 - ρ^2) * σ_z * randn()
    h_z = ρ_hz * h_z + σ_hz * randn()
    h_c = ρ_hc * h_c + σ_hc * randn()

    result = [z, h_z, h_c]
    return result
end

function eval_kappa(x::Float64, y::Float64, c_params::Array{Float64})

    μ_c, ρ, ϕ_z, σ_bar, ϕ_c, ρ_hz, σ_hz, ρ_hc, σ_hc = c_params
    z, h_z, h_c = x
    σ_c = ϕ_c * σ_bar * exp(h_c)

    result = μ_c + z + σ_c * randn()
    return result
end

function ssy_function_factory(ssy::SSYConsumption, ez::EpsteinZin)

    β, γ, ψ = ez.β, ez.γ, ez.ψ
    μ_c, ρ, ϕ_z, σ_bar, ϕ_c = ssy.μ_c, ssy.ρ, ssy.ϕ_z, ssy.σ_bar, ssy.ϕ_c
    ρ_hz, σ_hz, ρ_hc, σ_hc = ssy.ρ_hz, ssy.σ_hz, ssy.ρ_hc, ssy.σ_hc

    function ssy_compute_stat_mc(;initial_state = zeros(3),
                                n = 1000,
                                m = 1000,
                                seed = 1234,
                                burn_in = 500)

        c_params = [μ_c, ρ, ϕ_z, σ_bar, ϕ_c, ρ_hz, σ_hz, ρ_hc, σ_hc]

        Random.seed!(seed)
        x = initial_state

        for t in 1:burn_in
            x = update_state(x, c_params)
        end

        yn_vals = Array{Float64}(undef, m)
        θ = (1 - γ) / (1 - 1/ψ)

        for i in 1:m
            kappa_sum = 0.0

            for t in 1:n
                y = update_state(x, c_params)
                kappa_sum += eval_kappa(x, y, c_params)
                x = y
            end
            yn_vals[i] = exp((1 - γ) * kappa_sum)
        end

        mean_yns = mean(yn_vals)
        Lm = β * mean_yns^(1 / (n * θ))

        return Lm

    return ssy_compute_stat_mc

    end
end
