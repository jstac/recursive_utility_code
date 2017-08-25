#=

Code for solving the Bansal-Yaron model based around Monte Carlo and
interpolation.

=#

using Interpolations
include("ez_model.jl")
include("bansal_yaron_state_process.jl")

function T_interp(ez::EpsteinZin,
                  sv::StochasticVolatility,
                  w_in::Matrix{Float64},
                  w_out::Matrix{Float64},
                  z_grid::AbstractVector,
                  σ_grid::AbstractVector,
                  η_vec::Vector{Float64},
                  ω_vec::Vector{Float64},
                  μ=0.0015)

    # Unpack parameters
    β, γ, θ = ez.β, ez.γ, ez.θ
    ρ, v, d, ϕ_z, ϕ_σ = sv.ρ, sv.v, sv.d, sv.ϕ_z, sv.ϕ_σ

    η_shock_size = length(η_vec)
    ω_shock_size = length(ω_vec)

    # Interpolate w and allocate memory for new w
    w_func = interpolate((z_grid, σ_grid), w_in, Gridded(Linear()))

    # Apply the operator K
    for (i, z) in enumerate(z_grid)
        for (j, σ) in enumerate(σ_grid)
            mf = exp((1 - γ) * (μ + z) + (1 - γ)^2 * σ^2 / 2)
            s = 0.0
            for η in η_vec
                for ω in ω_vec
                    zp = ρ * z + ϕ_z * σ * η
                    σp2 = v * σ^2 + d + ϕ_σ * ω
                    σp = σp2 < 0 ? 1e-8 : sqrt(σp2)
                    s += w_func[zp, σp]^θ
                end
            end
            w_out[i, j] = 1 - β + β *(mf * s / (η_shock_size * ω_shock_size))^(1 / θ)
        end
    end
end
