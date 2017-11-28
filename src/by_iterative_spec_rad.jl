
include("by_model.jl")

""" 
Apply the operator K induced by the Bansal-Yaron model to a function g.
Populate the vector Kg and return the supremum

    || Kg || := sup_x | Kg(x) |

Uses Gaussian quadrature.  Updates g in BYComputableModel.


"""
function K_interp(g, bcm::BYComputableModel)      

    # Unpack parameters
    ez, by = bcm.ez, bcm.by
    β, γ, ψ = ez.β, ez.γ, ez.ψ
    θ = ez.θ
    ρ, v, d, ϕ_z, ϕ_σ = by.ρ, by.v, by.d, by.ϕ_z, by.ϕ_σ
    μ_c = by.μ_c
    p_vec, s_vec = bcm.p_vec, bcm.s_vec
    z_grid, σ_grid = bcm.z_grid, bcm.σ_grid

    # Interpolate g and allocate memory for new g
    g_func = interpolate((z_grid, σ_grid), g, Gridded(Linear()))
    Kg = similar(g)

    # Apply the operator K to g, computing Kg and || Kg ||
    sup = -Inf
    for (i, z) in enumerate(z_grid)
        for (j, σ) in enumerate(σ_grid)
            mf = exp((1 - γ) * (μ_c + z) + (1 - γ)^2 * σ^2 / 2)
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
            Kg[i, j] = result
        end
    end

    return sup, Kg
end


"""
Compute the spectral radius by iteration.

"""
function compute_spec_rad_coeff(ez::EpsteinZin,
                                by::BYConsumption;
                                tol=1e-6, 
                                max_iter=5000) 

    bcm = BYComputableModel(ez, by)

    g = ones(size(bcm.w_star))
    error = tol + 1
    r = 1
    i = 1

    while error > tol && i < max_iter
        s, Kg = K_interp(g, bcm)
        new_r = s^(1/i)
        error = abs(new_r - r)
        i += 1
        r = new_r
        g = Kg
    end

    return r^bcm.ez.θ
end

