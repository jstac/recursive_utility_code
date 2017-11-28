include("ssy_model.jl")

using Interpolations


"""
Apply the operator 

    Kg(z, h_z, h_c) 
        = E g(z', h_z', h_c') exp((1-γ)(μ_c + z) + (1-γ)^2 σ_c^2 / 2)

where σ_c = ϕ_c σ_bar exp(h_c), and (z', h_z', h_c') update via the dynamics
for the SSY model given above.  When we write x as the state, the meaning is

    x = (z, h_z, h_c)

induced by the SSY model to a function g.  Compute

    || Kg || := sup_x | Kg(x) |

Input data is g.  Integrals are computed by Monte Carlo.

"""
function K_interp(g, scm::SSYComputableModel)

    # Unpack parameters
    ez, sc = scm.ez, scm.sc
    β, γ = ez.β, ez.γ
    μ_c, ρ, ϕ_z, σ_bar, ϕ_c = sc.μ_c, sc.ρ, sc.ϕ_z, sc.σ_bar, sc.ϕ_c
    ρ_hz, σ_hz, ρ_hc, σ_hc = sc.ρ_hz, sc.σ_hz, sc.ρ_hc, sc.σ_hc
    p_vec, s_vec = scm.p_vec, scm.s_vec
    z_grid, h_z_grid, h_c_grid = scm.z_grid, scm.h_z_grid, scm.h_c_grid 

    # Some useful constants
    δ = sqrt(1 - ρ^2)
    τ_z = ϕ_z * σ_bar
    τ_c = ϕ_c * σ_bar

    # Storage
    Kg = similar(g)

    # Interpolate g 
    g_func = interpolate((z_grid, h_z_grid, h_c_grid), g, Gridded(Linear()))

    # Apply the operator K to g, computing Kg and || Kg ||
    n = length(scm.shocks)
    sup = -Inf
    for (i, z) in enumerate(z_grid)
        for (k, h_c) in enumerate(h_c_grid)
            σ_c = τ_c * exp(h_c)
            mf = exp((1 - γ) * (μ_c + z) + (1 - γ)^2 * σ_c^2 / 2)
            for (j, h_z) in enumerate(h_z_grid)
                σ_z = τ_z * exp(h_z)
                g_exp = 0.0
                for (η, ω, ϵ) in scm.shocks
                    zp = ρ * z + δ * σ_z * η
                    h_cp = ρ_hc * h_c + σ_hc * ω
                    h_zp = ρ_hz * h_z + σ_hz * ϵ
                    g_exp += g_func[zp, h_zp, h_cp]
                end
                result = β^ez.θ * mf * (g_exp / n)
                if result > sup
                    sup = result
                end
                Kg[i, j, k] = result
            end
        end
    end

    return sup, Kg
end



"""
Compute the spectral radius by iteration.

"""
function compute_spec_rad_coeff_by_iter(scm::SSYComputableModel;
                                tol=1e-6, 
                                max_iter=5000) 

    g = ones(size(scm.w_star))

    error = tol + 1
    r = 1
    i = 1

    while error > tol && i < max_iter
        s, Kg = K_interp(g, scm)
        new_r = s^(1/i)
        error = abs(new_r - r)
        i += 1
        r = new_r
        g = Kg
    end

    return r^scm.ez.θ
    
end

"""
Compute the spectral radius by iteration.

"""
function compute_spec_rad_coeff_by_iter(ez::EpsteinZin,
                                sc::SSYConsumption;
                                q=0.01, 
                                tol=1e-6, 
                                max_iter=5000) 

    scm = SSYComputableModel(ez, sc, q=q)
    
    return compute_spec_rad_coeff_by_iter(scm::SSYComputableModel,
                                  tol=tol,
                                  max_iter=max_iter)
end



