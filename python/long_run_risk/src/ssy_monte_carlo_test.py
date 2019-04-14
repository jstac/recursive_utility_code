"""
Monte Carlo based computation of the test value, SSY model.

"""

from numpy.random import randn
from numba import jit, njit, f8, prange

from ssy_model import *

@njit(f8[:](f8[:], f8[:]))
def update_state(x, c_params):

    μ_c, ρ, ϕ_z, σ_bar, ϕ_c, ρ_hz, σ_hz, ρ_hc, σ_hc = c_params
    z, h_z, h_c = x

    # compute sigs
    σ_z = ϕ_z * σ_bar * np.exp(h_z)
    σ_c = ϕ_c * σ_bar * np.exp(h_c)
    # update state
    z = ρ * z + np.sqrt(1 - ρ**2) * σ_z * randn()
    h_z = ρ_hz * h_z + σ_hz * randn()
    h_c = ρ_hc * h_c + σ_hc * randn()

    return np.array((z, h_z, h_c))


@njit(f8(f8[:], f8[:], f8[:]))
def eval_kappa(x, y, c_params):
    """
    Computes kappa_{t+1} given z_t and sigma_t
    """
    μ_c, ρ, ϕ_z, σ_bar, ϕ_c, ρ_hz, σ_hz, ρ_hc, σ_hc = c_params

    z, h_z, h_c = x
    σ_c = ϕ_c * σ_bar * np.exp(h_c)
    return μ_c + z + σ_c * randn()


def ssy_function_factory(ssy,  parallelization_flag=False):
    """
    Produces functions that compute the stability test value Lambda via 
    Monte Carlo.

    """
    β, γ, ψ = ssy.β, ssy.γ, ssy.ψ 
    μ_c, ρ, ϕ_z, σ_bar, ϕ_c = ssy.μ_c, ssy.ρ, ssy.ϕ_z, ssy.σ_bar, ssy.ϕ_c 
    ρ_hz, σ_hz, ρ_hc, σ_hc = ssy.ρ_hz, ssy.σ_hz, ssy.ρ_hc, ssy.σ_hc 

    @njit(parallel=parallelization_flag)
    def ssy_compute_stat_mc(initial_state=np.zeros(3), 
                         n=1000, 
                         m=1000,
                         seed=1234,
                         burn_in=500):
        """
        Compute the stability test value Lambda via Monte Carlo.

        """

        np.random.seed(seed)

        c_params = μ_c, ρ, ϕ_z, σ_bar, ϕ_c, ρ_hz, σ_hz, ρ_hc, σ_hc
        c_params = np.array(c_params)

        # Implement some burn in for the state
        x = initial_state
        for t in range(burn_in):
            x = update_state(x, c_params)

        # Compute samples for MC stat
        yn_vals = np.empty(m)
        θ = (1 - γ) / (1 - 1/ψ)

        for i in prange(m):
            kappa_sum = 0.0
            x = initial_state

            for t in range(n):
                y = update_state(x, c_params)
                kappa_sum += eval_kappa(x, y, c_params)
                x = y

            yn_vals[i] = np.exp((1-γ) * kappa_sum)

        mean_yns = np.mean(yn_vals)
        log_Lm = np.log(β) +  (1 / (n * θ)) * np.log(mean_yns)

        return np.exp(log_Lm)

    return ssy_compute_stat_mc




