"""
The EZ prob one test value, SSY model.

"""

import numpy as np
from ssy_model import *


def ssy_prob_one_test_val(ssy, eta_hat=3.0):
    """
    Compute the EZ prob one stability test value.

    * eta_hat is a common upper truncation val for all shocks

    Max vals have a hat after them.

    """
    β, γ, ψ = ssy.β, ssy.γ, ssy.ψ 
    μ_c, ρ, ϕ_z, σ_bar, ϕ_c = ssy.μ_c, ssy.ρ, ssy.ϕ_z, ssy.σ_bar, ssy.ϕ_c 
    ρ_hz, σ_hz, ρ_hc, σ_hc = ssy.ρ_hz, ssy.σ_hz, ssy.ρ_hc, ssy.σ_hc 

    hz_hat = σ_hz * eta_hat / (1 - ρ_hz)
    hc_hat = σ_hc * eta_hat / (1 - ρ_hc)

    σ_z_hat = ϕ_z * σ_bar * np.exp(hz_hat)
    σ_c_hat = ϕ_c * σ_bar * np.exp(hc_hat)

    z_hat = np.sqrt(1 - ρ**2) * σ_z_hat * eta_hat / (1 - ρ)

    B_c = np.exp(μ_c + z_hat + σ_c_hat * eta_hat)
    

    return β * B_c**(1 - 1/ψ)




