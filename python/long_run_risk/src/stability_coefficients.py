"""

Compute the stability test values for the BY model.

"""

import numpy as np
from models import *


"""
Compute the spectral radius by simulation.

"""
def compute_spec_rad_coef(ez, cp, M=1000, N=750):

    β, γ, ψ = ez.β, ez.γ, ez.ψ
    θ = (1 - γ) / (1 - 1/ψ)
    
    sum_obs = 0.0
    
    for m in range(M):
        c_growth = cp.simulate(seed=m, ts_length=N+1)
        sum_obs += np.exp((1 - γ) * c_growth.sum())

    rK = β**θ * (sum_obs / M)**(1/N)
    return rK**(1/θ)


"""
Compute EZ's stability coefficient.

"""
def compute_ez_coef(ez, cp, q=95):

    β, ψ = ez.β, ez.ψ
                            
    c_growth = cp.simulate()
    c_max = np.percentile(c_growth, q)

    return β * np.exp(c_max)**(1 - (1/ψ)) 





# """
# Convenience function for BY case.

# """
# def compute_by_coefs(β=0.998,
                     # γ=10.0, 
                     # ψ=1.5, 
                     # coef_type="SR", # SR is for spec rad coefficient
                     # q=95,
                     # M=1000, 
                     # N=2000):

    # if coef_type == "SR":
        # r = compute_spec_rad_coef(β, γ, ψ, simulate_by, M, N)
    # else:
        # r = compute_ez_coef(β, ψ, simulate_by, q)

    # return r

# """
# Convenience function for SSY case.

# """
# def compute_ssy_coefs(β=0.999,
                      # γ=8.89, 
                      # ψ=1.97, 
                      # coef_type="SR", # SR is for spec rad coefficient
                      # q=95,
                      # M=1000, 
                      # N=2000):

    # if coef_type == "SR":
        # r = compute_spec_rad_coef(β, γ, ψ, simulate_ssy, M, N)
    # else:
        # r = compute_ez_coef(β, ψ, simulate_by, q)

    # return r





