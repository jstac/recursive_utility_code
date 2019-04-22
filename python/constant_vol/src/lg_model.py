"""

Some functions for working with the Linear Gaussian model.  Consumption growth is

    ln (C_{t+1} / C_t) = μ_c + X_{t+1} + σ_c e_{t+1}

    X_{t+1} = ρ X_t + σ W_{t+1}

with e_t and W_t iid and N(0, 1).  

The aim is to compute the risk-adjusted long run mean growth rate of
consumption, which is the limit of

    { E exp( (1-γ) sum_{t=1}^n M_t }^{1/((1-γ) n)}

where

    M_t = μ_c + X_{t+1} + σ_c e_{t+1}

"""


import numpy as np
from numpy import sqrt, exp
from numpy.random import randn
from numba import njit, prange
from scipy.linalg import eigvals
import quantecon as qe


class LinearGaussian:
    """
    Represents the model.

    """

    def __init__(self, γ=10.0,
                       β=0.998,
                       ψ=1.5,   
                       ρ=0.979,
                       σ=0.00034, 
                       σ_c=0.0078,
                       μ_c=0.0015):

        self.ρ, self.σ, self.σ_c, self.μ_c = ρ, σ, σ_c, μ_c  
        self.γ, self.β, self.ψ = γ, β, ψ



def lrm_mc_factory(lg, m=1000, n=1000, parallel_flag=True):
    """
    Compute MC by Monte Carlo.

    * lg is an instance of LinearGaussian

    Below, 
    
        Y_j = exp((1 - γ) sum_{i=1}^n M_i^(j)) 
        
    where j indexes one time series path.

    The return value is 

         Y.mean()^(1 / ((1 - γ) * n)

    """

    γ, ρ, σ, σ_c, μ_c= lg.γ, lg.ρ, lg.σ, lg.σ_c, lg.μ_c 

    @njit(parallel=parallel_flag)
    def lrm_mc(m=1000, n=1000):

        Y_sum = 0.0

        for j in prange(m):

            M_sum = 0.0
            X = 0.0

            for i in range(n):
                M_sum += μ_c + X + σ_c * randn()
                X = ρ * X + σ * randn()

            Y_sum += np.exp((1 - γ) * M_sum)

        Y_mean = Y_sum / m
        return Y_mean**(1 / (n * (1 - γ) ))

    return lrm_mc


def lrm_analytic(lg):
    """
    Compute MC directly.

        * lg is an instance of LinearGaussian

    """
    # Unpack parameters
    γ, ρ, σ, σ_c, μ_c= lg.γ, lg.ρ, lg.σ, lg.σ_c, lg.μ_c 

    t = σ**2 /  (1 - ρ)**2
    return np.exp(μ_c + (1 - γ) * 0.5 * (σ_c**2 + t))


def lrm_discretized(lg, D=10):

    # Unpack parameters
    γ, ρ, σ, σ_c, μ_c= lg.γ, lg.ρ, lg.σ, lg.σ_c, lg.μ_c 

    # Discretize the state process
    X_mc = qe.rouwenhorst(D, 0.0, σ, ρ)
    x_vals = X_mc.state_values
    Q = X_mc.P

    # Build the matrix K(x, y) = exp((1-γ) y) Q(x, y)
    K = np.empty((D, D))
    for i, x in enumerate(x_vals):
        for j, y in enumerate(x_vals):
            K[i, j] = np.exp((1-γ) * (μ_c + x) + ((1-γ)**2) * 0.5 * σ_c**2 ) * Q[i, j]

    # Return MC
    rK = np.max(np.abs(eigvals(K)))
    return rK**(1 / (1-γ))
