"""

Compute the equilibrium wealth consumption ratio in the SSY model by first
computing the fixed point of A = phi K.

"""

from ssy_discretized_test import *
import numpy as np

default_K, default_I, default_J = 4, 4, 4

def wealth_cons_ratio(ssyd, 
               tol=1e-7, 
               init_val=1, 
               max_iter=1_000_000,
               verbose=False):
    """
    Iterate to convergence on the Koopmans operator associated with the SSY
    model and then return the wealth consumption ratio.

    """

    # Unpack and set up parameters EpsteinZin parameters
    ψ, γ, β = ssyd.ssy.ψ, ssyd.ssy.γ, ssyd.ssy.β
    θ = (1 - γ) / (1 - 1/ψ)
    ζ = 1 - β

    K_matrix = compute_K(ssyd)
    M = ssyd.K * ssyd.I * ssyd.J
    w = np.ones(M) * init_val
    iter = 0
    error = tol + 1

    r = compute_spec_rad(K_matrix)
    if verbose:
        print(f"Test value = {r**(1/θ)} and θ = {θ}")
        print("Beginning iteration\n\n")


    while error > tol and iter < max_iter:
        Tw = ζ + β * (K_matrix @ (w**θ))**(1/θ)
        error = np.max(np.abs(w - Tw))
        w = Tw
        iter += 1

    if verbose:
        print(f"Iteration converged after {iter} iterations") 

    return  w / ζ


def average_wealth_cons(ssy,
                        K=default_K,
                        I=default_I,
                        J=default_J,
                        verbose=False):
    """
    Computes the mean wealth consumption ratio under the stationary
    distribution pi.

    """

    ssyd = discretize(ssy, K, I, J, add_x_data=True)

    w = wealth_cons_ratio(ssyd, verbose=verbose)

    x_mc = MarkovChain(ssyd.x_P)
    x_pi = x_mc.stationary_distributions[0]

    mean_w = w @ x_pi

    return mean_w

