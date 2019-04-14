"""
The model of Johannes, Lochstoer and Mou (2061)

The JLM consumption process is

    ln(C_{t+1}/C_t) = mu(X_{t+1}) + sigma(X_{t+1}) xi_{t+1}

where {xi_t} is IID and N(0, 1), while X_t follows a two state Markov
chain. 

"""

import numpy as np
from numpy.random import rand, randn
from numba import jit, njit, prange, f8, int64
from quantecon import MarkovChain

# Switchable parallelization flag (a hack to deal with failure of seeding
# under parallelization).
parallelization_flag = True


@njit(int64(int64, f8[:]))
def update_state(x, c_params):

    μ_1, μ_2, σ_1, σ_2, p_11, p_22 = c_params

    u = rand()
    if x == 1:
        if u > p_11:
            x = 2
    else:
        if u > p_22:
            x = 1
    return x


@njit(f8(int64, int64, f8[:]))
def eval_kappa(x, y, c_params):
    """
    Computes kappa_{t+1} given z_t and sigma_t
    """

    μ_1, μ_2, σ_1, σ_2, p_11, p_22 = c_params

    xi = randn()
    if y == 1:
        return μ_1 + σ_1 * xi
    else:
        return μ_2 + σ_2 * xi


@njit(parallel=parallelization_flag)
def jlm_compute_stat(β=0.994,
                     γ=10.0, 
                     ψ=1.5,
                     μ_1=0.007,
                     μ_2=0.0013,
                     σ_1=0.0015,
                     σ_2=0.0063,
                     p_11=0.93,
                     p_22=0.83,
                     initial_state=1, 
                     n=1000, 
                     m=1000,
                     seed=1234,
                     burn_in=500):

    c_params = np.array((μ_1, μ_2, σ_1, σ_2, p_11, p_22))

    np.random.seed(seed)
    yn_vals = np.empty(m)

    x = initial_state
    θ = (1 - γ) / (1 - 1/ψ)

    # Implement some burn in for the state
    for t in range(burn_in):
        x = update_state(x, c_params)

    # Compute samples for MC stat
    for i in prange(m):
        kappa_sum = 0.0

        for t in range(n):
            y = update_state(x, c_params)
            kappa_sum += eval_kappa(x, y, c_params)
            x = y

        yn_vals[i] = np.exp((1-γ) * kappa_sum)

    mean_yns = np.mean(yn_vals)
    Lambda = β * mean_yns**(1 / (n * θ))

    return Lambda 


def jlm_compute_stat_discrete(β=0.994,
                             γ=10.0, 
                             ψ=1.5,
                             μ_1=0.007,
                             μ_2=0.0013,
                             σ_1=0.0015,
                             σ_2=0.0063,
                             p_11=0.93,
                             p_22=0.83):


    Q = ((p_11, 1 - p_11), (1 - p_22, p_22))
    Q = np.array(Q)
    K = np.empty_like(Q)
    θ = (1 - γ) / (1 - 1/ψ)
    g = 1 - γ
    m = μ_1, μ_2
    s = σ_1, σ_2

    for x in 0, 1:
        for y in 0, 1:
            K[x, y] = np.exp(g * m[y] + 0.5 * g**2 * s[y]**2) * Q[x, y]

    rK = np.max(np.linalg.eigvals(K))
    MC = rK**(1 / (1 - γ))

    return MC, β * MC**(1 - 1/ψ)


@njit
def replicate_default_sim(n=1000, m=1000, k=1000):

    draws = np.empty(k)
    for i in range(k):
        draws[i] = jlm_compute_stat(n=n, m=m, seed=i)

    return draws


