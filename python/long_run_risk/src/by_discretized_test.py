"""

Calculate stability test value Lambda via discretization, BY model.

Step one is discreization of the consumption process.  The discretization
method uses two iterations of Rouwenhorst.  

The discretized version is a representation of a Markov chain with finitely
many states x = (z, σ) and stochastic matrix giving transition probabilitites
between them.  The σ process is truncated at zero.

More specifically, discretization produces a (2, M) matrix x_states, each
element x of which is a pair (z, σ) stacked vertically, and a transition
matrix Q such that 

    Q[m, mp] = probability of transition x_states[m] -> x_states[mp]

The strategy is to 

1. Discretize the σ process to produce state values σ_1, ..., σ_I

2. For each σ_i, 

    * discretize the z process to get z_{i1}, ... z_{iJ}

In each case, discretization uses Rouwenhorst's method 

The final states are constructed as 

    x_m = (z_{ij}, σ_i), where m = i * J + j
    
Each x_m vector is stacked as a column of x_states.  The transition
probability Q[m, n] from x_m to x_n is computed from the transition matrices
arising from the discretization of σ and z discussed above.


"""

from by_model import *
from quantecon import MarkovChain, rouwenhorst
import numpy as np
from numpy.random import rand, randn
from numba import njit, prange, f8, int64

default_I, default_J = 3, 3

    
def compute_spec_rad(Q):
    """
    Function to compute spectral radius of a matrix.

    """
    return np.max(np.abs(np.linalg.eigvals(Q)))

@njit
def draw_from_cdf(F):
    return np.searchsorted(F, rand())

@njit
def single_to_multi(m, J):
    div = m // J 
    rem = m % J
    return  div, rem

@njit
def multi_to_single(i , j, J):
    return i * J + j


class BYConsumptionDiscretized:
    """
    A class to store the discretized version of BY consumption dynamics.

    """

    def __init__(self,
                    by,
                    I,
                    J,
                    σ_states,
                    σ_P,
                    z_states,
                    z_Q):

        self.by = by
        self.I, self.J = I, J
        self.σ_states = σ_states        # states
        self.σ_P = σ_P                  # transition probs
        self.z_states = z_states        # z[i, :] is z states when σ index = i
        self.z_Q = z_Q                  # z_Q[i, :, :] is trans probs when σ index = i
        self.x_states = None            # Optional storage for X state
        self.x_P = None                 # Optional storage for X transition probs


def discretize(by, I, J, add_x_data=False):
    """
    And here's the actual discretization process.  

    """

    # Unpack consumption parameters
    ρ, ϕ_z, v, d, ϕ_σ = by.ρ, by.ϕ_z, by.v, by.d, by.ϕ_σ


    # Discretize σ first
    σ_mc = rouwenhorst(I, d, ϕ_σ, v)
    σ_P = σ_mc.P
    σ_states = np.sqrt(np.maximum(σ_mc.state_values, 0))


    # Allocate memory
    M = I * J
    z_states = np.empty((I, J))
    z_Q = np.empty((I, J, J))
    x_states = np.empty((2, M))
    x_P = np.empty((M, M))
    
    # Discretize z at each σ_i and record state values for z in z_states.
    # Also, record transition probability from z_states[i, j] to 
    # z_states[i, jp] when σ = σ_i.  Store it as q[i, j, jp].
    for i, σ in enumerate(σ_states):
        mc_z = rouwenhorst(J, 0.0, ϕ_z * σ, ρ) 
        for j in range(J): 
            z_states[i, j] = mc_z.state_values[j]
            for jp in range(J):
                z_Q[i, j, jp] = mc_z.P[j, jp]  


    # Create an instance of BYConsumptionDiscretized to store output
    byd = BYConsumptionDiscretized(by, 
                                   I, J, 
                                   σ_states,
                                   σ_P,
                                   z_states, 
                                   z_Q) 

    if add_x_data:
        byd.x_states, byd.x_P = build_x_mc(byd)

    return byd


def build_x_mc(byd):
    """
    Build the overall state process X.

    """
    I, J, σ_states, σ_P, z_states, z_Q = \
            byd.I, byd.J, byd.σ_states, byd.σ_P, byd.z_states, byd.z_Q 

    M = I * J

    x_states = np.zeros((2, M))
    x_P = np.zeros((M, M))

    for m in range(M):
        i, j = single_to_multi(m, J)
        x_states[:, m] = (z_states[i, j], σ_states[i])
        for mp in range(M):
            ip, jp = single_to_multi(mp, J)
            x_P[m, mp] = σ_P[i, ip] * z_Q[i, j, jp]

    return x_states, x_P


def compute_K(byd):
    """
    Compute K matrix in the BY model.

    """

    ψ = byd.by.ψ
    γ = byd.by.γ
    β = byd.by.β

    θ = (1 - γ) / (1 - 1/ψ)
    μ_c = byd.by.μ_c
    g = 1 - γ
    I, J = byd.I, byd.J
    M = I * J

    σ_states = byd.σ_states
    z_states = byd.z_states

    if byd.x_states is None:
        x_states, x_P = build_x_mc(byd)
    else: 
        x_states, x_P = byd.x_states, byd.x_P

    K_matrix = np.empty((M, M))

    for m in range(M):
        for mp in range(M):
            i, j = single_to_multi(m, J)
            σ, z = σ_states[i], z_states[i, j]
            a = np.exp(g * (μ_c + z) + 0.5 * g**2 * σ**2)
            K_matrix[m, mp] =  a * x_P[m, mp]

    return β**θ * K_matrix


def test_val_spec_rad(by, I=default_I, J=default_J):
    """
    Compute the test value Lambda

    """
    ψ, γ, β = by.ψ, by.γ, by.β
    θ = (1 - γ) / (1 - 1/ψ)

    byd = discretize(by, I, J)
    K_matrix = compute_K(byd)
    rK = compute_spec_rad(K_matrix)

    return rK**(1/θ)


def mc_factory(by, I=default_I, J=default_J, parallel_flag=True):
    """
    Compute the test value Lambda via Monte Carlo, under the discretized
    consumption dynamics.

    Note that the seed currently has no effect when parallel_flag=True.

    """

    ψ = by.ψ
    γ = by.γ
    β = by.β
    μ_c = by.μ_c

    θ = (1 - γ) / (1 - 1/ψ)
    byd = discretize(by, I, J)

    σ_states = byd.σ_states
    σ_P_cdf = byd.σ_P.cumsum(axis=1)
    z_states = byd.z_states

    z_Q_cdf = np.empty_like(byd.z_Q)
    for i in range(I):
        for j in range(J):
            z_Q_cdf[i, j, :] = byd.z_Q[i, j, :].cumsum()

    M = I * J

    # Initial state is from the middle of the state space,
    # which approximates a draw from the stationary distribution.
    # Tried drawing from the true stationary distribution and it made little
    # difference to the output and greatly increased runtime.
    i_init, j_init = I // 2, J // 2

    @njit(parallel=parallel_flag)
    def test_val_mc(n=1000, m=1000, seed=1234):
            
        np.random.seed(seed)
        yn_vals = np.empty(m)

        for m_idx in prange(m):
            kappa_sum = 0.0
            i, j = i_init, j_init 

            for n_idx in range(n):
                # Increment consumption
                kappa_sum += μ_c + z_states[i, j] + σ_states[i] * randn()
                # Update state
                j = draw_from_cdf(z_Q_cdf[i, j, :])
                i = draw_from_cdf(σ_P_cdf[i, :])

            yn_vals[m_idx] = np.exp((1-γ) * kappa_sum)

        mean_yns = np.mean(yn_vals)
        Lm = β * mean_yns**(1 / (n * θ))

        return Lm

    return test_val_mc




############### Tests #################

@njit
def test_draw_from_cdf(num_reps=100_000_000):
    F = np.array((0.2, 0.3, 0.5))
    F = np.cumsum(F)
    x = np.zeros(3, dtype=int64)
    for i in range(num_reps):
        d = draw_from_cdf(F)
        x[d] += 1
    return x / num_reps



