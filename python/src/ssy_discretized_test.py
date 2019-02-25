"""
Calculate stability test value Lambda via discretization.

Step one is discreization of the consumption process in Schorfheide, Song and
Yaron, where log consumption growth g is given by

    g = μ + z + σ_c η'

    z' = ρ z + sqrt(1 - ρ^2) σ_z e'

    σ_z = ϕ_z σ_bar exp(h_z)

    σ_c = ϕ_c σ_bar exp(h_c)

    h_z' = ρ_hz h_z + σ_hz u'

    h_c' = ρ_hc h_c + σ_hc w'

Here {e}, {u} and {w} are IID and N(0, 1).  

The discretization method uses iterations of the Tauchen method.  The indices
are

    σ_c[k] for k in range(K)
    σ_z[i] for i in range(I)
    z[i, :] is all z states when σ_z = σ_z[i] and z[i, j] is j-th element

The discretized version is a representation of a Markov chain with finitely
many states 

    x = (σ_c, σ_z, z)

and stochastic matrix Q giving transition probabilitites between them.

The set of states x_states is computed as a 3 x M matrix with each column
corresponding to values of (σ_c, σ_z, z)'

Discretize the SSY state process builds the discretized state values for the
    states (σ_c, σ_z, z) and a transition matrix Q such that 

    Q[m, mp] = probability of transition x[m] -> x[mp]

where

    x[m] := (σ_c[k], σ_z[i], z[i,j])
    
The rule for the index is

    m = k * (I * J) + i * J + j



@Stearn 
@jstac

Tue Feb 26 04:38:57 AEDT 2019

"""

from ssy_model import *
from quantecon import tauchen
import numpy as np
from scipy.linalg import eigvals

    

class SSYConsumptionDiscretized:
    """
    A class to store the discretized version of SSY consumption dynamics.

    """

    def __init__(self,
                    ssy,
                    K,
                    I,
                    J,
                    σ_c_states,
                    σ_z_states,
                    z_states,
                    x_states,
                    Q):

        self.ssy = ssy
        self.K, self.I, self.J = K, I, J
        self.σ_c_states = σ_c_states
        self.σ_z_states = σ_z_states
        self.z_states = z_states
        self.x_states = x_states
        self.Q = Q


def split_index(i, M):
    """
    A utility function for the multi-index.
    """
    div = i // M
    rem = i % M
    return (div, rem)

def single_to_multi(m, I, J):
    k, temp = split_index(m, I * J)
    i, j = split_index(temp, J)
    return (k, i, j)

def multi_to_single(k, i , j, I, J):
    return k  * (I * J) + i * J + j


def discretize(ssy, K, I, J):
    """
    And here's the actual discretization process.  

    """

    ρ = ssy.ρ
    ϕ_z = ssy.ϕ_z
    σ_bar = ssy.σ_bar
    ϕ_c = ssy.ϕ_c
    ρ_hz = ssy.ρ_hz
    σ_hz = ssy.σ_hz
    ρ_hc = ssy.ρ_hc
    σ_hc = ssy.σ_hc

    hc_mc = tauchen(ρ_hc, σ_hc, n=K)
    hz_mc = tauchen(ρ_hz, σ_hz, n=I)

    σ_c_states = ϕ_c * σ_bar * np.exp(hc_mc.state_values)
    σ_z_states = ϕ_z * σ_bar * np.exp(hz_mc.state_values) 

    M = I * J * K
    z_states = np.zeros((I, J))
    q = np.zeros((I, J, J))
    Q = np.zeros((M, M))
    x_states = np.zeros((3, M))

    for i, σ_z in enumerate(σ_z_states):
        mc_z = tauchen(ρ, np.sqrt(1 - ρ**2) * σ_z, n=J)
        for j in range(J):
            z_states[i, j] = mc_z.state_values[j]
            for jp in range(J):
                q[i, j, jp] = mc_z.P[j, jp]

    for m in range(M):
        k, i, j = single_to_multi(m, I, J)
        x_states[:, m] = [σ_c_states[k], σ_z_states[i], z_states[i, j]]
        for mp in range(M):
            kp, ip, jp = single_to_multi(mp, I, J)
            Q[m, mp] = hc_mc.P[k, kp] * hz_mc.P[i, ip] * q[i, j, jp]

    # Create an instance of SSYConsumptionDiscretized to store output
    ssyd = SSYConsumptionDiscretized(ssy, 
            K, I, J, σ_c_states, σ_z_states, z_states, x_states, Q)

    return ssyd


def compute_spec_rad(Q):
    """
    Function to compute spectral radius of a matrix.

    """
    return np.max(np.abs(eigvals(Q)))


def compute_K(ssy, K, I, J):
    """
    Compute K in the SSY model.

    """

    ψ = ssy.ψ
    γ = ssy.γ
    β = ssy.β

    θ = (1 - γ) / (1 - 1/ψ)
    μ = ssy.μ_c

    ssyd = discretize(ssy, K, I, J)

    σ_c_states = ssyd.σ_c_states
    σ_z_states = ssyd.σ_z_states
    z_states = ssyd.z_states
    Q = ssyd.Q

    M = I * J * K
    K_matrix = np.empty((M, M))

    for m in range(M):
        for mp in range(M):
            k, i, j = single_to_multi(m, I, J)
            σ_c, σ_z, z = σ_c_states[k], σ_z_states[i], z_states[i, j]
            a = np.exp((1 - γ) * (μ + z) + (1 - γ)**2 * σ_c**2 / 2)
            K_matrix[m, mp] =  a * Q[m, mp]

    return β**θ * K_matrix


def compute_test_val(ssy, K=6, I=6, J=6):
    """
    Compute the test value Lambda

    """

    ψ = ssy.ψ
    γ = ssy.γ
    β = ssy.β

    θ = (1 - γ) / (1 - 1/ψ)

    K_matrix = compute_K(ssy, K, I, J)

    rK = compute_spec_rad(K_matrix)

    return rK**(1/θ)

