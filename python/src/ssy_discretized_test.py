"""
Translation of ssy_discretized_test.jl to Python. 

@Stearn 
@jstac

Tue Feb 26 04:38:57 AEDT 2019

"""

from ssy_model import *
import quantecon
import numpy as np
from numpy import linalg as la


class SSYConsumptionDiscretized:
    """
    A class to store the discretized version of SSY consumption dynamics.

    """

    def __init__(self,
                    ssy,
                    K,
                    I,
                    J,
                    sigma_c_states,
                    sigma_z_states,
                    z_states,
                    x_states,
                    Q):

        self.ssy = ssy
        self.K, self.I, self.J = K, I, J
        self.sigma_c_states = sigma_c_states
        self.sigma_z_states = sigma_z_states
        self.z_states = z_states
        self.x_states = x_states
        self.Q = Q

def split_index(i, M):
    div = (i - 1) // M + 1
    rem = (i - 1) % M + 1
    return (div, rem)

def single_to_multi(m, I, J):
    k, temp = split_index(m, I * J)
    i, j = split_index(temp, J)
    return (k, i, j)

def multi_to_single(k, i , j, I, J):
    return (k - 1) * (I * J) + (i - 1) * J + j

def discretize(ssy, K, I, J):

    ρ = ssy.ρ
    ϕ_z = ssy.ϕ_z
    σ_bar = ssy.σ_bar
    ϕ_c = ssy.ϕ_c
    ρ_hz = ssy.ρ_hz
    σ_hz = ssy.σ_hz
    ρ_hc = ssy.ρ_hc
    σ_hc = ssy.σ_hc

    hc_mc = tauchen(ρ_hc, σ_hc, K)
    hz_mc = tauchen(ρ_hz, σ_hz, I)

    σ_c_states = ϕ_c * σ_bar * np.exp(collect(hc_mc.state_values))
    σ_z_states = ϕ_z * σ_bar * np.exp(collect(hz_mc.state_values))

    M = I * J * K
    z_states = np.zeros(I, J)
    q = np.zeros(I, J, J)
    Q = np.zeros(M, M)
    x_states = np.zeros(3, M)

    for (i, σ_z) in enumerate(σ_z_states):
        mc_z = tauchen(J, ρ, sqrt(1 - ρ**2) * σ_z)
        for j in range(J):
            z_states[i, j] = mc_z.state_values[j]
            for jp in range(J):
                q[i, j, jp] = mc_z.p[j, jp]
            end
        end
    end

    for m in range(M):
        k, i, j = single_to_multi(m, I, J)
        x_states[:, m] = [σ_c_states[k], σ_z_states[i], z_states[i, j]]
        for mp in range(M):
            kp, ip, jp = single_to_multi(mp, I, J)
            Q[m, mp] = hc_mc.p[k, kp] * hz_mc.p[i, ip] * q[i, j, jp]
        end
    end

    # Create an instance of SSYConsumptionDiscretized to store output
    ssyd = SSYConsumptionDiscretized(ssy, 
            K, I, J, σ_c_states, σ_z_states, z_states, x_states, Q)

    return ssyd

def easydiscretize(ssy, intvec):
    K, I, J = intvec
    return discretize(ssy, K, I, J)

def compute_spec_rad(Q):
    return max(abs(la.eig(Q)))

def compute_K(ez, ssyd):

    ψ = ez.ψ
    γ = ez.γ
    β = ez.β

    θ = (1 - γ) / (1 - 1/ψ)
    μ = ssyd.ssy.μ_c

    K = ssyd.K
    I = ssyd.I
    J = ssyd.J

    σ_c_states = ssyd.σ_c_states
    σ_z_states = ssyd.σ_z_states
    z_states = ssyd.z_states
    Q = ssyd.Q

    for m in range(M):
        for mp in range(M):
            k, i, j = single_to_multi(m, I, J)
            σ_c, σ_z, z = σ_c_states[k], σ_z_states[i], z_states[i, j]
            a = exp((1 - γ) * (μ + z) + (1 - γ)**2 * σ_c**2 / 2)
            K_matrix[m, mp] =  a * Q[m, mp]
        end
    end

    return β**θ * K_matrix

def compute_test_val(ez, ssy, K, I, J):

    ψ = ez.ψ
    γ = ez.γ
    β = ez.β

    θ = (1 - γ) / (1 - 1/ψ)

    ssyd = discretize(ssy, K, I, J)
    K_matrix = compute_K(ez, ssyd)

    rK = compute_spec_rad(K_matrix)
    return rK**(1/θ)

