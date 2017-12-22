"""
Representations of EZ utility and Bansal--Yaron and Schorfheide, Song & Yaron
consumption processes

"""


import numpy as np
from numpy.random import randn


# EZ parameters 

class BYPrefs:
    
    def __init__(self, β=0.998, γ=10.0, ψ=1.5):

        self.β, self.γ, self.ψ = β, γ, ψ


class SSYPrefs:
    
    def __init__(self, β=0.999, γ=8.89, ψ=1.97):

        self.β, self.γ, self.ψ = β, γ, ψ




# BY and SSY consumption process parameters

class BYConsumption:

    def __init__(self, 
                μ_c=0.0015,
                ρ=0.979,
                ϕ_z=0.044,
                v=0.987,
                d=7.9092e-7,
                ϕ_σ=2.3e-6):


        self.μ_c, self.ρ, self.ϕ_z, self.v, self.d, self.ϕ_σ = \
                μ_c, ρ, ϕ_z, v, d, ϕ_σ


    """

    Simulate the Bansal Yaron consumption process 

        z' = ρ z + ϕ_z σ e'                 # z, ϕ_z here is x, ϕ_e in BY

        g = μ_c + z + σ η                   # consumption growth, μ_c is μ

        (σ^2)' = v σ^2 + d + ϕ_σ w'         # v, d, ϕ_σ is v_1, σ^2(1-v_1), σ_w

    where {e} and {w} are IID and N(0, 1). 

    See table IV on page 1489 for parameter values.

    Returns gc[2], ..., X[ts_length] where

        gc[t] = ln(C[t]) - ln(C[t-1])

    """
    def simulate(self, ts_length=1000000, seed=1234):

        np.random.seed(seed)

        # Unpack
        μ_c, ρ, ϕ_z, v, d, ϕ_σ = \
            self.μ_c, self.ρ, self.ϕ_z, self.v, self.d, self.ϕ_σ 

        # Allocate memory
        c_growth = np.zeros(ts_length)

        z = 0.0
        σ = d / (1 - v)

        for t in range(ts_length-1):
            # Evaluate consumption and dividends
            c_growth[t] = μ_c + z + σ * randn()

            # Update state
            σ2 = v * σ**2 + d + ϕ_σ * randn()
            σ = np.sqrt(max(σ2, 0))
            z = ρ * z + ϕ_z * σ * randn()

        return c_growth[2:]





class SSYConsumption:

    def __init__(self,
                 μ_c=0.0016,
                 ρ=0.987,
                 ϕ_z=0.215,
                 σ_bar=0.0032,
                 ϕ_c=1.0,
                 ρ_hz=0.992,
                 σ_hz=np.sqrt(0.0039),
                 ρ_hc=0.991,
                 σ_hc=np.sqrt(0.0096)):  

        self.μ_c, self.ρ, self.ϕ_z = μ_c, ρ, ϕ_z
        self.σ_bar, self.ϕ_c, self.ρ_hz = σ_bar, ϕ_c, ρ_hz
        self.σ_hz, self.ρ_hc, self.σ_hc  = σ_hz, ρ_hc, σ_hc  


    """

    Simulate the Schorfheide, Song and Yaron model consumption process. 
        
    Log consumption growth g_c is given by

        g_c = μ_c + z + σ_c η'

        z' = ρ z + sqrt(1 - ρ^2) σ_z e'

        σ_z = ϕ_z σ_bar exp(h_z)

        σ_c = ϕ_c σ_bar exp(h_c)

        h_z' = ρ_hz h_z + σ_hz u'

        h_c' = ρ_hc h_c + σ_hc w'

    Here {e}, {u} and {w} are IID and N(0, 1).  

    Default consumption process values from May 2017 version of Schorfheide, Song
    and Yaron.  See p. 28.

    Returns gc[2], ..., gc[ts_length] where

        gc[t] = ln(C[t]) - ln(C[t-1])


    """
    def simulate(self, ts_length=1000000, seed=1234):

        np.random.seed(seed)

        # Unpack
        μ_c, ρ, ϕ_z = self.μ_c, self.ρ, self.ϕ_z 
        σ_bar, ϕ_c, ρ_hz = self.σ_bar, self.ϕ_c, self.ρ_hz 
        σ_hz, ρ_hc, σ_hc= self.σ_hz, self.ρ_hc, self.σ_hc  

        # Initial conditions
        z, h_z, h_c = 0.0, 0.0, 0.0

        # Allocate memory consumption 
        c_growth = np.zeros(ts_length)

        # Simulate all stochastic processes 
        for t in range(ts_length-1):
            # Simplify names
            σ_z = ϕ_z * σ_bar * np.exp(h_z)
            σ_c = ϕ_c * σ_bar * np.exp(h_c)
            
            # Evaluate consumption and dividends
            c_growth[t] = μ_c + z + σ_c * randn()

            # Update states
            z = ρ * z + np.sqrt(1 - ρ**2) * σ_z * randn()
            h_z = ρ_hz * h_z + σ_hz * randn()
            h_c = ρ_hc * h_c + σ_hc * randn()

        return c_growth[2:]





