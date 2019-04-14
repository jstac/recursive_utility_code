"""

Implements BY preferences and the Bansal Yaron consumption process 

    g = μ_c + z + σ η                   # consumption growth, μ_c is μ

    z' = ρ z + ϕ_z σ e'                 # z, ϕ_z here is x, ϕ_e in BY

    (σ^2)' = v σ^2 + d + ϕ_σ w'         # v, d, ϕ_σ is v_1, σ^2(1-v_1), σ_w

where {e} and {w} are IID and N(0, 1). 

The state is (z, σ)

See table IV on page 1489 for parameter values.


"""

import numpy as np


class BY:
    
    def __init__(self, 
                β=0.998, 
                γ=10.0, 
                ψ=1.5,
                μ_c=0.0015,
                ρ=0.979,
                ϕ_z=0.044,
                v=0.987,
                d=7.9092e-7,
                ϕ_σ=2.3e-6):


        self.β, self.γ, self.ψ = β, γ, ψ
        self.μ_c, self.ρ, self.ϕ_z, self.v, self.d, self.ϕ_σ = \
                μ_c, ρ, ϕ_z, v, d, ϕ_σ




