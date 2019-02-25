"""

The Schorfheide, Song and Yaron model consumption process. 
    
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

"""


import numpy as np


class SSY:

    def __init__(self, 
                β=0.999, 
                γ=8.89, 
                ψ=1.97,
                μ_c=0.0016,
                ρ=0.987,
                ϕ_z=0.215,
                σ_bar=0.0032,
                ϕ_c=1.0,
                ρ_hz=0.992,
                σ_hz=np.sqrt(0.0039),
                ρ_hc=0.991,
                σ_hc=np.sqrt(0.0096)):

        self.β, self.γ, self.ψ = β, γ, ψ
        self.μ_c, self.ρ, self.ϕ_z, self.σ_bar, self.ϕ_c = μ_c, ρ, ϕ_z, σ_bar, ϕ_c
        self.ρ_hz, self.σ_hz, self.ρ_hc, self.σ_hc = ρ_hz, σ_hz, ρ_hc, σ_hc 




