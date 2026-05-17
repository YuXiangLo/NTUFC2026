import numpy as np
import config
from cholesky import cholesky

def generate_samples(Z):
    """Induces correlation in crude random normals."""
    L = cholesky(config.rho)
    return Z @ L.T

def price_basic(Z):
    """
    Prices the maximum rainbow option using standard Cholesky decomposition.
    Z: Crude random normal matrix of shape (M, n)
    """
    # Induce correlation
    Z_corr = generate_samples(Z)
    
    # Calculate terminal stock prices (Geometric Brownian Motion)
    drift = (config.r - config.q - 0.5 * config.sigma**2) * config.T
    diffusion = config.sigma * np.sqrt(config.T) * Z_corr
    ST = config.S0 * np.exp(drift + diffusion)
    
    # Calculate payoff: max(max(S_1T, ..., S_nT) - K, 0)
    max_ST = np.max(ST, axis=1)
    payoffs = np.maximum(max_ST - config.K, 0)
    
    # Discount back to present value
    return np.exp(-config.r * config.T) * np.mean(payoffs)
