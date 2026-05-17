import numpy as np
import config
from cholesky import cholesky

def generate_samples(Z_half):
    """Applies Antithetic Variates + 1D Moment Matching."""
    # 1. Antithetic Variates
    Z_anti = np.vstack((Z_half, -Z_half))
    
    # 2. Moment Matching
    Z_std = np.std(Z_anti, axis=0, ddof=1)
    Z_matched = Z_anti / Z_std
    
    # 3. Induce correlation
    L = cholesky(config.rho)
    Z_corr = Z_matched @ L.T
    
    return Z_corr, Z_matched

def price_bonus_1(Z_half):
    """
    Prices using Antithetic Variates + 1D Moment Matching.
    Z_half: Crude random normal matrix of shape (M/2, n)
    """
    # Generate transformed samples
    Z_corr, _ = generate_samples(Z_half)
    
    # Calculate terminal stock prices
    drift = (config.r - config.q - 0.5 * config.sigma**2) * config.T
    diffusion = config.sigma * np.sqrt(config.T) * Z_corr
    ST = config.S0 * np.exp(drift + diffusion)
    
    # Calculate payoff
    max_ST = np.max(ST, axis=1)
    payoffs = np.maximum(max_ST - config.K, 0)
    
    return np.exp(-config.r * config.T) * np.mean(payoffs)
