import numpy as np
import config
from cholesky import cholesky

def price_bonus_1(Z_half):
    """
    Prices using Antithetic Variates + 1D Moment Matching.
    Z_half: Crude random normal matrix of shape (M/2, n)
    """
    # 1. Antithetic Variates: Append negative of the samples
    Z_anti = np.vstack((Z_half, -Z_half))
    
    # 2. Moment Matching (1D standard normalization)
    # Mean is strictly 0 due to antithetic variates, so we only need to adjust standard deviation
    Z_std = np.std(Z_anti, axis=0, ddof=1)
    Z_matched = Z_anti / Z_std
    
    # Cholesky decomposition of target correlation matrix
    L = cholesky(config.rho)
    
    # Induce correlation
    Z_corr = Z_matched @ L.T
    
    # Calculate terminal stock prices
    drift = (config.r - config.q - 0.5 * config.sigma**2) * config.T
    diffusion = config.sigma * np.sqrt(config.T) * Z_corr
    ST = config.S0 * np.exp(drift + diffusion)
    
    # Calculate payoff
    max_ST = np.max(ST, axis=1)
    payoffs = np.maximum(max_ST - config.K, 0)
    
    return np.exp(-config.r * config.T) * np.mean(payoffs)
