import numpy as np
import config

def price_basic(Z):
    """
    Prices the maximum rainbow option using standard Cholesky decomposition.
    Z: Crude random normal matrix of shape (M, n)
    """
    # Cholesky decomposition of target correlation matrix
    L = np.linalg.cholesky(config.rho)
    
    # Induce correlation
    Z_corr = Z @ L.T
    
    # Calculate terminal stock prices (Geometric Brownian Motion)
    drift = (config.r - config.q - 0.5 * config.sigma**2) * config.T
    diffusion = config.sigma * np.sqrt(config.T) * Z_corr
    ST = config.S0 * np.exp(drift + diffusion)
    
    # Calculate payoff: max(max(S_1T, ..., S_nT) - K, 0)
    max_ST = np.max(ST, axis=1)
    payoffs = np.maximum(max_ST - config.K, 0)
    
    # Discount back to present value
    return np.exp(-config.r * config.T) * np.mean(payoffs)
