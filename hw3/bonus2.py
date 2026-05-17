import numpy as np
import config
from cholesky import cholesky

def generate_samples(Z_half):
    """Applies Antithetic Variates + Inverse Cholesky method (Wang 2008)."""
    # 1. Antithetic Variates
    Z_anti = np.vstack((Z_half, -Z_half))
    
    # 2. Inverse Cholesky
    sample_cov = np.cov(Z_anti, rowvar=False)
    L_hat = cholesky(sample_cov)
    L_hat_inv = np.linalg.inv(L_hat)
    
    # Orthogonalize the samples
    Z_uncorr = Z_anti @ L_hat_inv.T
    
    # 3. Induce TARGET correlation
    L = cholesky(config.rho)
    Z_corr = Z_uncorr @ L.T
    
    return Z_corr, Z_uncorr, sample_cov

def price_bonus_2(Z_half):
    """
    Prices using Antithetic Variates + Inverse Cholesky method (Wang 2008).
    Z_half: Crude random normal matrix of shape (M/2, n)
    """
    # Generate transformed samples
    Z_corr, _, _ = generate_samples(Z_half)
    
    # Calculate terminal stock prices
    drift = (config.r - config.q - 0.5 * config.sigma**2) * config.T
    diffusion = config.sigma * np.sqrt(config.T) * Z_corr
    ST = config.S0 * np.exp(drift + diffusion)
    
    # Calculate payoff
    max_ST = np.max(ST, axis=1)
    payoffs = np.maximum(max_ST - config.K, 0)
    
    return np.exp(-config.r * config.T) * np.mean(payoffs)
