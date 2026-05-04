import numpy as np
from scipy.stats import binom

def combinatorial_method(S0, K, r, q, sigma, T, n, option_type='call'):
    """
    Prices European options using the combinatorial method.
    This fulfills Bonus 3.
    
    Inputs:
    S0, K, r, q, sigma, T: Standard parameters
    n: Number of time steps
    option_type: 'call' or 'put'
    """
    dt = T / n
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)
    
    phi = 1 if option_type == 'call' else -1
    
    # Calculate probabilities for all possible outcomes at step n
    j = np.arange(n + 1)
    probabilities = binom.pmf(j, n, p)
    
    # Calculate terminal stock prices
    ST = S0 * (u**j) * (d**(n - j))
    
    # Calculate payoffs
    payoffs = np.maximum(phi * (ST - K), 0)
    
    # Discounted expected payoff
    option_value = np.exp(-r * T) * np.sum(probabilities * payoffs)
    
    return option_value
