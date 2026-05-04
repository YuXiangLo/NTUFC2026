import numpy as np
from scipy.stats import norm

def black_scholes_call(S0, K, r, q, sigma, T):
    """
    Calculates the Black-Scholes price for a European call option.
    
    Inputs:
    S0: Current stock price
    K: Strike price
    r: Risk-free interest rate
    q: Continuous dividend yield
    sigma: Volatility
    T: Time to maturity
    """
    if T <= 0:
        return max(S0 - K, 0.0)
    
    d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    call_price = S0 * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

def black_scholes_put(S0, K, r, q, sigma, T):
    """
    Calculates the Black-Scholes price for a European put option.
    
    Inputs:
    S0: Current stock price
    K: Strike price
    r: Risk-free interest rate
    q: Continuous dividend yield
    sigma: Volatility
    T: Time to maturity
    """
    if T <= 0:
        return max(K - S0, 0.0)
    
    d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S0 * np.exp(-q * T) * norm.cdf(-d1)
    return put_price
