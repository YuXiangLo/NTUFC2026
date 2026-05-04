import numpy as np
from black_scholes import black_scholes_call, black_scholes_put

def bbs_tree(S0, K, r, q, sigma, T, n, option_type='call', exercise_style='european'):
    """
    Prices options using the Binomial Black-Scholes (BBS) model.
    This fulfills Bonus 2.
    
    Inputs:
    S0, K, r, q, sigma, T: Standard parameters
    n: Number of time steps
    option_type: 'call' or 'put'
    exercise_style: 'european' or 'american'
    """
    if n < 1:
        # Fallback to pure Black-Scholes if n=0 or invalid
        if option_type == 'call':
            return black_scholes_call(S0, K, r, q, sigma, T)
        else:
            return black_scholes_put(S0, K, r, q, sigma, T)

    dt = T / n
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)
    
    phi = 1 if option_type == 'call' else -1
    
    # Step n-1: Use Black-Scholes values
    V = np.zeros(n) # Penultimate step has n nodes (0 to n-1)
    for j in range(n):
        S_penultimate = S0 * (u**j) * (d**(n - 1 - j))
        
        if option_type == 'call':
            bs_val = black_scholes_call(S_penultimate, K, r, q, sigma, dt)
        else:
            bs_val = black_scholes_put(S_penultimate, K, r, q, sigma, dt)
            
        if exercise_style == 'american':
            intrinsic = max(phi * (S_penultimate - K), 0)
            V[j] = max(bs_val, intrinsic)
        else:
            V[j] = bs_val
            
    # Backward induction from step n-2 down to 0
    discount = np.exp(-r * dt)
    for i in range(n - 2, -1, -1):
        for j in range(i + 1):
            continuation_value = discount * (p * V[j + 1] + (1 - p) * V[j])
            
            if exercise_style == 'american':
                ST_ij = S0 * (u**j) * (d**(i - j))
                exercise_value = max(phi * (ST_ij - K), 0)
                V[j] = max(continuation_value, exercise_value)
            else:
                V[j] = continuation_value
                
    return V[0]
