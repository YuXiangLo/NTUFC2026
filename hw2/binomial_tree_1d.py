import numpy as np

def crr_binomial_tree_1d(S0, K, r, q, sigma, T, n, option_type='call', exercise_style='european'):
    """
    Prices options using the CRR binomial tree model with a 1D array (column vector).
    This fulfills Bonus 1.
    
    Inputs:
    S0, K, r, q, sigma, T: Standard parameters
    n: Number of time steps
    option_type: 'call' or 'put'
    exercise_style: 'european' or 'american'
    """
    dt = T / n
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)
    
    phi = 1 if option_type == 'call' else -1
    
    # Initialize 1D array for terminal option values
    # V[j] corresponds to j up-moves and (n-j) down-moves at step n
    V = np.zeros(n + 1)
    for j in range(n + 1):
        ST = S0 * (u**j) * (d**(n - j))
        V[j] = max(phi * (ST - K), 0)
        
    # Backward induction
    discount = np.exp(-r * dt)
    for i in range(n - 1, -1, -1):
        # We only update the first i+1 elements
        for j in range(i + 1):
            continuation_value = discount * (p * V[j + 1] + (1 - p) * V[j])
            
            if exercise_style == 'american':
                ST_ij = S0 * (u**j) * (d**(i - j))
                exercise_value = max(phi * (ST_ij - K), 0)
                V[j] = max(continuation_value, exercise_value)
            else:
                V[j] = continuation_value
                
    return V[0]
