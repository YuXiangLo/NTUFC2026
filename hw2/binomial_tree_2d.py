import numpy as np

def crr_binomial_tree_2d(S0, K, r, q, sigma, T, n, option_type='call', exercise_style='european'):
    """
    Prices options using the CRR binomial tree model with a 2D array.
    
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
    
    # Initialize trees
    stock_tree = np.zeros((n + 1, n + 1))
    option_tree = np.zeros((n + 1, n + 1))
    
    # Build stock price tree
    for i in range(n + 1):
        for j in range(i + 1):
            stock_tree[i, j] = S0 * (u**j) * (d**(i - j))
            
    # Initialize terminal option values
    for j in range(n + 1):
        option_tree[n, j] = max(phi * (stock_tree[n, j] - K), 0)
        
    # Backward induction
    discount = np.exp(-r * dt)
    for i in range(n - 1, -1, -1):
        for j in range(i + 1):
            continuation_value = discount * (p * option_tree[i + 1, j + 1] + (1 - p) * option_tree[i + 1, j])
            
            if exercise_style == 'american':
                exercise_value = max(phi * (stock_tree[i, j] - K), 0)
                option_tree[i, j] = max(continuation_value, exercise_value)
            else:
                option_tree[i, j] = continuation_value
                
    return option_tree[0, 0]
