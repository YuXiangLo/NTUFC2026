import numpy as np
import math
from scipy.stats import binom
from scipy.special import gammaln

def log_factorial_manual(n):
    """Computes ln(n!) by summing logs manually."""
    if n < 2:
        return 0.0
    return sum(math.log(i) for i in range(1, n + 1))

def binom_pmf_manual(n, j, p):
    """Computes binom pmf using manual log-sum of factorials."""
    # To be efficient, we calculate ln(n!) once and reuse it for all j
    ln_n_fact = log_factorial_manual(n)
    
    # Pre-calculate log_factorial for all values up to n to avoid O(n^2) complexity
    # We use a prefix sum approach for efficiency
    log_facts = [0.0] * (n + 1)
    current_sum = 0.0
    for i in range(1, n + 1):
        current_sum += math.log(i)
        log_facts[i] = current_sum
        
    log_p = math.log(p)
    log_1p = math.log(1 - p)
    
    probs = []
    for k in j:
        ln_prob = log_facts[n] - log_facts[k] - log_facts[n - k] + k * log_p + (n - k) * log_1p
        probs.append(math.exp(ln_prob))
    return np.array(probs)

def binom_pmf_gammaln(n, j, p):
    """Computes binom pmf using scipy.special.gammaln."""
    ln_prob = (gammaln(n + 1) - 
               gammaln(j + 1) - 
               gammaln(n - j + 1) + 
               j * np.log(p) + 
               (n - j) * np.log(1 - p))
    return np.exp(ln_prob)

def combinatorial_price(S0, K, r, q, sigma, T, n, method='package'):
    dt = T / n
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)
    
    j = np.arange(n + 1)
    
    if method == 'package':
        probabilities = binom.pmf(j, n, p)
    elif method == 'gammaln':
        probabilities = binom_pmf_gammaln(n, j, p)
    elif method == 'manual':
        probabilities = binom_pmf_manual(n, j, p)
    else:
        raise ValueError("Unknown method")
        
    ST = S0 * (u**j) * (d**(n - j))
    payoffs = np.maximum(ST - K, 0) # Call option
    option_value = np.exp(-r * T) * np.sum(probabilities * payoffs)
    
    return option_value, probabilities

def run_comparison():
    S0, K, r, q, sigma, T = 100, 100, 0.05, 0.02, 0.3, 0.5
    n_list = [100, 1000, 5000, 10000, 50000]
    
    print(f"{'n':>6} | {'Method':>10} | {'Option Price':>15} | {'Price Diff vs Pkg':>18} | {'Max Prob Diff vs Pkg'}")
    print("-" * 85)
    
    for n in n_list:
        p_pkg, probs_pkg = combinatorial_price(S0, K, r, q, sigma, T, n, 'package')
        print(f"{n:6} | {'package':10} | {p_pkg:15.10f} | {'0.0000000000e+00':>18} | {'0.0000000000e+00'}")
        
        for method in ['gammaln', 'manual']:
            p_val, probs_val = combinatorial_price(S0, K, r, q, sigma, T, n, method)
            price_diff = p_val - p_pkg
            max_prob_diff = np.max(np.abs(probs_val - probs_pkg))
            
            print(f"{n:6} | {method:10} | {p_val:15.10f} | {price_diff:18.10e} | {max_prob_diff:10.10e}")
        print("-" * 85)

if __name__ == "__main__":
    run_comparison()
