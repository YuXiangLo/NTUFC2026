import numpy as np

def monte_carlo_option_pricing(S0, K, r, q, sigma, T, num_simulations, num_repetitions, option_type='call'):
    """
    Prices European options using Monte Carlo simulation.
    
    Inputs:
    S0, K, r, q, sigma, T: Standard option parameters
    num_simulations: Number of paths per repetition
    num_repetitions: Number of times the simulation is repeated
    option_type: 'call' or 'put'
    
    Outputs:
    mean_price: Average price across repetitions
    confidence_interval: (lower, upper) 95% confidence interval
    """
    repetition_results = []
    
    for _ in range(num_repetitions):
        # Generate random normal variables
        Z = np.random.standard_normal(num_simulations)
        
        # Calculate ST
        ST = S0 * np.exp((r - q - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
        
        # Calculate payoffs
        if option_type == 'call':
            payoffs = np.maximum(ST - K, 0)
        else:
            payoffs = np.maximum(K - ST, 0)
            
        # Discount to present value
        discounted_payoffs = np.exp(-r * T) * payoffs
        
        # Store the average of this repetition
        repetition_results.append(np.mean(discounted_payoffs))
        
    mean_price = np.mean(repetition_results)
    std_err = np.std(repetition_results, ddof=1) / np.sqrt(num_repetitions)
    
    # 95% Confidence Interval using normal distribution approximation
    # For better accuracy with small num_repetitions, one could use t-distribution,
    # but 1.96 is standard for large-ish samples.
    confidence_interval = (mean_price - 1.96 * std_err, mean_price + 1.96 * std_err)
    
    return mean_price, confidence_interval
