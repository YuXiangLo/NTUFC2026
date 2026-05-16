import numpy as np

# Option Parameters
K = 100.0             # Strike price
r = 0.05              # Risk-free interest rate
T = 1.0               # Time to maturity (in years)

# Simulation Parameters
M = 10000             # Number of simulations (must be even for antithetic)
R = 20                # Number of repetitions

# Asset Parameters (n = 3 assets)
n = 3
S0 = np.array([100.0, 100.0, 100.0])       # Initial stock prices: S_{10}, S_{20}, ..., S_{n0}
q = np.array([0.02, 0.02, 0.02])           # Dividend yields: q_1, q_2, ..., q_n
sigma = np.array([0.2, 0.25, 0.3])         # Volatilities: \sigma_1, \sigma_2, ..., \sigma_n

# Correlation Matrix (\rho_{ij})
rho = np.array([
    [1.0, 0.5, 0.5],
    [0.5, 1.0, 0.5],
    [0.5, 0.5, 1.0]
])

# Ensure M is even for the antithetic variate approach
assert M % 2 == 0, "Number of simulations (M) must be an even number."
