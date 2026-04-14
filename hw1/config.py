"""Centralized parameters for Assignment 1 / Bonus 1 pricing."""

# Market inputs
S0 = 100.0
r = 0.03
q = 0.00
sigma = 0.20
T = 1.0

# Strike inputs for the trapezoid payoff
K1 = 90.0
K2 = 100.0
K3 = 110.0
K4 = 120.0

# Monte Carlo inputs for Bonus 2
MC_NUM_SAMPLES = 10_000
MC_NUM_REPETITIONS = 20
MC_RANDOM_SEED = 4242
