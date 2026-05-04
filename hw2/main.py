from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from black_scholes import black_scholes_call, black_scholes_put
from monte_carlo import monte_carlo_option_pricing
from binomial_tree_2d import crr_binomial_tree_2d
from binomial_tree_1d import crr_binomial_tree_1d
from bbs_tree import bbs_tree
from combinatorial import combinatorial_method

def run_assignment():
    # --- Input Parameters ---
    S0 = 100.0
    K = 100.0
    r = 0.05
    q = 0.02
    sigma = 0.3
    T = 0.5
    n = 105
    num_simulations = 10000
    num_repetitions = 20

    print("="*50)
    print("Assignment 2: Option Pricing Methods")
    print(f"Parameters: S0={S0}, K={K}, r={r}, q={q}, sigma={sigma}, T={T}, n={n}")
    print(f"MC: {num_simulations} sims, {num_repetitions} reps")
    print("="*50)

    # --- Call Options ---
    print("\n[EUROPEAN CALL]")
    bs_call = black_scholes_call(S0, K, r, q, sigma, T)
    mc_call_val, mc_call_ci = monte_carlo_option_pricing(S0, K, r, q, sigma, T, num_simulations, num_repetitions, 'call')
    crr_2d_call = crr_binomial_tree_2d(S0, K, r, q, sigma, T, n, 'call', 'european')
    crr_1d_call = crr_binomial_tree_1d(S0, K, r, q, sigma, T, n, 'call', 'european')
    bbs_call_val = bbs_tree(S0, K, r, q, sigma, T, n, 'call', 'european')
    comb_call = combinatorial_method(S0, K, r, q, sigma, T, n, 'call')

    print(f"{'Black-Scholes:':<25} {bs_call:.6f}")
    print(f"{'Monte Carlo:':<25} {mc_call_val:.6f} (95% CI: [{mc_call_ci[0]:.6f}, {mc_call_ci[1]:.6f}])")
    print(f"{'CRR Binomial (2D):':<25} {crr_2d_call:.6f}")
    print(f"{'CRR Binomial (1D):':<25} {crr_1d_call:.6f}")
    print(f"{'BBS Tree:':<25} {bbs_call_val:.6f}")
    print(f"{'Combinatorial:':<25} {comb_call:.6f}")

    print("\n[AMERICAN CALL]")
    crr_2d_call_am = crr_binomial_tree_2d(S0, K, r, q, sigma, T, n, 'call', 'american')
    crr_1d_call_am = crr_binomial_tree_1d(S0, K, r, q, sigma, T, n, 'call', 'american')
    bbs_call_am = bbs_tree(S0, K, r, q, sigma, T, n, 'call', 'american')
    print(f"{'CRR Binomial (2D):':<25} {crr_2d_call_am:.6f}")
    print(f"{'CRR Binomial (1D):':<25} {crr_1d_call_am:.6f}")
    print(f"{'BBS Tree:':<25} {bbs_call_am:.6f}")

    # --- Put Options ---
    print("\n" + "="*50)
    print("\n[EUROPEAN PUT]")
    bs_put = black_scholes_put(S0, K, r, q, sigma, T)
    mc_put_val, mc_put_ci = monte_carlo_option_pricing(S0, K, r, q, sigma, T, num_simulations, num_repetitions, 'put')
    crr_2d_put = crr_binomial_tree_2d(S0, K, r, q, sigma, T, n, 'put', 'european')
    crr_1d_put = crr_binomial_tree_1d(S0, K, r, q, sigma, T, n, 'put', 'european')
    bbs_put_val = bbs_tree(S0, K, r, q, sigma, T, n, 'put', 'european')
    comb_put = combinatorial_method(S0, K, r, q, sigma, T, n, 'put')

    print(f"{'Black-Scholes:':<25} {bs_put:.6f}")
    print(f"{'Monte Carlo:':<25} {mc_put_val:.6f} (95% CI: [{mc_put_ci[0]:.6f}, {mc_put_ci[1]:.6f}])")
    print(f"{'CRR Binomial (2D):':<25} {crr_2d_put:.6f}")
    print(f"{'CRR Binomial (1D):':<25} {crr_1d_put:.6f}")
    print(f"{'BBS Tree:':<25} {bbs_put_val:.6f}")
    print(f"{'Combinatorial:':<25} {comb_put:.6f}")

    print("\n[AMERICAN PUT]")
    crr_2d_put_am = crr_binomial_tree_2d(S0, K, r, q, sigma, T, n, 'put', 'american')
    crr_1d_put_am = crr_binomial_tree_1d(S0, K, r, q, sigma, T, n, 'put', 'american')
    bbs_put_am = bbs_tree(S0, K, r, q, sigma, T, n, 'put', 'american')
    print(f"{'CRR Binomial (2D):':<25} {crr_2d_put_am:.6f}")
    print(f"{'CRR Binomial (1D):':<25} {crr_1d_put_am:.6f}")
    print(f"{'BBS Tree:':<25} {bbs_put_am:.6f}")

    # --- Convergence Plot (Bonus 2) ---
    print("\n" + "="*50)
    print("Generating Convergence Plot (CRR vs BBS)...")
    n_values = range(105, 501, 5)
    crr_prices = []
    bbs_prices = []

    # Using American Put for demonstration as it's common for these plots
    for n_val in tqdm(n_values):
        crr_prices.append(crr_binomial_tree_1d(S0, K, r, q, sigma, T, n_val, 'put', 'american'))
        bbs_prices.append(bbs_tree(S0, K, r, q, sigma, T, n_val, 'put', 'american'))

    plt.figure(figsize=(10, 6))
    plt.plot(n_values, crr_prices, label='CRR (1D)', marker='.', linestyle='-', alpha=0.7)
    plt.plot(n_values, bbs_prices, label='BBS', marker='s', linestyle='--', alpha=0.7)
    plt.xlabel('Number of Time Steps (n)')
    plt.ylabel('American Put Price')
    plt.legend()
    plt.grid(True)
    plt.savefig('convergence_plot.png')
    print("Plot saved as 'convergence_plot.png'.")
    print("="*50)

if __name__ == "__main__":
    run_assignment()
