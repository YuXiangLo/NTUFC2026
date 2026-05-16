import numpy as np
from scipy import stats
import config
from cholesky import cholesky
from basic_method import price_basic
from bonus1 import price_bonus_1
from bonus2 import price_bonus_2

def calculate_ci(results):
    """Calculates mean and 95% Confidence Interval for a list of estimates."""
    mean_val = np.mean(results)
    std_err = np.std(results, ddof=1) / np.sqrt(config.R)
    ci_half_width = 1.96 * std_err
    return mean_val, (mean_val - ci_half_width, mean_val + ci_half_width), ci_half_width * 2


def _fmt_vector(vec):
    return np.array2string(vec, precision=4, suppress_small=True)


def _fmt_matrix(mat):
    return np.array2string(mat, precision=4, suppress_small=True)


def print_diagnostics(Z_half):
    print("=" * 80)
    print("Diagnostics (using first repetition Z_half)")
    print("=" * 80)

    # Antithetic checks
    Z = Z_half
    Z_anti = np.vstack((Z, -Z))
    print("\n[Antithetic checks]")
    print(f"mean(Z):      {_fmt_vector(np.mean(Z, axis=0))}")
    print(f"mean(Z_anti): {_fmt_vector(np.mean(Z_anti, axis=0))}")
    print(f"std(Z):       {_fmt_vector(np.std(Z, axis=0, ddof=1))}")
    print(f"std(Z_anti):  {_fmt_vector(np.std(Z_anti, axis=0, ddof=1))}")

    # Bonus 1 checks
    Z_std = np.std(Z_anti, axis=0, ddof=1)
    Z_matched = Z_anti / Z_std
    print("\n[Moment matching checks - Bonus 1]")
    print(f"mean(Z_matched): {_fmt_vector(np.mean(Z_matched, axis=0))}")
    print(f"std(Z_matched):  {_fmt_vector(np.std(Z_matched, axis=0, ddof=1))}")

    # Bonus 2 checks
    sample_cov = np.cov(Z_anti, rowvar=False)
    L_hat = cholesky(sample_cov)
    L_hat_inv = np.linalg.inv(L_hat)
    Z_uncorr = Z_anti @ L_hat_inv.T
    L = cholesky(config.rho)
    Z_corr = Z_uncorr @ L.T

    print("\n[Inverse Cholesky checks - Bonus 2]")
    print("cov(Z_anti):")
    print(_fmt_matrix(sample_cov))
    print("cov(Z_uncorr) (should be close to I):")
    print(_fmt_matrix(np.cov(Z_uncorr, rowvar=False)))
    print("cov(Z_corr) (should be close to rho):")
    print(_fmt_matrix(np.cov(Z_corr, rowvar=False)))

    # Normality checks
    print("\n[Normality checks]")
    for label, data in (("Z_uncorr", Z_uncorr), ("Z_corr", Z_corr)):
        mean_vec = np.mean(data, axis=0)
        std_vec = np.std(data, axis=0, ddof=1)
        skew_vec = stats.skew(data, axis=0, bias=False)
        print(f"{label} mean: {_fmt_vector(mean_vec)}")
        print(f"{label} std:  {_fmt_vector(std_vec)}")
        print(f"{label} skew: {_fmt_vector(skew_vec)}")

def main():
    np.random.seed(42) # Set seed for reproducibility
    
    results_basic = []
    results_b1 = []
    results_b2 = []

    print(f"Running {config.R} repetitions of {config.M} simulations each...\n")

    # Print one detailed transformation audit before the Monte Carlo loop.
    Z_diag = np.random.standard_normal((int(config.M / 2), config.n))
    print_diagnostics(Z_diag)
    print()

    for i in range(config.R):
        # Generate Common Random Numbers (CRN) for this repetition
        # We need M/2 samples for antithetic methods, and another M/2 for the basic method
        Z_half_1 = np.random.standard_normal((int(config.M / 2), config.n))
        Z_half_2 = np.random.standard_normal((int(config.M / 2), config.n))
        
        # Basic requirement uses all M crude random numbers
        Z_full_crude = np.vstack((Z_half_1, Z_half_2))
        
        # Execute pricing functions
        val_basic = price_basic(Z_full_crude)
        val_b1 = price_bonus_1(Z_half_1)
        val_b2 = price_bonus_2(Z_half_1)
        
        results_basic.append(val_basic)
        results_b1.append(val_b1)
        results_b2.append(val_b2)

            # Calculate statistics
    mean_basic, ci_basic, width_basic = calculate_ci(results_basic)
    mean_b1, ci_b1, width_b1 = calculate_ci(results_b1)
    mean_b2, ci_b2, width_b2 = calculate_ci(results_b2)

    # Print Results
    print("-" * 65)
    print(f"{'Method':<20} | {'Option Value':<12} | {'95% CI':<20} | {'CI Width'}")
    print("-" * 65)
    print(f"{'Basic Requirement':<20} | {mean_basic:>12.6f} | [{ci_basic[0]:>8.4f}, {ci_basic[1]:>8.4f}] | {width_basic:.6f}")
    print(f"{'Bonus 1 (Moment)':<20} | {mean_b1:>12.6f} | [{ci_b1[0]:>8.4f}, {ci_b1[1]:>8.4f}] | {width_b1:.6f}")
    print(f"{'Bonus 2 (Inv Chol)':<20} | {mean_b2:>12.6f} | [{ci_b2[0]:>8.4f}, {ci_b2[1]:>8.4f}] | {width_b2:.6f}")
    print("-" * 65)

    # Verify the CI width constraint (Basic > Bonus 1 > Bonus 2)
    print("\nVerifying CI Width Constraint:")
    if width_basic > width_b1 > width_b2:
        print("SUCCESS: Basic Width > Bonus 1 Width > Bonus 2 Width")
    else:
        print("FAILED: Variance reduction ordering not strictly observed in this run.")

if __name__ == "__main__":
    main()
