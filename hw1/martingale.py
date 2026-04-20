"""
Prices the trapezoid payoff option using the unreduced, raw Martingale Pricing formula.
Maps directly to the three indicator segments [K1, K2], [K2, K3], and [K3, K4].
"""

import math
from statistics import NormalDist
import config

def N(x: float) -> float:
    """Standard Normal cumulative distribution function."""
    return NormalDist(0.0, 1.0).cdf(x)

def calc_d1(S0: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    """Calculates the d1 parameter."""
    return (math.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))

def calc_d2(d1_val: float, T: float, sigma: float) -> float:
    """Calculates the d2 parameter based on d1."""
    return d1_val - sigma * math.sqrt(T)

def main():
    # 1. Load parameters from config
    S0 = config.S0
    r = config.r
    q = config.q
    sigma = config.sigma
    T = config.T
    
    K1 = config.K1
    K2 = config.K2
    K3 = config.K3
    K4 = config.K4

    # 2. Calculate the slope ratio (m)
    m = (K2 - K1) / (K4 - K3)

    # 3. Pre-calculate all d1 and d2 values for clean integration into the formula
    d1_K1 = calc_d1(S0, K1, T, r, q, sigma)
    d2_K1 = calc_d2(d1_K1, T, sigma)
    
    d1_K2 = calc_d1(S0, K2, T, r, q, sigma)
    d2_K2 = calc_d2(d1_K2, T, sigma)
    
    d1_K3 = calc_d1(S0, K3, T, r, q, sigma)
    d2_K3 = calc_d2(d1_K3, T, sigma)
    
    d1_K4 = calc_d1(S0, K4, T, r, q, sigma)
    d2_K4 = calc_d2(d1_K4, T, sigma)

    # 4. Calculate Segment 1: [K1, K2] (Rising Slope)
    # Formula: S0*e^(-qT)*[N(d1(K1)) - N(d1(K2))] - K1*e^(-rT)*[N(d2(K1)) - N(d2(K2))]
    part1_asset = S0 * math.exp(-q * T) * (N(d1_K1) - N(d1_K2))
    part1_cash  = K1 * math.exp(-r * T) * (N(d2_K1) - N(d2_K2))
    segment_1   = part1_asset - part1_cash

    # 5. Calculate Segment 2: [K2, K3] (Flat Top)
    # Formula: (K2 - K1)*e^(-rT)*[N(d2(K2)) - N(d2(K3))]
    segment_2 = (K2 - K1) * math.exp(-r * T) * (N(d2_K2) - N(d2_K3))

    # 6. Calculate Segment 3: [K3, K4] (Falling Slope)
    # Formula: m*K4*e^(-rT)*[N(d2(K3)) - N(d2(K4))] - m*S0*e^(-qT)*[N(d1(K3)) - N(d1(K4))]
    part3_cash  = m * K4 * math.exp(-r * T) * (N(d2_K3) - N(d2_K4))
    part3_asset = m * S0 * math.exp(-q * T) * (N(d1_K3) - N(d1_K4))
    segment_3   = part3_cash - part3_asset

    # 7. Final Summation
    V0 = segment_1 + segment_2 + segment_3

    # Output results
    # print("--- Raw Martingale Pricing Execution ---")
    # print(f"Segment [K1, K2] Value: {segment_1:.6f}")
    # print(f"Segment [K2, K3] Value: {segment_2:.6f}")
    # print(f"Segment [K3, K4] Value: {segment_3:.6f}")
    # print("-" * 38)
    print('-' * 50)
    print(f"Martingale\t= {V0}")
    print('-' * 50)

if __name__ == "__main__":
    main()
