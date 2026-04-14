"""Bonus 1: price the trapezoid payoff via plain-vanilla option replication.

Replicating portfolio (calls only):
    +1 * C(K1) - 1 * C(K2) - a * C(K3) + a * C(K4)
where
    a = (K2 - K1) / (K4 - K3)

This yields a payoff that:
- increases linearly from K1 to K2,
- is flat at height (K2 - K1) between K2 and K3,
- decreases linearly to 0 at K4.
"""

from math import erf, exp, log, sqrt

import config


def norm_cdf(x: float) -> float:
    """Standard normal CDF using erf."""
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def bs_call_price(S0: float, K: float, r: float, q: float, sigma: float, T: float) -> float:
    """Black-Scholes price of a dividend-paying European call."""
    if T <= 0.0:
        return max(S0 - K, 0.0)
    if sigma <= 0.0:
        fwd = S0 * exp((r - q) * T)
        return exp(-r * T) * max(fwd - K, 0.0)

    d1 = (log(S0 / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return S0 * exp(-q * T) * norm_cdf(d1) - K * exp(-r * T) * norm_cdf(d2)


def bs_put_price(S0: float, K: float, r: float, q: float, sigma: float, T: float) -> float:
    """Black-Scholes price of a dividend-paying European put."""
    call = bs_call_price(S0, K, r, q, sigma, T)
    return call - S0 * exp(-q * T) + K * exp(-r * T)


def replication_weight(K1: float, K2: float, K3: float, K4: float) -> float:
    """Weight on the K3/K4 call spread to match plateau height at K4."""
    width_right = K4 - K3
    if width_right <= 0.0:
        raise ValueError("K4 must be greater than K3.")
    return (K2 - K1) / width_right


def price_bonus1(
    S0: float,
    r: float,
    q: float,
    sigma: float,
    T: float,
    K1: float,
    K2: float,
    K3: float,
    K4: float,
) -> dict:
    """Return component values and total option value for Bonus 1."""
    a = replication_weight(K1, K2, K3, K4)

    c1 = bs_call_price(S0, K1, r, q, sigma, T)
    c2 = bs_call_price(S0, K2, r, q, sigma, T)
    c3 = bs_call_price(S0, K3, r, q, sigma, T)
    c4 = bs_call_price(S0, K4, r, q, sigma, T)

    total = c1 - c2 - a * c3 + a * c4

    return {
        "a": a,
        "C(K1)": c1,
        "C(K2)": c2,
        "C(K3)": c3,
        "C(K4)": c4,
        "value": total,
    }


def payoff_terminal(ST: float, K1: float, K2: float, K3: float, K4: float) -> float:
    """Terminal payoff from the replicating portfolio."""
    a = replication_weight(K1, K2, K3, K4)
    return (
        max(ST - K1, 0.0)
        - max(ST - K2, 0.0)
        - a * max(ST - K3, 0.0)
        + a * max(ST - K4, 0.0)
    )


def main() -> None:
    result = price_bonus1(
        S0=config.S0,
        r=config.r,
        q=config.q,
        sigma=config.sigma,
        T=config.T,
        K1=config.K1,
        K2=config.K2,
        K3=config.K3,
        K4=config.K4,
    )

    print("Bonus 1 replication with plain-vanilla calls")
    print(f"a = (K2-K1)/(K4-K3) = {result['a']:.6f}")
    print(f"C(K1) = {result['C(K1)']:.6f}")
    print(f"C(K2) = {result['C(K2)']:.6f}")
    print(f"C(K3) = {result['C(K3)']:.6f}")
    print(f"C(K4) = {result['C(K4)']:.6f}")
    print(f"Option value = {result['value']:.6f}")


if __name__ == "__main__":
    main()
