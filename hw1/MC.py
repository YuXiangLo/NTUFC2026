"""Bonus 2: Monte Carlo pricing for the trapezoid payoff option.

The assignment asks for:
- 10,000 simulated terminal prices per repetition,
- 20 repetitions,
- 95% CI as mean +/- 2 * (std of 20 repetition prices),
  without dividing by sqrt(20) or sqrt(10000).
"""

from math import exp, log, sqrt
from random import Random
from statistics import mean, stdev

import config


def replication_weight(K1: float, K2: float, K3: float, K4: float) -> float:
    """Weight on the K3/K4 call spread to match plateau height."""
    width_right = K4 - K3
    if width_right <= 0.0:
        raise ValueError("K4 must be greater than K3.")
    return (K2 - K1) / width_right


def payoff_terminal(ST: float, K1: float, K2: float, K3: float, K4: float) -> float:
    """Terminal payoff represented by the call-spread replication."""
    a = replication_weight(K1, K2, K3, K4)
    return (
        max(ST - K1, 0.0)
        - max(ST - K2, 0.0)
        - a * max(ST - K3, 0.0)
        + a * max(ST - K4, 0.0)
    )


def simulate_one_price(
    S0: float,
    r: float,
    q: float,
    sigma: float,
    T: float,
    K1: float,
    K2: float,
    K3: float,
    K4: float,
    num_samples: int,
    rng: Random,
) -> float:
    """Return one Monte Carlo option price estimate using num_samples paths."""
    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")

    drift = (r - q - 0.5 * sigma * sigma) * T
    vol = sigma * sqrt(T)

    payoff_sum = 0.0
    for _ in range(num_samples):
        z = rng.gauss(0.0, 1.0)
        ST = exp(log(S0) + drift + vol * z)
        payoff_sum += payoff_terminal(ST, K1, K2, K3, K4)

    return exp(-r * T) * (payoff_sum / num_samples)


def price_bonus2_mc(
    S0: float,
    r: float,
    q: float,
    sigma: float,
    T: float,
    K1: float,
    K2: float,
    K3: float,
    K4: float,
    num_samples: int,
    num_repetitions: int,
    seed: int,
) -> dict:
    """Run repeated MC pricing and build CI as required by the assignment."""
    if num_repetitions <= 1:
        raise ValueError("num_repetitions must be at least 2.")

    rng = Random(seed)

    repetition_prices = [
        simulate_one_price(
            S0=S0,
            r=r,
            q=q,
            sigma=sigma,
            T=T,
            K1=K1,
            K2=K2,
            K3=K3,
            K4=K4,
            num_samples=num_samples,
            rng=rng,
        )
        for _ in range(num_repetitions)
    ]

    avg_price = mean(repetition_prices)
    sd_price = stdev(repetition_prices)

    ci_low = avg_price - 2.0 * sd_price
    ci_high = avg_price + 2.0 * sd_price

    return {
        "repetition_prices": repetition_prices,
        "mean": avg_price,
        "sd": sd_price,
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


def main() -> None:
    result = price_bonus2_mc(
        S0=config.S0,
        r=config.r,
        q=config.q,
        sigma=config.sigma,
        T=config.T,
        K1=config.K1,
        K2=config.K2,
        K3=config.K3,
        K4=config.K4,
        num_samples=config.MC_NUM_SAMPLES,
        num_repetitions=config.MC_NUM_REPETITIONS,
        seed=config.MC_RANDOM_SEED,
    )

    print('=' * 50)
    print("Bonus 2 Monte Carlo pricing")
    print(f"samples per repetition = {config.MC_NUM_SAMPLES}")
    print(f"number of repetitions  = {config.MC_NUM_REPETITIONS}")
    print(f"random seed            = {config.MC_RANDOM_SEED}")
    print(f"mean of 20 repetitions = {result['mean']:.6f}")
    print(f"s.d. of 20 repetitions = {result['sd']:.6f}")
    print(f"95% CI (mean +/- 2*sd) = [{result['ci_low']:.6f}, {result['ci_high']:.6f}]")
    print('=' * 50)


if __name__ == "__main__":
    main()
