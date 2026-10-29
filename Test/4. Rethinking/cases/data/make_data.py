"""Simulated datasets for the cross-language cases (one make_* per case)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd


def make_negative_binomial():
    """Reproduces the simulation of '7.Negative binomial.py' (np.random.seed(1))."""
    np.random.seed(1)
    num_days = 300
    y_days = np.random.poisson(1.5, num_days)
    num_weeks = 40
    y_weeks = np.random.poisson(0.5 * 7, num_weeks)

    y_all = np.concatenate([y_days, y_weeks])
    exposure = np.concatenate([np.repeat(1, num_days), np.repeat(7, num_weeks)])
    monastery = np.concatenate([np.repeat(0, num_days), np.repeat(1, num_weeks)])
    df = pd.DataFrame({'y': y_all, 'days': exposure, 'monastery': monastery})
    df['log_days'] = np.log(df['days'])

    df.to_csv(Path(__file__).parent / '7.Negative binomial.csv', index=False)


def make_multinomial():
    """Reproduces the simulation of '8.Multinomial.py' (np.random.seed(1))."""
    import jax

    N = 1000
    np.random.seed(1)
    # Each observation sees 3 careers with different incomes
    income_sim = np.random.uniform(1, 10, size=(N, 3))

    a_true_gen = np.array([0.5, -0.2])  # a3 is pivot 0
    b_true_gen = 0.4

    s1 = a_true_gen[0] + b_true_gen * income_sim[:, 0]
    s2 = a_true_gen[1] + b_true_gen * income_sim[:, 1]
    s3 = b_true_gen * income_sim[:, 2]
    scores = np.stack([s1, s2, s3], axis=1)
    p_true = np.asarray(jax.nn.softmax(scores, axis=1), dtype=np.float64)
    p_true = p_true / p_true.sum(axis=1, keepdims=True)

    career = np.array([np.random.choice([0, 1, 2], p=p) for p in p_true])
    df = pd.DataFrame({
        'career': career,
        'inc0': income_sim[:, 0],
        'inc1': income_sim[:, 1],
        'inc2': income_sim[:, 2]
    })

    df.to_csv(Path(__file__).parent / '8.Multinomial.csv', index=False)


def make_zero_inflated():
    """Reproduces the simulation of '10.Zero inflated.py' (np.random.seed(42))."""
    prob_drink = 0.2  # 20% of days
    rate_work = 1     # average 1 manuscript per day
    N = 365

    np.random.seed(42)
    drink = np.random.binomial(1, prob_drink, N)
    y = (1 - drink) * np.random.poisson(rate_work, N)

    df = pd.DataFrame({'y': y})
    df.to_csv(Path(__file__).parent / '10.Zero inflated.csv', index=False)


if __name__ == '__main__':
    wanted = sys.argv[1:]
    for name, fn in sorted(globals().items()):
        if name.startswith('make_') and callable(fn) and getattr(fn, '__module__', None) == __name__:
            if not wanted or any(name == w or name == f'make_{w}' for w in wanted):
                fn()
                print(name)
