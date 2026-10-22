"""Variance of the response functions of Chapter 5 under the study distribution Beta(5, 5) and the
target distributions Beta(a, 10 - a), a = 2, ..., 8, by numerical integration (scipy expect).
Used for the caption of Table "experiments" and the target-sampling footnote in Section "Setup".
Run: python response_variances.py
"""
import numpy as np
from scipy import stats

FUNCTIONS = {
    "sine mu0 3 sin(4 pi x)": lambda x: 3 * np.sin(4 * np.pi * x),
    "linear 7x": lambda x: 7 * x,
    "step 3 1{x > 0.7}": lambda x: 3.0 * (x > 0.7),
    "constant 1": lambda x: 1.0 + 0 * x,
}
JUMPS = [0.7]  # passed to the integrator so the step is integrated exactly


def variance(dist, f):
    mean = dist.expect(f, points=JUMPS)
    return dist.expect(lambda x: (f(x) - mean) ** 2, points=JUMPS)


print("distribution".ljust(20) + "".join(name.ljust(26) for name in FUNCTIONS))
for a in range(2, 9):
    dist = stats.beta(a, 10 - a)
    label = f"Beta({a},{10 - a})" + (" study" if a == 5 else "")
    print(label.ljust(20) + "".join(f"{variance(dist, f):<26.3f}" for f in FUNCTIONS.values()))
