"""Mass of the study distribution Beta(5, 5) and the target distributions Beta(a, 10 - a),
a = 2, ..., 8, in the tails x < 0.2 and x > 0.8, and the propensity score e(x) = 1 / (1 + exp(-5 (x - 0.5)))
at x = 0, 0.2, 0.8, 1, and the expected number of treated and control units in both tails of a study
sample of size n = 1000. Used for the paragraph "Covariate shift" and the propensity range in the paragraph
"Data-generating process" of Chapter 5, and the unit counts in the paragraph "X-learner and T-learner".
Run: python tail_masses.py
"""
import numpy as np
from scipy import integrate, stats

LOWER, UPPER = 0.2, 0.8
N_STUDY = 1000


def propensity(x):
    return 1 / (1 + np.exp(-5 * (x - 0.5)))


print(f"{'distribution':20s}{'P(X < 0.2)':>12s}{'P(X > 0.8)':>12s}")
for a in range(2, 9):
    dist = stats.beta(a, 10 - a)
    label = f"Beta({a},{10 - a})" + (" study" if a == 5 else "")
    print(f"{label:20s}{dist.cdf(LOWER):12.3f}{dist.sf(UPPER):12.3f}")

print()
for x in [0.0, LOWER, UPPER, 1.0]:
    print(f"e({x}) = {propensity(x):.3f}")

# expected number of units of each arm in the tails: n * int f(x) e(x) dx resp. n * int f(x) (1 - e(x)) dx
print()
study = stats.beta(5, 5)
print(f"{'tail':10s}{'treated':>10s}{'control':>10s}")
for label, lo, hi in [("x < 0.2", 0.0, LOWER), ("x > 0.8", UPPER, 1.0)]:
    treated = N_STUDY * integrate.quad(lambda x: study.pdf(x) * propensity(x), lo, hi)[0]
    control = N_STUDY * integrate.quad(lambda x: study.pdf(x) * (1 - propensity(x)), lo, hi)[0]
    print(f"{label:10s}{treated:10.1f}{control:10.1f}")
