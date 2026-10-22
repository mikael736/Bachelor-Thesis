"""Second moment of the density ratio r = p_T / p_S for the study distribution Beta(5, 5) and the
target distributions Beta(a, 10 - a), a = 2, ..., 8 (Appendix "Second moment of the density ratio",
Lemma and Table r-moment).

E_S[r(X)^2] = E_T[r(X)] drives the variance of the AIPW correction. With p_S = Beta(5, 5) and
p_T = Beta(a, 10 - a), r(x) = B(5, 5) / B(a, 10 - a) * x^(a - 5) (1 - x)^(5 - a), so
    E_T[r] = B(5, 5) B(2a - 5, 15 - 2a) / B(a, 10 - a)^2,
finite only if 2a - 5 > 0 and 15 - 2a > 0, i.e. infinite at the ends a = 2 and a = 8.
Checked against numerical integration. n / E_S[r^2] is the effective sample size of the reweighted
study sample (n = 1000). Run: python density_ratio_moments.py
"""
import numpy as np
from scipy import integrate, special, stats

N_STUDY = 1000
study = stats.beta(5, 5)

print(f"{'target':12s}{'E_S[r^2] closed form':>22s}{'numerical':>12s}{'n / E_S[r^2]':>14s}")
for a in range(2, 9):
    target = stats.beta(a, 10 - a)
    if 2 * a - 5 > 0 and 15 - 2 * a > 0:
        closed = special.beta(5, 5) * special.beta(2 * a - 5, 15 - 2 * a) / special.beta(a, 10 - a) ** 2
        numerical, _ = integrate.quad(lambda x: target.pdf(x) ** 2 / study.pdf(x), 0, 1)
        print(f"Beta({a},{10 - a})  {closed:20.3f}{numerical:12.3f}{N_STUDY / closed:14.0f}")
    else:
        print(f"Beta({a},{10 - a})  {'inf':>20s}{'diverges':>12s}{0:14d}")
