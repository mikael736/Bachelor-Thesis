"""Numbers quoted in Section "ATE estimators and confidence intervals" (Chapter 5), in Appendix "ATE
estimators in E1 and E3" and in the ATE-estimator paragraph of Chapter 7, from
visualisation_output/ate_estimator_experiment_{1..4}/summary.csv. Printed per paragraph of the section:
bias, MSE, naive interval, AIPW interval. "Ends" are the target distributions Beta(2,8) and Beta(8,2), the
"center" the targets Beta(4,6) to Beta(6,4), and "moderate" the targets Beta(3,7) to Beta(7,3). The number of
capped density ratios is printed by density_ratio_cap.py.
Run: python ate_estimator_comparison.py
"""
from summaries import CENTER, ENDS, TARGETS, detectable, load

PI, AIPW = "Plug-in (naive CI)", "AIPW-corrected"
res = load("ate_estimator_experiment")
CENTRAL = TARGETS[2:5]  # Beta(4,6) to Beta(6,4)
MODERATE = TARGETS[1:6]  # Beta(3,7) to Beta(7,3)


def values(estimator, targets, field, experiments=(1, 2, 3, 4)):
    return [getattr(res[i][estimator, t], field) for i in experiments for t in targets]


print("== Bias (absolute values)")
for estimator in (PI, AIPW):
    b = lambda targets: max(abs(v) for v in values(estimator, targets, "bias"))
    ends = [abs(v) for v in values(estimator, ENDS, "bias")]
    print(f"{estimator}: max at {CENTER} {b([CENTER]):.3f}, from Beta(4,6) to Beta(6,4) {b(CENTRAL):.3f}, "
          f"at the ends {min(ends):.3f}-{max(ends):.3f}")
for estimator in (PI, AIPW):
    for i in res:
        within = [t for t in CENTRAL if abs(res[i][estimator, t].bias) <= 2 * res[i][estimator, t].mcse_bias]
        ratio = min(abs(res[i][estimator, t].bias) / res[i][estimator, t].mcse_bias for t in TARGETS)
        print(f"{estimator}, E{i}: |bias| <= 2 MCSE from Beta(4,6) to Beta(6,4) at {within}; "
              f"|bias| / MCSE over all targets at least {ratio:.1f}")
reduced = all(abs(res[i][AIPW, t].bias) < abs(res[i][PI, t].bias) for i in res for t in ENDS)
print(f"correction reduces the absolute bias at all ends: {reduced}")

print("\n== MSE")
mse_ratio = [res[i][AIPW, CENTER].mse / res[i][PI, CENTER].mse for i in res]
var_ratio = [res[i][AIPW, CENTER].variance / res[i][PI, CENTER].variance for i in res]
print(f"at {CENTER}: MSE of AIPW / MSE of plug-in {min(mse_ratio):.1f}-{max(mse_ratio):.1f} "
      f"(detectable: {[detectable(res[i][AIPW, CENTER], res[i][PI, CENTER]) for i in res]}), "
      f"variance ratio {min(var_ratio):.1f}-{max(var_ratio):.1f}")
for i in res:
    print(f"E{i}: " + ", ".join(
        f"{t} lower {'AIPW' if res[i][AIPW, t].mse < res[i][PI, t].mse else 'plug-in'} "
        f"(detectable: {detectable(res[i][AIPW, t], res[i][PI, t])})" for t in TARGETS))

print("\n== Naive interval")
naive = values(PI, TARGETS, "coverage")
naive_ends = values(PI, ENDS, "coverage")
print(f"coverage at most {max(naive):.2f}; at the ends {min(naive_ends):.2f}-{max(naive_ends):.2f}")

print("\n== AIPW interval")
for label, targets in (("Beta(4,6) to Beta(6,4)", CENTRAL), ("Beta(3,7) to Beta(7,3)", MODERATE),
                       ("ends", ENDS)):
    cov = values(AIPW, targets, "coverage")
    print(f"coverage, {label}: {min(cov):.2f}-{max(cov):.2f}")
for label, targets in ((CENTER, [CENTER]), ("ends", ENDS)):
    width = values(AIPW, targets, "mean_ci_width")
    print(f"mean width, {label}: {min(width):.2f}-{max(width):.2f}")
for i in res:
    cov = values(AIPW, MODERATE, "coverage", (i,))
    cov_ends = values(AIPW, ENDS, "coverage", (i,))
    naive_max = max(values(PI, TARGETS, "coverage", (i,)))
    print(f"E{i}: naive coverage at most {naive_max:.2f}, AIPW coverage Beta(3,7) to "
          f"Beta(7,3) {min(cov):.2f}-{max(cov):.2f}, at the ends {min(cov_ends):.2f}-{max(cov_ends):.2f}")
