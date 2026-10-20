"""Numbers quoted in Section "Plug-in estimates of the meta-learners" (Chapter 5) and in the meta-learner
paragraph of Chapter 7, from visualisation_output/metalearner_experiment_{1..4}/summary.csv. Printed per
paragraph of the section: effect of the covariate shift, complex baseline (E1, E4), simple baseline (E2, E3),
DR-learner, summary. "Ends" are the target distributions Beta(2,8) and Beta(8,2).
Run: python meta_learner_comparison.py
"""
from summaries import CENTER, ENDS, TARGETS, detectable, load

LEARNERS = ["T-learner", "X-learner", "DR-learner"]
res = load("metalearner_experiment")
cases = [(i, t) for i in res for t in ENDS]  # the eight (experiment, end) cases


def mse(i, learner, target):
    return res[i][learner, target].mse


print("== Effect of the covariate shift")
for learners in (["T-learner", "X-learner"], ["DR-learner"]):
    center = [mse(i, l, CENTER) for i in res for l in learners]
    ends = [mse(i, l, t) for i, t in cases for l in learners]
    print(f"{' and '.join(learners)}: MSE at {CENTER} {min(center):.4f}-{max(center):.4f}, "
          f"at the ends {min(ends):.3f}-{max(ends):.3f}")
growth = [mse(i, l, t) / mse(i, l, CENTER) for i, t in cases for l in LEARNERS]
print(f"MSE at an end / MSE at {CENTER}, same learner and experiment: at least {min(growth):.1f}")
ratio_tx = [mse(i, "T-learner", t) / mse(i, "X-learner", t) for i in res for t in TARGETS]
ratio_tx = [max(q, 1 / q) for q in ratio_tx]
print(f"MSE ratio of the T- and X-learner (larger / smaller), any target: at most {max(ratio_tx):.1f}")
print(f"T vs X detectable at {CENTER}:",
      [detectable(res[i]["T-learner", CENTER], res[i]["X-learner", CENTER]) for i in res])

print("\n== Complex and simple baseline: MSE of the T-learner / MSE of the X-learner at the ends")
for i in res:
    print(f"E{i}: " + ", ".join(
        f"{t} {mse(i, 'T-learner', t) / mse(i, 'X-learner', t):.2f} "
        f"(detectable: {detectable(res[i]['T-learner', t], res[i]['X-learner', t])})" for t in ENDS))

print("\n== Mean estimate (E1) and bias, variance at the ends")
for l in ["T-learner", "X-learner"]:
    print(f"E1 {l}: mean estimate {CENTER} {res[1][l, CENTER].mean_estimate:.2f}, "
          + ", ".join(f"{t} {res[1][l, t].mean_estimate:.2f}" for t in ENDS))
for i in (2, 3):
    for l in ["T-learner", "X-learner"]:
        print(f"E{i} {l}: " + ", ".join(
            f"{t} bias {res[i][l, t].bias:+.2f} variance {res[i][l, t].variance:.3f}" for t in ENDS))

print("\n== DR-learner")
for other in ["T-learner", "X-learner"]:
    center = [mse(i, "DR-learner", CENTER) / mse(i, other, CENTER) for i in res]
    ends = [mse(i, "DR-learner", t) / mse(i, other, t) for i, t in cases]
    print(f"MSE of the DR-learner / MSE of the {other}: at {CENTER} {min(center):.1f}-{max(center):.1f}, "
          f"at the ends {min(ends):.1f}-{max(ends):.1f}")
largest = all(mse(i, "DR-learner", t) > max(mse(i, "T-learner", t), mse(i, "X-learner", t))
              for i in res for t in TARGETS)
not_detectable = [(f"E{i}", t, other) for i in res for t in TARGETS for other in ["T-learner", "X-learner"]
                  if not detectable(res[i]["DR-learner", t], res[i][other, t])]
print(f"DR-learner has the largest MSE everywhere: {largest}; not detectable: {not_detectable}")
share = [res[i]["DR-learner", t].variance / mse(i, "DR-learner", t) for i in res for t in TARGETS]
print(f"variance share of the DR-learner's MSE: at least {min(share):.2f}")
smaller_bias = [(f"E{i}", t) for i, t in cases
                if abs(res[i]["DR-learner", t].bias) < abs(res[i]["T-learner", t].bias)]
print(f"ends where |bias| of the DR-learner < |bias| of the T-learner: {len(smaller_bias)} of {len(cases)}, "
      f"exceptions: {[(f'E{i}', t) for i, t in cases if (f'E{i}', t) not in smaller_bias]}")
var_ratio = [res[i]["DR-learner", t].variance / res[i]["T-learner", t].variance for i, t in cases]
print(f"variance of the DR-learner / variance of the T-learner at the ends: "
      f"{min(var_ratio):.1f}-{max(var_ratio):.1f}")

print("\n== Summary: lowest MSE at the ends")
for i, t in cases:
    best = min(LEARNERS, key=lambda l: mse(i, l, t))
    others = [l for l in LEARNERS if l != best]
    print(f"E{i} {t}: {best}, detectably vs both others: "
          f"{all(detectable(res[i][best, t], res[i][o, t]) for o in others)}")
