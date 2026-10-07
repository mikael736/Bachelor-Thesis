# Experiment design

Working notes on which experiments go into the thesis. Not final.

## Fixed across all experiments

Each setting with a short justification, phrased so it can go into the thesis.

- **Meta-learners: T-, X- and DR-learner, random forests as base learners.** The comparison is
  between meta-learners, so the base learner is held fixed.
- **Propensity: `propensity.sigmoid` (center 0.5, multiplier 5), clipped to [0.05, 0.95].**
  Treatment probability rises with x, so the arms cover different x regions (confounding). It is
  about 0.12 at x = 0.1 and 0.88 at x = 0.9, and the clip guarantees overlap.
- **Noise: N(0, 1).** All errors are in units of the noise variance, so MSEs are comparable across
  experiments; the signal strength is set by scaling mu0 and tau instead (see Scaling).
- **Training covariates: Beta(5, 5), n = 1000.** At this size the tails are sparse: per 1000
  training units, about 3 controls and 16 treated have x > 0.8, while Beta(8, 2) puts 56% of its
  test mass there. The learners must extrapolate from few units at the ends of the sweep, which is
  the phenomenon under study. n = 1000 is also a common scale in CATE simulation studies.
- **Test populations: the Beta shape sweep Beta(a, 10 - a), a = 2, 3, ..., 8 (test mean 0.2 to
  0.8; covariate shift into the tails, Beta(5, 5) = training in the middle), 100,000 units per
  sweep point, drawn once.** The actual ATE is computed exactly; the test sample only approximates
  the learner's estimated ATE, E_test[tau_hat(X)], by Monte Carlo integration. Its effect on the
  reported MSE is at most a few percent at the ends of the sweep and at most ~10% at its centre,
  where all MSEs are smallest (derivation in "Monte Carlo error analysis", part B). Since the sample
  is drawn once, more replications would not reduce this error, hence the large size. It is not
  meant to mimic a real sample size.
- **Random forests: `min_samples_leaf = 5` for the regressors and the propensity classifier.**
  Larger leaves (e.g. 20 / 80) minimize the out-of-bag error, but that error is weighted by the
  training distribution. Under covariate shift the test mass sits in the sparse tails, where large
  leaves flatten the fit; in end-to-end comparisons (5/5, 5/10, 20/80, classifier 10 vs. 50),
  smaller leaves won clearly in curved settings and lost only slightly in flat ones. 5 is also the
  standard regression default (Breiman; R `randomForest`; `grf`).
- **Replications: N_REPS = 350 for the thesis runs (50 during development).** Each MSE is an
  average over replications and so carries Monte Carlo error. Pilot runs (50 reps) on a hard (sine
  mu0) and an easy (constant) setting showed that about 340-360 reps bound the relative Monte
  Carlo standard error of every reported MSE at about 10% (derivation in "Monte Carlo error
  analysis", part A). Thesis sentence: "Each configuration
  was replicated R = 350 times, which bounds the Monte Carlo standard error of every reported MSE
  at about 10% of its value (Morris, White & Crowther, 2019)."

## Main part: mu0 complexity x tau type

The two axes along which the meta-learner literature (Kuenzel et al. 2019, Kennedy 2020) explains
when T-, X- and DR-learner differ. Each experiment isolates one mechanism with a clear expectation.

|                     | tau constant | tau smooth (e.g. linear) | tau abrupt (e.g. step) |
|---------------------|--------------|--------------------------|------------------------|
| mu0 simple (linear) | - (trivial)  | **E2**                   | **E3**                 |
| mu0 complex (sine)  | **E1**       | **E4**                   | appendix               |

- **E1 - complex mu0, constant tau: false-heterogeneity baseline.** The actual ATE is flat over the
  sweep, so any trend in the estimates is a learner artefact. Expectation: the T-learner fits mu0
  and mu1 separately on arms that cover different x regions (sigmoid propensity), so their
  difference shows spurious heterogeneity, mostly in the tails; X and DR stay flatter.
  Var(tau) = 0 here, so report the effect size d = ATE / sigma instead of the SNR.
- **E2 - simple mu0, smooth tau: extrapolation under shift.** The ATE changes along the sweep.
  Forests predict a constant beyond the training support, so all learners should flatten tau in
  the tails and underestimate the ATE shift. Question: does any learner do better, or is this a
  shared limit of random forests?
- **E3 - simple mu0, abrupt tau: local heterogeneity under poor overlap.** The step sits at
  x = 0.7, where controls are sparse. Forests handle steps well where data exist; this
  tests the learners where heterogeneity and weak overlap coincide.
- **E4 - complex mu0, smooth tau: realistic combination.** tau is simpler than mu0, the setting
  where X and DR should beat T. Checks whether the advantage from E1 survives with real heterogeneity.

### Scaling (sigma = 1, variances over the training distribution Beta(5, 5))

- tau: Var(tau) ~ 1 for E2-E4 (SNR = Var(tau) / sigma^2 ~ 1).
- simple mu0: Var ~ 1. Complex mu0: clearly larger, Var ~ 4-9, so outcome-model difficulty dominates.
- Reference values on Beta(5, 5) (Var(X) = 1/44):
  - `linear(slope=s)`: Var = s^2 / 44 -> s ~ 6.6 for Var 1.
  - `step(jump=j, threshold=t)`: j * 1(X > t) is j times a Bernoulli(p) variable with
    p = P(X > t), so Var = j^2 p (1 - p). For t = 0.7: p ~ 0.099, Var ~ 0.089 j^2 -> j ~ 3.35 for Var 1.
  - `sine(amplitude=A, frequency=2)`: Var ~ 0.5 A^2 -> A = 3 gives Var ~ 4.5, A = 4 gives ~ 8.0.
  - `exponential(scale=0.5, multiplier=3)`: Var ~ 1.32 (scales with scale^2).

### Concrete specifications (proposed)

| Exp | mu0                               | Var(mu0) | tau                                   | Var(tau)        |
|-----|-----------------------------------|----------|---------------------------------------|-----------------|
| E1  | `sine(amplitude=3, frequency=2)`  | ~4.5     | `constant(value=1)`                   | 0 (d = 1)       |
| E2  | `linear(slope=7)`                 | ~1.1     | `linear(slope=7)`                     | ~1.1            |
| E3  | `linear(slope=7)`                 | ~1.1     | `step(jump=3, threshold=0.7)`         | ~0.8            |
| E4  | `sine(amplitude=3, frequency=2)`  | ~4.5     | `linear(slope=7)`                     | ~1.1            |

Parameters rounded to whole numbers (exact Var ~ 1 would need slope 6.6, jump 3.35).

- Step at 0.7 (E3): asymmetric, on the side where controls are sparse. e(0.7) ~ 0.73, and per
  1000 training units only ~21 controls (vs. ~77 treated) have x > 0.7. About 10% of the training
  mass lies above the step, so the jump must be large for Var(tau) ~ 1.
- E1 and E4 share mu0, E2 and E3 share mu0, and E2 and E4 share tau, so neighbouring grid cells
  differ in exactly one function.

### Why SNR ~ 1 and not a "realistic" effect size

Empirical effects are typically small relative to outcome noise (standardized effect d ~ 0.1-0.2).
At heterogeneity that weak (sd(tau) / sigma ~ 0.1) and n = 1000, tau(x) is barely learnable, the
actual ATE hardly moves across the sweep, and all learners look alike - the covariate-shift
mechanism under study would not be visible. SNR ~ 1 is a design choice that makes it visible; the
realistic regime goes into the appendix as a robustness check.

## Why 1-D covariates in the main part

- The test sweep shifts one distribution, so "where the test mass goes" relative to the training
  data and the propensity is directly interpretable.
- tau(x), mu0(x) and the overlap can be drawn as curves, so every mechanism above can be read off
  the plots.
- Each experiment then varies exactly one thing (the shapes of mu0 and tau). In d > 1, the
  covariate dependence, irrelevant covariates and the forest's `max_features` all become
  additional factors that would mix into the comparison.
- The thesis question (how meta-learners transport the ATE under covariate shift) is already
  posed in 1-D. Multidimensional effects are a separate question, better treated in the appendix.

## Appendix ideas

- **Remaining grid cell:** complex mu0, abrupt tau.
- **Low-signal regime:** sd(tau) / sigma ~ 0.2-0.3 for E2 or E4 - does the learner ranking hold?
- **Overlap:** sigmoid multiplier 2 vs. 5 vs. 10.
- **DR-learner with vs. without cross-fitting:** one ablation plot.
- **Training size:** n in {500, 1000, 2000, 5000} for one or two experiments. If the errors at the
  ends of the sweep barely shrink with n, they come from extrapolation, not from small samples.
- **3-D covariates** (`covariates.coordinatewise_3d`: x1 = x0^2, x2 independent; 0-based coordinates): repeat one or
  two main experiments (e.g. E1 and E4) with mu0 / tau depending on x0 only. Questions: do
  correlated or irrelevant covariates hurt the learners, and does `max_features` matter then?

## Monte Carlo error analysis (material for the appendix)

Two independent sources of Monte Carlo error enter every reported MSE: the finite number of
replications R (part A) and the finite test sample of size M used to integrate the estimated ATE
(part B). Notation for one learner and one test distribution P:

- theta = E_P[tau(X)]: actual ATE, computed exactly.
- Replication r = 1..R: training data D_r (independent seeds) -> fitted CATE tau_hat_r.
- theta_hat_r = E_P[tau_hat_r(X)]: estimated ATE with exact integration; error e_r = theta_hat_r - theta.
- MSE = E[e^2] (target); MSE_hat = (1/R) sum_r e_r^2 (what part A analyses).

### A. Replications: choosing N_REPS

1. Z_r = e_r^2 are i.i.d. over r (independent training draws, fixed test sample), so
   E[MSE_hat] = MSE (unbiased).
2. Var(MSE_hat) = Var(Z) / R, so MCSE = sd(Z) / sqrt(R), estimated with the sample sd of Z_r.
3. Relative MCSE: rho_R = MCSE / MSE = CV(Z) / sqrt(R), with CV(Z) = sd(Z) / E[Z] a property of
   the error distribution, independent of R.
4. Required replications for a target rho*: R >= (CV(Z) / rho*)^2. From a pilot with R0 reps:
   R = R0 * (rho_hat_R0 / rho*)^2. Taken as the worst case over learners and test distributions.
5. Interpretation (CLT): MSE_hat lies in MSE * (1 +- 1.96 rho_R) with ~95% probability.
6. Normal benchmark: if e = b + sigma * xi, xi ~ N(0, 1), then E[Z] = b^2 + sigma^2 and
   Var(Z) = 4 b^2 sigma^2 + 2 sigma^4 (Cov(xi, xi^2) = E[xi^3] = 0, Var(xi^2) = 2). With
   k = b^2 / sigma^2: CV^2 = (2 + 4k) / (1 + k)^2 <= 2, so rho_R <= sqrt(2 / R); bias only helps.
   R = 200 gives ~10%.
7. Heavy tails: without bias, CV^2 = kappa - 1 with kurtosis kappa = E[e^4] / E[e^2]^2 (normal: 3).
   The pilot's worst point had rho_hat ~ 26% at R0 = 50, i.e. CV^2 ~ 3.4, kappa ~ 4.4: occasional
   reps with very large errors (typically the DR-learner at the ends of the sweep). Hence ~350 reps.
8. Caveats: CV is a fourth-moment quantity estimated from 50 reps, so it is itself noisy (and
   tends to be underestimated under heavy tails); the result is an approximate worst case, not a
   strict bound - keep "about 10%" in the wording.

Reference: Morris, White & Crowther (2019), "Using simulation studies to evaluate statistical
methods", Statistics in Medicine.

### B. Test sample: justifying Monte Carlo integration of the estimated ATE

Setup: test sample S = {X_1..X_M} i.i.d. from P, drawn once, independent of all D_r. The code
reports theta_tilde_r = (1/M) sum_i tau_hat_r(X_i) = theta_hat_r + delta_r and
MSE_tilde = (1/R) sum_r (theta_tilde_r - theta)^2.

1. Decomposition: MSE_tilde = MSE_hat + C + Q, with cross term C = (2/R) sum_r e_r delta_r and
   quadratic term Q = (1/R) sum_r delta_r^2. Analyse C and Q with the fitted learners held fixed
   (randomness over S only).
2. Single integration error: S is independent of D_r, so E[delta_r | D_r] = 0 and
   Var(delta_r | D_r) = v_r / M, with v_r = Var_P(tau_hat_r(X)) (how much the fitted CATE varies
   over the test distribution).
3. Q is a small positive offset: E[Q | D] = v_bar / M, v_bar = (1/R) sum_r v_r.
4. C has mean zero: E[C | D] = (2/R) sum_r e_r E[delta_r | D_r] = 0 - no systematic error.
   Its variance must account for the shared test sample (delta_r correlated across r). Rewrite
   C = (2/M) sum_i g(X_i) with g(x) = (1/R) sum_r e_r (tau_hat_r(x) - theta_hat_r), E_P[g] = 0, so
   sd(C | D) = 2 sd_P(g) / sqrt(M). Cauchy-Schwarz over r, pointwise in x:
   g(x)^2 <= [(1/R) sum_r e_r^2] [(1/R) sum_r (tau_hat_r(x) - theta_hat_r)^2], so
   E_P[g^2] <= MSE_hat * v_bar and sd(C | D) <= 2 sqrt(MSE_hat * v_bar / M).
5. Relative error: with q = v_bar / (M * MSE_hat), the test sample inflates the reported MSE by
   ~q (relative) and adds a mean-zero fluctuation with relative sd <= 2 sqrt(q). Total <~ q + 2 sqrt(q).
6. Numbers (M = 100,000, v_bar ~ 1):
   - centre of the sweep (MSE ~ 0.004): q ~ 0.0025, 2 sqrt(q) ~ 10%
   - ends of the sweep (MSE 0.1-1): q <= 1e-4, 2 sqrt(q) <= 2%
7. Remarks:
   - The bound is conservative (Cauchy-Schwarz is tight only if e_r and the shapes of
     tau_hat_r(x) - theta_hat_r align perfectly across reps).
   - v_bar depends on the setting: for constant tau (E1), tau_hat varies only through spurious
     heterogeneity, so v_bar << 1. It can be read off a run as the variance of the predictions on
     the test sample.
   - The error scales with 1/sqrt(M) (M = 10^6 -> ~3% at the centre).
   - S is drawn once, so the fluctuation is the same in every rep and does not average out over
     replications; only M controls it.
   - All learners share S, so their fluctuations are correlated and partly cancel in comparisons.
   - It is largest where it matters least: at the centre of the sweep, where all MSEs are smallest.

Draft appendix text:

> The actual ATE theta = E_P[tau(X)] is computed exactly. Each estimated ATE E_P[tau_hat(X)] is
> approximated by Monte Carlo integration over a fixed sample of M = 100,000 draws from P,
> independent of the training data. The integration error delta has conditional mean zero and
> variance Var_P(tau_hat(X)) / M. It inflates the reported MSE by v_bar / M on average and adds a
> mean-zero fluctuation with standard deviation at most 2 sqrt(MSE * v_bar / M), i.e. at most
> q + 2 sqrt(q) relative to the MSE, with q = v_bar / (M * MSE). In our settings this is at most
> about 2% at the ends of the covariate-shift sweep and about 10% at its centre, where all MSEs
> are smallest.
