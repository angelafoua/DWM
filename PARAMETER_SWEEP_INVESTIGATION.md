# DWM Parameter Sweep — Root Cause Investigation Report

**Date:** 2026-06-29  
**Branch:** `claude/dwm-parameter-sweep-investigation-hg041b`  
**Dataset examined:** S1G.txt (50 records, truth: truthABCgoodDQ.txt)

---

## 1. Executive Summary

The parameter sweep consistently produces precision 0.04–0.23 and recall ≈ 1.0 because of **two compounding problems**, both verified by code inspection and quantitative data analysis:

**Problem A — The dataset has a collapsed effective parameter space.**  
S1G.txt has only 50 records, so the maximum possible token frequency is 47 ("NC"). This means:
- Any beta ≥ 47 produces **identical blocking behavior** (104 of 149 beta values, 69.8%).
- Any sigma ≥ 48 removes **zero tokens** from scoring (103 of 145 sigma values, 71.0%).
- On the combined (beta, sigma) space, roughly **48.5%** of valid drawn configurations sit in the fully-saturated zone where neither parameter changes anything.
- Beta has only **8 distinct behavioral breakpoints** for this dataset (not 149).
- Sigma has only **9 distinct behavioral breakpoints** (not 145).

**Problem B — `mu_iterate=0.05` is hard-wired into `run_sweep()`, invalidating the mu and epsilon sensitivity analysis.**  
Every sweep configuration runs as a multi-iteration algorithm starting at the sampled `mu` and incrementing by 0.05 each iteration until mu > 1.0. A configuration with mu=0.1 runs 19 iterations; one with mu=0.9 runs 3 iterations. Links created in early low-mu iterations are **permanently retained** in `linkIndex`. The reported "mu" is the *starting* threshold, not the effective threshold, so configurations with different starting values are not comparable.

Together, these two defects explain all observed symptoms: flat F1 landscape, parameter insensitivity, high recall, and low precision.

---

## 2. Parameter Dependency Graph

```
DWM10_Parms (global state)
│
├─ beta ────────────────────────────────────→ DWM42_BuildBlockPairs
│                                               Condition: 2 ≤ token_freq ≤ beta
│                                               Output: blockPairList (candidate pairs)
│
├─ sigma ───────────────────────────────────→ DWM55_LinkBlockPairs
│                                               removeStopWords(): drops tokens with freq ≥ sigma
│                                               Output: filtered tokenList1/2 before scoring
│
├─ mu ──────────────────────────────────────→ DWM55_LinkBlockPairs (threshold: result ≥ mu)
│                                           → DWM65_ScoringMatrixStd (early-exit: score < mu)
│                                               Output: linkedPairList
│
├─ epsilon ─────────────────────────────────→ DWM90_IterateClusters
│                                           → DWM95_CalculateEntropy (early-exit: quality < epsilon)
│                                               Output: filters which clusters enter linkIndex
│
├─ mu_iterate ──────────────────────────────→ DWM00_Driver / sweep_runner.py
│                                               DWM10_Parms.mu += mu_iterate each iteration
│                                               Controls when outer loop terminates (mu > 1.0)
│
└─ epsilon_iterate ─────────────────────────→ DWM00_Driver / sweep_runner.py
                                                DWM10_Parms.epsilon += epsilon_iterate
```

**What does NOT influence mu or epsilon computation:** sigma, beta, and transitive closure are upstream; they never feed back into threshold values.

**What epsilon does NOT do:** it cannot remove false positives that were accepted in a previous iteration. `linkIndex` is write-only per entry — once a refID is assigned a clusterID, it is never re-evaluated.

---

## 3. Sensitivity Analysis

### 3.1 Beta

**Intended role:** controls which tokens qualify as blocking keys (2 ≤ freq ≤ beta).

**Observed on S1G.txt:**

| beta range | blocking tokens added | blocking behavior |
|---|---|---|
| 2 | freq = 2 only | 61 tokens |
| 3 | freq = 2–3 | 97 tokens (+36) |
| 4–5 | freq = 2–5 | 105 tokens |
| 6 | freq = 2–6 | 114 tokens |
| 7–9 | freq = 2–9 | 115 tokens |
| 10–11 | freq = 2–11 | 116 tokens |
| 12 | freq = 2–12 | 117 tokens |
| 13–46 | freq = 2–46 | 118 tokens |
| **47–150** | freq = 2–47 (all non-singletons) | **122 tokens — no change above 47** |

**The maximum token frequency in this dataset is 47 ("NC").** Any beta ≥ 47 includes all available blocking tokens and produces identical blocking. Beta values 47–150 cover 104 of 149 candidate values (69.8%). The sweep samples this flat zone roughly 70% of the time.

**Effect on precision/recall:** increasing beta from 2 to 47 admits more common tokens as blocking keys, which increases blocking recall (more candidate pairs) but reduces precision (common-token collisions). Above 47, no change.

### 3.2 Sigma

**Intended role:** stop-word threshold — removes tokens with freq ≥ sigma before scoring.

**Observed on S1G.txt:**

| sigma | tokens removed from scoring |
|---|---|
| 6 | 13 (NC, AARON, WINSTON, SALEM, RD, DR, 27104, 27106, ST, 27103, 2475, SPICEWOOD, CT) |
| 7 | 8 |
| 8–10 | 7 |
| 11–12 | 6 |
| 13 | 5 |
| 14–31 | 4 |
| 32–33 | 2 |
| 34–47 | 1 (only "NC") |
| **48–150** | **0 — nothing removed** |

Sigma values 48–150 cover 103 of 145 candidate values (**71.0%**). For 71% of drawn sigma values, the scoring function sees the full, undiscriminated token set.

The most important sigma transitions are at 6, 7, 11, 13, and 48. All other sigma values within each band are functionally identical.

### 3.3 Mu

**Intended role:** pairwise similarity threshold — two records link only if `score ≥ mu`.

**Actual behavior in the sweep:** `run_sweep()` hardcodes `mu_iterate=0.05` (sweep_runner.py line 259). This means:
- mu=0.10 → algorithm runs 19 iterations at thresholds 0.10, 0.15, 0.20, ..., 1.00
- mu=0.50 → 11 iterations at 0.50, 0.55, ..., 1.00
- mu=0.90 → 3 iterations at 0.90, 0.95, 1.00
- mu=1.00 → 1 iteration at 1.00

Records linked in iteration 1 at the starting (low) threshold are **permanently kept** in `linkIndex` regardless of mu in later iterations. The sweep result for "mu=0.1" is not "performance at threshold 0.1"; it is "cumulative performance after 19 passes starting at 0.1". These are fundamentally different experiments.

**Effect:** mu sensitivity is suppressed and confounded. Low starting-mu runs accumulate more false positives from iteration 1, which transitive closure then amplifies. This creates the false appearance that "mu doesn't matter."

### 3.4 Epsilon

**Intended role:** entropy-based cluster quality gate — rejects clusters with quality < epsilon.

**Actual limitations:**
1. Epsilon gates only the clusters formed in the **current** iteration. It cannot evict previously-accepted false positives from `linkIndex`.
2. With `epsilon_iterate=0.0` (the sweep default), epsilon is static across all iterations.
3. On a 50-record dataset with moderate cluster sizes (2–5 records), entropy quality is generally high (clusters of true duplicates share most tokens), so epsilon rarely rejects good clusters. It may also fail to reject bad ones if the bad member shares many tokens with the true cluster.
4. Because the dataset is small, most clusters form in the first 1–2 iterations. Epsilon's influence window is therefore narrow.

---

## 4. Root Cause of Poor Precision

### 4.1 The mu_iterate iteration structure accumulates false positives

**Code location:** `sweep/sweep_runner.py`, lines 159–189.

```python
# Default: mu_iterate=0.05
current_mu = DWM10_Parms.mu          # e.g., 0.30
while more:
    blockPairList = DWM42_BuildBlockPairs.buildBlockPairs(...)
    linkedPairList = DWM55_LinkBlockPairs.linkBlockPairs(...)  # uses current_mu
    clusterList    = DWM80_TransitiveClosure.transitiveClosure(linkedPairList)
    DWM90_IterateClusters.iterateClusters(clusterList, refDict, linkIndex)  # writes linkIndex
    
    current_mu += DWM10_Parms.muIterate   # 0.30 → 0.35 → … → 1.00
    DWM10_Parms.mu = current_mu
    if current_mu > 1.0:
        more = False
```

At mu=0.30 (first iteration), many similar-but-not-matching person records score ≥ 0.30 because they share city, state, zip, or street name tokens. These links enter `linkIndex` permanently. Transitive closure then merges them into larger clusters.

In iteration 2 (mu=0.35), only *unlinked* records are re-blocked. The false positives from iteration 1 are already gone from the unlinked pool — they remain as accepted clusters.

### 4.2 Transitive closure is the amplifier

**Code location:** `DWM80_TransitiveClosure.py`.

If A↔B is a true match and A↔C is a false positive (both linked at low mu), transitive closure creates cluster {A, B, C}. This generates linked pair (B, C) — a false positive never directly evaluated by scoring. The explosion is proportional to cluster size: a cluster of n records creates C(n,2) linked pairs.

On S1G.txt, records share "WINSTON", "SALEM", "NC", "27104", etc. Records for different entities that share the same zip code + street name will score ≥ 0.30. Once linked to a true-match cluster via transitive closure, they contaminate all pairwise counts.

### 4.3 Epsilon cannot undo transitive-closure-amplified false positives

**Code location:** `DWM90_IterateClusters.py`, lines 14, 47–52.

```python
epsilon = DWM10_Parms.epsilon
...
if quality >= epsilon:
    for k in range(0, len(clusterIndex)):
        linkIndex[indexVal] = currentCID     # write only if quality passes
# write ALL clusters to iterationLinkIndex regardless of quality
for k in range(0, len(clusterIndex)):
    iterationLinkIndex[indexVal] = currentCID
```

A cluster that passes epsilon in iteration 1 is written to `linkIndex`. In iteration 2, those records are already linked and are not re-blocked. The entropy check on iteration 2's new clusters cannot retroactively modify iteration 1's accepted clusters.

### 4.4 The dataset geometry ensures near-perfect recall

All 50 records are person records from the Winston-Salem, NC area. True duplicates (records for the same person) share most of their name, address, and SSN tokens. With blockByPairs=True and beta ≥ 13, true duplicate pairs almost always share at least 2 blocking tokens. Blocking recall is therefore nearly 100% for most beta values, and scoring recall at any reasonable mu is also near 100%. This explains why recall is consistently 1.0 (or near it) across the sweep.

---

## 5. Root Cause of Parameter Insensitivity

### 5.1 Beta and sigma operate in a collapsed effective space

With 50 records, the maximum token frequency is 47. This creates:
- **8 distinct beta behaviors** for 149 candidate integer values (18.6× compression)
- **9 distinct sigma behaviors** for 145 candidate integer values (16.1× compression)

A random draw from beta ∈ [2, 150] × sigma ∈ [6, 150] (with sigma > beta) has:
- 48.5% probability of landing where BOTH beta and sigma are fully saturated (identical to any other saturated config)
- 90.5% probability of sigma being in its no-effect zone (≥ 48)

The vast majority of drawn configurations map to the same or nearly the same algorithmic behavior.

### 5.2 mu is confounded by mu_iterate

As shown in §3.3, the mu parameter in the sweep controls the *starting* threshold of a multi-iteration algorithm. Different starting mu values can produce similar final linkage because:
- The algorithm always terminates at mu≈1.0
- Records that would only match at low mu are typically true matches that would also be caught at high mu (they share many tokens), or they are false positives caught at low mu that inflate false positive counts
- On a small 50-record dataset, the algorithm exhausts unlinked pairs within a few iterations regardless of starting mu

### 5.3 Epsilon is narrowly scoped

With mu_iterate=0.05 and most records linking in the first 1–2 iterations, epsilon only gates a small number of clusters. Furthermore, the entropy quality metric for true-duplicate clusters (which share most tokens) is high and passes easily at any epsilon < 0.8. For contaminated clusters, whether they pass depends on the ratio of shared to unique tokens, which varies unpredictably.

---

## 6. Bugs and Suspicious Implementation Issues

### Bug 1 (CRITICAL): `mu_iterate=0.05` default in `run_sweep()` invalidates sensitivity analysis

**File:** `sweep/sweep_runner.py`, line 259  
**Evidence:** `def run_sweep(..., mu_iterate: float = 0.05, ...)`.  
**Impact:** The sweep is not testing "what is performance at mu=X?" It tests "what is performance after iterating from mu=X to 1.0 in steps of 0.05?" This confounds all mu-sensitivity conclusions. Low starting mu → more iterations → more false-positive accumulation → lower precision. The effect is real but it is an artifact of the iterative design, not of mu itself.

**Fix:** For single-pass sensitivity analysis, call `run_sweep` with `mu_iterate=0.0`. If the iterative behavior is intended, document it explicitly and include `mu_iterate` as a sweep parameter.

### Bug 2 (HIGH): Parameter ranges are grossly oversized for the dataset

**File:** `sweep/parameter_search.py`, lines 14–15  
**Evidence:** `beta_range=(2, 150)`, `sigma_range=(6, 150)`. With S1G.txt having max token frequency 47, any beta ≥ 47 is identical to beta=47. Any sigma ≥ 48 removes nothing.  
**Impact:** ~70% of beta draws and ~71% of sigma draws have no behavioral effect. This wastes compute and dilutes sensitivity estimates with duplicate observations.

**Fix:** Set `beta_range=(2, 50)` and `sigma_range=(6, 50)` for this dataset. Ideally, profile the dataset first (max token frequency) and cap the ranges at 2× the maximum meaningful threshold.

### Bug 3 (HIGH): `mu_iterate` is not a swept parameter but strongly determines outcomes

**File:** `sweep/sweep_runner.py`, lines 259, 319–330.  
**Evidence:** `mu_iterate` defaults to 0.05 and is passed directly to every `run_single_dwm` call without variation. The sweep records it in output CSV but never varies it.  
**Impact:** All 1000 configurations share the same iteration structure. The true effect of `mu` cannot be isolated.

**Fix:** Either fix `mu_iterate=0.0` for the sensitivity study, or sweep over `mu_iterate` as an explicit parameter.

### Bug 4 (MEDIUM): `DWM95_CalculateEntropy.calculateEntropy` mutates its input

**File:** `DWM95_CalculateEntropy.py`, line 33.  
```python
cluster[k].remove(token)  # removes token from cluster[k] in-place
```
**Evidence:** The function removes tokens from the list objects inside `cluster`. Because `DWM90` does `.copy()` (a shallow copy of the list) before appending, `refDict` itself is not corrupted. However, if `calculateEntropy` is ever called with un-copied data or called twice on the same cluster structure, it will produce wrong results silently.  
**Impact:** Currently benign due to the `.copy()` in DWM90, but fragile. A deep copy should be used or the mutation should be eliminated.

### Bug 5 (LOW): Dead code in entropy calculation

**File:** `DWM95_CalculateEntropy.py`, line 43.  
```python
cnt = 0   # this line is never reached before cnt is reset to 1
```
`cnt` is re-initialized to 1 at the top of each `for token in jList` iteration. The reset to 0 at line 43 has no effect. This is dead code.

### Bug 6 (INFORMATIONAL): `DWM99_ERmetrics` reads the truth file starting at line 2

**File:** `DWM99_ERmetrics.py`, lines 32–33.  
```python
line = (truthFile.readline()).strip()   # skips line 1 (header)
line = (truthFile.readline()).strip()   # this is the first data line
```
This correctly skips the header. However, if the truth file has no header or a different format, this silently drops the first data record. This is an assumption embedded in code with no validation.

---

## 7. Evidence Supporting Each Conclusion

| Conclusion | Evidence |
|---|---|
| Max token freq = 47 | Python analysis of S1G.txt: `NC` appears 47 times, no token higher |
| Beta ≥ 47 is redundant | DWM42 condition `freq < 2 or freq > beta`; since no token has freq > 47, any beta ≥ 47 passes all tokens |
| 8 distinct beta behaviors | Enumerated breakpoints: 2, 3, 6, 7, 10, 12, 13, 47 |
| 9 distinct sigma behaviors | Enumerated breakpoints: 6, 7, 8, 11, 13, 14, 32, 34, 48 |
| 71% of sigma range is inert | sigma ≥ 48 removes 0 tokens; range [48,150] = 103/145 values |
| 69.8% of beta range is inert | beta ≥ 47 saturates; range [47,150] = 104/149 values |
| mu_iterate=0.05 is the default | `sweep_runner.py` line 259: `mu_iterate: float = 0.05` |
| Low mu → more iterations | Loop condition `if current_mu > 1.0: more = False`; starting at 0.10 → 19 iterations |
| linkIndex is write-only | DWM90 only appends to linkIndex; no eviction path exists |
| Epsilon cannot undo iteration 1 | DWM90 lines 47–52: epsilon gates current clusters; linked records are not re-blocked |
| Transitive closure amplifies FPs | DWM80 converts pairwise links to equivalence classes; A↔B + A↔C → cluster {A,B,C} |
| High recall is structural | True duplicates share name+address tokens → always share ≥ 2 blocking tokens → always blocked |
| mu_iterate=0.05 confounds mu | A configuration with mu=0.30 links pairs at 0.30, 0.35, ..., 1.00; these links are indistinguishable from those at mu=0.50 |

---

## 8. Parameter Importance Ranking

From most to least impactful on final F1 (for this dataset and sweep design):

| Rank | Parameter | Effect | Why |
|---|---|---|---|
| 1 | `mu` (starting value) | High, but via iteration count | Low starting mu → more false positives accumulated; effect is through iteration count, not threshold |
| 2 | `beta` | Medium (below 47); zero (above 47) | Controls blocking coverage; below 47 determines which token pairs generate candidates |
| 3 | `sigma` | Small (below 48); zero (above 48) | Controls stop-word removal; only 13 distinct behaviors, mostly minor |
| 4 | `epsilon` | Minimal for this dataset | Only gates current-iteration clusters; cannot reach past false positives; dataset is small enough that most clusters have high entropy quality |

**`mu_iterate` (not a swept parameter but the dominant structural variable):** The most impactful factor on outcomes is `mu_iterate`, which is fixed at 0.05 and never varied. Its effect swamps all four swept parameters.

---

## 9. Objective Landscape Characterization

The landscape is **flat over most of the parameter space** for two distinct reasons:

1. **Structural flatness** (beta/sigma): 70%+ of the (beta, sigma) space maps to the same blocking and scoring behavior because the dataset's token frequencies lie well below the parameter upper bounds.

2. **Procedural flatness** (mu): The multi-iteration structure with fixed mu_iterate=0.05 makes all configurations eventually converge to the same final-iteration threshold. The starting mu only determines how many iterations run at low thresholds.

The landscape is **not noisy** — it is genuinely flat. There are no local optima to find. The few regions of sensitivity are at low beta (< 13) and low sigma (< 13), where parameter changes materially alter the token selection logic.

---

## 10. Prioritized Recommendations

### Algorithm Improvements

**A1. Add per-cluster re-evaluation across iterations (HIGH IMPACT)**  
Currently, `linkIndex` is monotonically written — records never leave a cluster once linked. Consider maintaining a "provisional" vs. "confirmed" cluster tier, where provisional clusters can be upgraded, split, or removed based on quality reassessment. This would allow epsilon to undo false positive clusters created in early low-mu iterations.

**A2. Apply transitive closure only within epsilon-passing scope (MEDIUM IMPACT)**  
Currently, transitive closure runs on all linked pairs, then epsilon filters the resulting clusters. Inverting the order — epsilon-filter pairwise links before transitive closure — would prevent false positive pairs from contaminating otherwise-good clusters via transitivity. This is architecturally significant but would directly reduce the false positive amplification described in §4.2.

**A3. Separate blocking token set from scoring stop-word set (LOW IMPACT)**  
The constraint `sigma > beta` forces these two thresholds to be coupled, but their roles are orthogonal. Blocking tokens are selected for their co-occurrence properties; stop words are selected for being too common to discriminate. Decoupling them (separate parameters) would allow more flexible configuration.

### Parameter Improvements

**P1. Cap parameter ranges at dataset-informed bounds (CRITICAL)**  
Before sweeping, profile the dataset to find the maximum token frequency (`max_freq`) and the frequency distribution. Set:
- `beta_range = (2, max_freq)` — upper bound where blocking saturates
- `sigma_range = (6, max_freq + 1)` — upper bound where stop-word removal saturates

For S1G.txt: `beta_range=(2, 47)`, `sigma_range=(6, 48)`. This eliminates the flat zone and makes the sweep informative.

**P2. Fix `mu_iterate=0.0` for sensitivity analysis (CRITICAL)**  
In `sweep/sweep_runner.py`, change the default for sensitivity experiments:
```python
def run_sweep(..., mu_iterate: float = 0.0, ...)
```
This makes each configuration a single-pass evaluation at exactly the specified mu, enabling proper sensitivity measurement. If iterative behavior is desired as a separate experiment, add it as an explicit sweep dimension.

**P3. Sweep `mu_iterate` as an explicit parameter (HIGH IMPACT)**  
Add `mu_iterate` to the sweep as a categorical variable: {0.0, 0.05, 0.10}. This would reveal how much of the observed performance variance is attributable to the iteration structure vs. the starting thresholds.

**P4. Use log-scale or threshold-aware sampling for beta and sigma (MEDIUM IMPACT)**  
Uniform integer sampling over [2, 50] still over-represents the flat zone above the dataset's frequency distribution. Sample from the breakpoints directly:
- Beta breakpoints for this dataset: {2, 3, 6, 7, 10, 12, 13, 20, 30, 47}
- Sigma breakpoints: {6, 7, 8, 11, 13, 14, 32, 34, 48}

### Experimental Improvements

**E1. Profile the dataset before sweeping (CRITICAL)**  
`dataset_profiler.py` already exists. Run it first and use its output to set parameter ranges. Specifically, use `max_freq` to bound beta and sigma, and use token frequency percentiles to identify the meaningful range.

**E2. Use stratified sampling over behavioral zones (HIGH IMPACT)**  
Instead of uniform random sampling (which oversamples flat regions), explicitly enumerate the behavioral breakpoints for beta and sigma, then uniformly sample within those discrete categories. This guarantees coverage of all distinct algorithmic behaviors.

**E3. Run single-iteration baselines (HIGH IMPACT)**  
Run all configurations with `mu_iterate=0.0` first. This establishes clean single-pass baselines where mu is precisely the threshold in effect. Only after establishing single-pass sensitivity should iterative runs (mu_iterate > 0) be introduced.

**E4. Report intermediate metrics per iteration, not just final (MEDIUM IMPACT)**  
Set `runIterationProfile=True` in sweep runs and log precision/recall after each iteration. This would reveal at which iteration precision collapses and when false positives enter, making root cause diagnosis straightforward.

**E5. Consider Bayesian optimization or Optuna after fixing E1–E3 (LOW IMPACT)**  
Bayesian optimization and Optuna are only beneficial when the objective landscape has meaningful structure. Currently the landscape is predominantly flat (§9). Fixing the parameter ranges and mu_iterate first is prerequisite; sophisticated search over a meaningless space will not improve results.

**E6. Run on a larger, more varied dataset (HIGH IMPACT for generalizability)**  
S1G.txt with 50 records is too small for meaningful parameter sensitivity analysis. With 50 records, the maximum token frequency is bounded at 50, collapsing the parameter ranges. Use datasets with 5,000–50,000 records to make beta and sigma operate in their intended ranges. Several other datasets (S2G, S4G, S5G, etc.) appear to be available in the repository.

---

## Summary Table

| Issue | Type | Severity | Location |
|---|---|---|---|
| `mu_iterate=0.05` default invalidates sensitivity analysis | Bug / Design flaw | Critical | `sweep/sweep_runner.py:259` |
| Beta range [2,150] collapses to 8 behaviors on S1G | Design flaw | Critical | `sweep/parameter_search.py:14` |
| Sigma range [6,150] collapses to 9 behaviors on S1G | Design flaw | Critical | `sweep/parameter_search.py:15` |
| 70% of sampled (beta,sigma) space is functionally identical | Consequence | Critical | Dataset × parameter mismatch |
| False positives from early iterations are never evicted | Algorithm limitation | High | `DWM90_IterateClusters.py` |
| Transitive closure amplifies iteration-1 false positives | Algorithm behavior | High | `DWM80_TransitiveClosure.py` |
| `calculateEntropy` mutates its input silently | Bug (currently benign) | Medium | `DWM95_CalculateEntropy.py:33` |
| `mu_iterate` not swept but dominates outcomes | Experimental gap | High | `sweep/sweep_runner.py:259,319` |
| Dead code `cnt = 0` | Code smell | Low | `DWM95_CalculateEntropy.py:43` |
| Truth file skips first data row silently if no header | Latent bug | Low | `DWM99_ERmetrics.py:32-33` |
