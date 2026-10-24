# medprivacy-screen: per-patient privacy-risk screening with attack-validated rankings and measured mitigation

## Abstract

Privacy audits of machine-learning models usually report a single global
membership-inference (MIA) number. That average can hide the patients who
are actually at risk. We present medprivacy-screen, a pre-deployment
screening tool for medical imaging models that (i) ranks individual
training patients by vulnerability using two independent signals,
(ii) validates the ranking by running real membership-inference attacks and
measuring how concentrated the realized attack success is in the top of
the ranking, and (iii) measures the effect of removing the most vulnerable
patients before deployment. On PneumoniaMNIST with a small CNN the global
attack AUC is 0.507 (loss) / 0.509 (LiRA) — statistically indistinguishable
from chance — while a combined vulnerability ranking places 18.7% of all
realized LiRA identifications in its top 5% (random baseline 5.1%) and
26.6% in its top 10% (baseline 10.0%). That concentration, however, is
carried by one of two target seeds, the two underlying signals disagree
with each other (Spearman ρ = −0.12), and removing the top 10% of flagged
patients does not reliably reduce the identifiability of the patients who
remain (improved in 3 of 8 attack×condition cells). We report the
screening claim as *partly supported, seed-dependent*, and the naive
removal mitigation as *not supported* in this regime. All numbers are
measured locally; protocol, code, and per-patient scores are in this
repository.

## 1. Background and positioning

- Shokri et al. (2017) introduced membership inference; Ye et al. and
  Carlini et al. (LiRA, IEEE S&P 2022) made per-example attacks strong by
  calibrating each example's score against shadow IN/OUT distributions.
- Song & Mittal (USENIX Security 2021) defined a per-sample *privacy risk
  score* and showed risk is heterogeneous across samples. We adopt their
  framing; our screener's signals are computable by a model owner without
  an adversary's knowledge.
- Long et al. (2018/2020) and Felps et al. showed per-point vulnerability
  can be quantified attacker-side.
- Knolle, Kaissis, Rückert et al. (*Nature* 2026, DOI
  10.1038/s41586-026-10688-0; arXiv:2610.30229) audited 7 real clinical
  datasets at the level of individual patients: near-perfect identification
  for some patients, near-chance global averages — the direct motivation
  for a screening tool.
- Audit frameworks (ML Privacy Meter; Meta PrivacyGuard) report global
  metrics per model. None of the above ships the defender workflow built
  here: rank → validate against live attacks (concentration curve) →
  remove → re-measure. `KILLGATE.md` documents the search.

Our findings are deliberately reported as measured, including where they
undercut the screening story: concentration of LiRA success in the top of
the ranking is real on average but seed-dependent; the two standard
vulnerability signals disagree with each other; and the obvious mitigation
— delete the flagged patients and retrain — does not reliably work in
this regime. We regard a tool that can demonstrate *that* on a user's own
model as more valuable than one that only confirms the hopeful story.

## 2. Method

**Data.** PneumoniaMNIST (MedMNIST v2): 28×28 grayscale chest X-rays,
binary label (pneumonia / normal). Train+validation splits are pooled
(5,232 images) as the candidate patient universe; the official test split
(624) is used only for utility measurement. A fixed pool of 3,072 patients
is drawn once (seed 12345).

**Model.** A deliberately ordinary small CNN (~50k parameters: 3 conv
blocks with batch-norm, global average pooling, linear head), Adam
(lr 1e-3), batch 128, 40 epochs. Each target model trains on 1,536 patients
(a random half of the pool); the other half are known non-members. Two
target seeds (0, 1).

**Shadow stage.** 20 shadow models, same recipe. Balanced design: every
pool patient is IN exactly 10 shadows and OUT of 10 (randomized), so each
patient has equally-sized IN/OUT logit samples for the true class.

**Signals** (per patient; higher = more vulnerable):
- `target_loss`: negative loss of the target model on the patient
  (members only; free — no extra training).
- `shadow_gap`: mean true-class logit when IN minus mean when OUT of
  shadow training (memorization gap; independent of the target model).
- `atypicality`: mean loss under OUT shadows (hard/unusual patients).
- `combined`: mean of the rank percentiles of `target_loss` and
  `shadow_gap` within the member set.

**Attacks (validation oracle).**
- Loss-threshold attack: score = −loss under the target model.
- Offline LiRA: per-patient Gaussian fit of shadow IN/OUT logits; score =
  log-likelihood ratio of the target model's logit.
An attack *identifies* a patient at the threshold giving FPR = 10% on
known non-members. Global AUC and TPR@FPR∈{10%, 1%} are also reported.

**Concentration.** For each attack and ranking, members are sorted by the
ranking; we report the share of all identified members falling in the top
1/5/10/20%. The random-ranking baseline (mean ± sd over 200 permutations)
is the uniform-vulnerability null: share ≈ k%.

**Signal agreement.** Spearman correlation between rankings over members,
and each signal's AUC as a predictor of realized attack success.

**Mitigation.** Remove the top 5% / 10% of members by the combined
ranking, retrain from scratch (same recipe/seed), re-run both attacks
(non-members = everyone not trained on, including removed patients), and
re-measure test accuracy.

## 3. Results

All values are from `results.json` (full protocol: pool 3,072; 20
shadows; 40 epochs; 2 target seeds). Runtime 3,038 s wall on a contended
2-core VM.

### 3.1 The global view: nothing to see

| | seed 0 | seed 1 | mean |
|---|---|---|---|
| Test accuracy | 0.851 | 0.784 | 0.817 |
| Loss-attack AUC | 0.502 | 0.513 | 0.507 |
| LiRA AUC | 0.531 | 0.487 | 0.509 |
| LiRA TPR @ FPR 10% | 0.129 | 0.083 | 0.106 |
| LiRA TPR @ FPR 1% | 0.016 | 0.011 | 0.013 |
| Members identified by LiRA @ FPR 10% | 198/1536 | 127/1536 | — |
| Members identified by loss attack @ FPR 10% | 119/1536 | 160/1536 | — |

By the standard global audit these models are private: attack AUC ≈ 0.51.
Note also that identifications at FPR 10% (8–13% of members) sit close to
the 10% rate expected from threshold noise alone, and TPR @ FPR 1%
(1.1–1.6%) barely exceeds its 1% nominal level. At this model/dataset
scale there is no near-perfect per-patient attacker of the kind the
*Nature* 2026 study found on larger clinical models — an honest scale
difference, and a reason per-patient claims must be measured, not
assumed in either direction.

### 3.2 Concentration: real on average, carried by one seed

Share of realized LiRA identifications in the top-k% of each ranking
(permutation baseline in parentheses, ≈ k%):

| ranking | top1 | top5 | top10 | top20 |
|---|---|---|---|---|
| combined (mean) | 0.046 (0.010) | 0.187 (0.051) | 0.266 (0.100) | 0.315 (0.200) |
| shadow_gap (mean) | 0.045 | 0.185 | 0.257 | 0.366 |
| combined, seed 0 | 0.061 | 0.192 | 0.303 | 0.394 |
| shadow_gap, seed 0 | 0.051 | 0.212 | 0.333 | 0.495 |
| combined, seed 1 | 0.031 | 0.181 | 0.228 | 0.236 |
| shadow_gap, seed 1 | 0.039 | 0.157 | 0.181 | 0.236 |
| target_loss, seed 0 / 1 | 0.010/0.008 | 0.066/0.039 | 0.111/0.055 | 0.182/0.126 |
| atypicality, seed 0 / 1 | 0.071/0.039 | 0.207/0.079 | 0.283/0.118 | 0.394/0.126 |

Each signal's AUC as a predictor of realized LiRA success among members
tells the same story: seed 0 — shadow_gap 0.716, atypicality 0.649,
combined 0.632, target_loss 0.468; seed 1 — all four between 0.351 and
0.479, i.e. at or below chance. The concentration effect in seed 0 is
substantial (a third of all LiRA identifications in the top 10% by
shadow_gap; half in the top 20%). In seed 1 it is weak at top-5/10 and
absent at top-20. We therefore claim concentration *conditionally*: it
materialized clearly in one of two seeds, and a screening programme that
does not re-validate per model would not know which case it is in.

For the loss attack, the combined ranking concentrates success at
0.246/0.360 (top5/top10, mean) — but this is partly definitional: the
target_loss component of the ranking is the loss attack's own score, and
its top-10 share against the loss attack is 1.000/0.963 by construction.
The cross-signal numbers above (shadow-based rankings vs LiRA, which
uses independently trained models) are the non-circular evidence.

### 3.3 The signals disagree with each other

Spearman correlation between rankings over members: target_loss vs
shadow_gap ρ = −0.187 (seed 0), −0.044 (seed 1); target_loss vs
atypicality ρ = −0.703 / −0.177; shadow_gap vs atypicality ρ =
+0.403 / +0.374. "Low loss under your model" and "large shadow IN/OUT
gap" are not two views of one quantity here — they rank almost unrelated
patient subsets (the loss-attack and LiRA success sets are likewise
nearly disjoint: shadow_gap's top-10 captures only 0.034/0.125 of
loss-attack success). Any single-signal screening story is therefore
fragile by construction; the combined ranking's robustness in §3.2 comes
precisely from averaging two disagreeing signals.

### 3.4 Mitigation: removing the flagged patients does not reliably work

Identifiable patients among those *kept* (attack re-run after retraining
without the dropped top-k%; "before" counts the same kept patients under
the original model):

| condition | attack | identified on kept: before → after | test acc before → after |
|---|---|---|---|
| seed 0, drop 5% | LiRA | 160 → 140 | 0.851 → 0.705 |
| seed 0, drop 5% | loss | 85 → 81 | |
| seed 0, drop 10% | LiRA | 138 → 163 | 0.851 → 0.873 |
| seed 0, drop 10% | loss | 75 → 91 | |
| seed 1, drop 5% | LiRA | 104 → 124 | 0.784 → 0.753 |
| seed 1, drop 5% | loss | 127 → 123 | |
| seed 1, drop 10% | LiRA | 98 → 91 | 0.784 → 0.846 |
| seed 1, drop 10% | loss | 104 → 115 | |

Identifiability improved in 3 of 8 cells and worsened in 5. The mechanism
is visible in the design: memorization is not a fixed property of a
patient list — retrain without the top decile and the model memorizes
the next tier instead. Accuracy deltas (−14.6, −3.1, +2.2, +6.3 points)
are larger than any plausible direct effect of dropping 77–154 of 1,536
examples and are dominated by retraining variance (single retrain per
condition — itself a methodological warning). The defensible conclusion
is narrow: in this regime, flag-and-delete is not a validated mitigation;
the flagged patients themselves are of course no longer exposed *by this
model* once removed, which may still be the right call for specific
high-risk individuals — but it must not be sold as reducing the model's
overall leakage.

## 4. Discussion

Three lessons survive the mixed results. First, the global-AUC audit and
the per-patient view can both be "quiet" while a ranking still finds
structure: in seed 0, a fifth to a third of realized LiRA success sat in
the top 5–10% of patients a defender can rank *before* deployment, on a
model whose global AUC was 0.53. Second, that structure is not stable:
seed 1, identical in every protocol parameter, showed almost none of it.
Screening is therefore a per-model measurement to repeat, not a property
to assert — which is the entire argument for shipping it as a tool with
the attack validation built in rather than as a one-off study. Third,
mitigation is the weak link: the intuitive fix (delete the risky
patients) redistributes memorization instead of removing it, at least at
this scale and drop rate. Defenders should treat a flagged list as an
input to stronger interventions (DP training, dedup, consent review)
rather than as a deletion list — and should demand the re-measurement
this tool performs before believing any mitigation claim, including
ours.

## 5. Limitations

- Single dataset (PneumoniaMNIST), single small architecture, two target
  seeds; effect sizes will move with dataset, model size, and training
  set size. The protocol, not any single number, is the deliverable.
- One image = one "patient". Real cohorts link multiple records per
  patient; per-patient aggregation (as in the *Nature* 2026 audit) can
  only increase individual risk.
- Validation attacks are standard, not strongest-possible (no online
  LiRA with per-example shadow retraining, no white-box attacks). The
  screener therefore lower-bounds risk: it removes measured leakage, it
  cannot certify privacy. Differential privacy remains the only defense
  with guarantees; screening is the cheap triage before that decision.
- Removed patients are excluded from training entirely — appropriate
  when consent or risk demands it, but it changes the training
  distribution, and the flagged subpopulation may be exactly the
  underrepresented group a clinical model most needs. Flag-and-protect
  alternatives (DP fine-tuning on the flagged subset, stronger
  augmentation, per-example clipping) are future work; this tool measures
  the removal option honestly rather than prescribing it.
- Atypicality alone is a weak proxy for realized attack success here —
  being hard to learn is not the same as being memorized (see Results).

## References

1. Shokri, Stronati, Song, Shmatikov. Membership Inference Attacks
   Against Machine Learning Models. IEEE S&P 2017.
2. Carlini et al. Membership Inference Attacks From First Principles.
   IEEE S&P 2022.
3. Song & Mittal. Systematic Evaluation of Privacy Risks of Machine
   Learning Models. USENIX Security 2021. (arXiv:2003.10595)
4. Long et al. Understanding Membership Inferences on Well-Generalized
   Learning Models. 2020; Felps et al., arXiv:2011.08738.
5. Knolle et al. Patient-level privacy risk in medical AI. *Nature* 2026.
   DOI 10.1038/s41586-026-10688-0. (Preprint: arXiv:2610.30229.)
6. Yang et al. MedMNIST v2. 2023. Dataset: Zenodo record 6496656.
