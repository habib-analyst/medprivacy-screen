# KILLGATE — medprivacy-screen (written BEFORE building, 2026-10-24)

## Claim under test
A defender-side, pre-deployment **screening tool** for medical imaging models that
(1) ranks individual training patients by membership-inference (MIA) vulnerability using
>=2 independent signals, (2) **validates the ranking against live attacks** and reports a
**concentration curve** (share of realized attack success sitting in the top-k% of the
ranking), and (3) **closes the loop**: remove/flag the top-k%, retrain, and re-measure
attack success vs accuracy cost. Packaged as a CLI/library, demonstrated on MedMNIST.

## Prior art found (web + GitHub search, 2026-10-24)

| Work | What it does | Why it does not cover the claim |
|---|---|---|
| Song & Mittal, "Systematic Evaluation of Privacy Risks of ML Models", USENIX Security 2021 (arXiv:2003.10595) | Introduces a per-sample **privacy risk score** (likelihood-ratio style) and shows risk is heterogeneous. | General (non-medical) benchmarks; attacker/auditor analysis paper, not a defender screening CLI for medical imaging; no attack-validated concentration curve as the deliverable; no remove-and-remeasure mitigation validation. |
| Knolle, Kaissis, Rückert et al., Nature 2026, DOI 10.1038/s41586-026-10688-0 (and arXiv:2610.30229, the inspiration) | Patient-level privacy **audit study** on 7 real clinical datasets; some patients identifiable with near-perfect accuracy while global average MIA looks near chance. | A study, not a reusable screening tool a practitioner runs on their own dataset/model before deployment; no packaged mitigation loop (drop top-k% -> retrain -> re-measure attack success vs accuracy cost); not a MedMNIST screener. |
| Long et al. 2018/2020; Felps et al., "Bootstrap Aggregation for Point-based Generalized MIA" (arXiv:2011.08738) | Generalized/point-based MIA: quantifies per-point attack vulnerability. | Attacker-side research on general data; no medical screening workflow, no concentration-curve reporting for defenders, no mitigation validation. |
| Carlini et al., "Membership Inference Attacks From First Principles" (LiRA), IEEE S&P 2022 | Per-example attack using shadow IN/OUT Gaussian likelihood ratios. | An attack, not a screening product; we use a LiRA-style attack only as the *validation oracle* for the screener's ranking. |
| ML Privacy Meter (privacytrustlab) / Meta PrivacyGuard (arXiv:2510.23427) | Privacy auditing frameworks; report global attack metrics (AUC/accuracy) per model. | Model-level reporting; per-patient screening + concentration curve + mitigation re-measurement is not their deliverable; not medical-imaging specific. |
| "Free Record-Level Privacy Risk Evaluation Through Artifact-Based Methods" (LT-IQR, arXiv:2411.05743) | Per-sample vulnerability estimate from training loss traces at ~zero extra cost. | Single signal class (loss-trace dynamics), general benchmarks; no second independent signal, no attack-validated concentration curve, no mitigation loop, no medical data. |
| PSS-MIA (arXiv:2606.30081) | Attacker-side pre-query sample selection (Loss-Gap Ranking) + TAPC coverage metric to spend query budget efficiently. | Attacker efficiency, the opposite direction: it helps an attacker choose victims; it does not screen-and-mitigate for a defender. |
| "Membership Inference Attacks Beyond Overfitting" (arXiv:2511.16792) | Characterizes *why* some samples are vulnerable (outliers far from class centroid); discusses defenses. | Analysis paper; no shipped screening tool, no validated concentration curve + mitigation tradeoff on medical imaging. |

## Verdict: PROCEED (narrowed claim)
The ingredients exist separately in the literature: per-sample risk scores
(Song & Mittal), per-patient concentration of risk in real clinical data
(Nature 2026 / arXiv:2610.30229), per-example attacks (LiRA). What is **not**
shipped anywhere we could find is the combination as a defender's tool:
a pre-deployment screening CLI for medical imaging that ranks patients with
two independent signals (one needing no shadow models), validates its own
ranking against live attacks with a concentration curve, and validates the
mitigation (remove top-k% -> retrain -> attack success vs accuracy cost).
All write-ups must credit Song & Mittal for the risk-score concept and the
Nature 2026 study for demonstrating per-patient concentration clinically;
our contribution is the validated screening + mitigation workflow and fresh
measurements on MedMNIST. If the concentration does NOT reproduce in our
runs, we publish that negative result instead — no fabricated claims.
