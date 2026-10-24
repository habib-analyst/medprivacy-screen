# medprivacy-screen

**Per-patient privacy-risk screening for medical imaging models.**

Standard membership-inference (MIA) audits report **one global number** — an
attack AUC averaged over every patient in the training set. Work by Knolle,
Kaissis, Rückert and colleagues (*Nature* 2026, DOI
10.1038/s41586-026-10688-0; arXiv:2610.30229) showed what that average hides:
in real clinical datasets, a small subset of patients can be identified with
near-perfect accuracy while the global average looks like random guessing.

`medprivacy-screen` turns that finding into a defender's tool you run
**before deployment**, on your own training set:

1. **Rank** every training patient by MIA vulnerability, using two
   independent signals (plus a third, reference-only signal):
   - *target loss/confidence* — how memorized the example is by your model
     (free: no extra training);
   - *shadow gap* — mean true-class logit when the example was IN vs OUT of
     training, across 20 shadow models (never touches your model);
   - *atypicality* — how badly models that never saw the example do on it.
2. **Validate the ranking against live attacks.** The tool actually runs a
   loss-threshold attack and an offline-LiRA attack against your model, then
   reports the **concentration curve**: what share of the attack's *realized*
   identifications sit in the top 1/5/10/20% of the ranking — against a
   permutation (random-ranking) baseline.
3. **Measure the mitigation.** Drop the top 5%/10% most vulnerable patients,
   retrain, and re-run both attacks: how many fewer patients are identifiable,
   and what did it cost in test accuracy?

The per-sample risk-score idea is Song & Mittal's (USENIX Security 2021);
the per-patient concentration phenomenon in clinical data is the *Nature*
2026 study's. The contribution here is the packaged, self-validating
screening + mitigation workflow, and the measurements below. See
`KILLGATE.md` for the full prior-art analysis.

## Results (PneumoniaMNIST, 2 target seeds, this repo's `results.json`)

The headline is a **mixed, partly negative result** — which is exactly what
the tool exists to surface before you trust a screening story:

| Quantity | Value |
|---|---|
| Global attack AUC — loss / LiRA (mean of 2 seeds) | **0.507 / 0.509** (≈ chance) |
| Target test accuracy (mean) | 0.817 (0.851, 0.784) |
| Members identified by LiRA @ FPR 10% | 198/1536 (12.9%), 127/1536 (8.3%) |
| Share of LiRA identifications in top 5% (combined ranking) | **0.187** (random baseline 0.051) |
| Share of LiRA identifications in top 10% (combined ranking) | **0.266** (random baseline 0.100) |
| Signal agreement (Spearman, target-loss vs shadow-gap) | **−0.115** (the signals disagree) |
| Mitigation drop-10%: identifiable kept patients (LiRA) | 138→163 (seed 0, *worse*), 98→91 (seed 1) |
| Mitigation drop-10%: test accuracy change | +2.2 pts (seed 0), +6.3 pts (seed 1) |

Reading: the global audit says "safe" (AUC ≈ 0.51). A combined ranking
does concentrate LiRA identifications ~2.7–3.7× above the random baseline
on average — but the effect is carried by one of two seeds (shadow-gap
signal AUC for realized LiRA success: 0.716 vs 0.379), the two standard
vulnerability signals barely correlate with each other, and removing the
flagged top 5–10% **does not reliably reduce** how identifiable the
remaining patients are after retraining (identifiability improved in 3 of
8 attack×condition cells, worsened in 5; accuracy deltas of −14.6 to
+6.3 points are dominated by retraining variance). Full per-seed tables
and the reasoning are in `PAPER.md`; per-patient scores and flagged lists
are in `outputs/`.

## Quickstart

```bash
pip install -r requirements.txt
# PneumoniaMNIST (MedMNIST v2) — or point --data at any MedMNIST .npz
curl -L -o data/pneumoniamnist.npz \
  "https://zenodo.org/record/6496656/files/pneumoniamnist.npz?download=1"

python -m medprivacy.cli screen --data data/pneumoniamnist.npz \
    --out outputs --results results.json      # full protocol (~25-50 min on 2 CPU cores)
python -m medprivacy.cli report --results results.json
python run_experiments.py --quick             # 1-minute smoke config
pytest                                        # 12 tests
```

## Honest limits

- **One dataset, one small architecture.** All numbers are PneumoniaMNIST
  (28×28 chest X-rays, binary) with a ~50k-parameter CNN trained on 1,536
  patients per target model — the small-data regime where memorization is
  plausible, not a hospital-scale deployment.
- MedMNIST images stand in for "patients" (one image per patient record);
  real EHR/imaging cohorts have multiple records per patient, which can
  raise per-patient risk further.
- The attacks used for validation (loss threshold, offline LiRA) are
  standard but not the strongest possible; a stronger attacker could
  identify more patients. Screening removes *measured* risk, it does not
  certify privacy — only differential privacy gives guarantees.
- Dropping flagged patients changes the training distribution; the accuracy
  cost measured here is specific to this dataset and drop rate.
- Shadow stage cost scales linearly in the number of shadow models
  (20 here); the target-loss signal needs no shadows and is the cheap
  first pass.
- **Seed variance is large.** With two target seeds, one showed clear
  concentration and one showed almost none. Treat any single-run screening
  result — including ours — as a measurement with error bars, not a fact.
- **Circularity caveat:** the loss attack's score *is* the target loss, so
  the target-loss ranking "predicting" loss-attack success (share 1.000 at
  top-10%) is definitional, not validation. The meaningful validations are
  cross-signal ones (shadow-gap / combined rankings vs LiRA success).
- **Removal is not a reliable mitigation in this regime** (see Results):
  after retraining, memorization redistributes onto the kept patients.
  Screening tells you *who* is at risk; it does not follow that deleting
  them fixes the model.

## Layout

```
medprivacy/   data.py, model.py, signals.py, attacks.py, screen.py,
              experiment.py, cli.py
tests/        unit tests for signals, attacks, concentration + smoke test
KILLGATE.md   prior-art search and the narrowed novelty claim
PAPER.md      method and full measured results
results.json  machine-readable headline numbers
```
