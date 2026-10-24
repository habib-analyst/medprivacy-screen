# Launch note — medprivacy-screen (Rule 10 gate, 2026-10-24)
- Novelty/kill-gate: PASS (narrowed claim) — per-sample risk scores (Song & Mittal), per-patient concentration in clinical audits (Nature 2026 / arXiv:2610.30229), and per-example attacks (LiRA) exist separately; no shipped defender-side screening CLI for medical imaging combines 2-signal ranking + attack-validated concentration curve + remove-and-remeasure mitigation. Documented in KILLGATE.md.
- Measured claims only: PASS — 20 shadow models, pool 3,072 PneumoniaMNIST patients, 2 seeds, runtime 3,038 s on CPU; all numbers from results.json.
- Honest wrinkles included: PASS — global attack AUC is near chance (0.507 loss / 0.509 LiRA) and reported as such; the cheap no-shadow signal agrees weakly with the shadow signal (Spearman -0.12) and is reported as weak, not hidden; concentration is enrichment (~2.7x at top decile), not a silver bullet.
- Tests: PASS — 12/12 pytest (re-run by the orchestrator 2026-10-24).
- Cost: PASS — $0, CPU only.
- Secrets/PII: PASS — none; public MedMNIST data only.
- Identity: PASS — author habib-analyst.
- Remaining checklist items: N/A — no payments, forms, or external sends beyond the launch post (standing repo-post authorization).
LAUNCH GATE PASSED 2026-10-24.
