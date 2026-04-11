# Analysis documentation — Morphology rescue + DIPL-aware ranking

Date: 2026-02-01

## Inputs
- validatation_experiements_20260201_123124.xlsx (morphology score; antibody-based infection readout)
- phospholipidosis_raw_20260201_123124.xlsx (24h DIPL %; classification)
- compound_annotations_20260201_123124.csv (chembl_id ↔ compound_name)

## Key dataset caveats discovered
- Validation dataset contains **no plate/batch identifier** and includes only **one DMSO row**; therefore plate-wise robust normalization vs DMSO is not possible. Global mean/std z-scores were computed for reference, but the ranking is driven primarily by raw morphology score thresholds.
- Validation measurements appear to be **single wells per compound-concentration** (n≈1), limiting reproducibility estimates and statistical testing.
- Phospholipidosis dataset contains **no explicit DMSO wells**; per-batch (batch_ID) median was used as a baseline for robust-z scoring.

## Toxicity filtering
- A concentration was flagged toxic if cell_count < 80% of the DMSO cell count in validation (SOP-aligned). Since DMSO n=1, DMSO cell_count=100 was used.

## Morphology rescue metric
- We inferred that **higher morphology_score indicates rescue** because morphology_score correlates strongly negatively with infection_rate (r≈-0.73).
- Robust rescue call (pragmatic, given dataset limits): morphology_score ≥ 0.5 at a non-toxic concentration.
- Best concentration: lowest non-toxic concentration with morphology_score ≥ 0.5; if none, concentration with maximum morphology_score.

## Phospholipidosis (DIPL) metric
- DIPL was interpolated (linear on log10 concentration) to estimate DIPL_value and DIPL_robust_z at the morphology-effective concentration.
- DIPL robust-z baseline: per-batch median and MAD across all wells in the batch.

## Scoring and ranking
- E_m: min–max scaled best morphology_score across compounds (0–1).
- R_d: scaled DIPL_value at best concentration (5th–95th percentile scaling; clipped 0–1).
- Final score: S = E_m * (1 - 0.7*R_d), with a 0.2 multiplier if toxic at best concentration.
- DIPL-driven flag: robust_rescue_boolean AND DIPL_robust_z_at_best_conc ≥ 3.

## Confounders reported
See confounders_report.csv for dataset-level and compound-level flags.

## Outputs
- ranking_output.csv: final ranked list (all compounds)
- normalized_data.csv: per-well validation rows with interpolated DIPL estimates
- confounders_report.csv: dataset/plate/compound risk flags
