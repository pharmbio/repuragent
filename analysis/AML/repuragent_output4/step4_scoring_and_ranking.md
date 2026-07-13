# Step 4 — Integrated scoring, full ranking, top-20 extraction, IC50 concentration series (AML / MOLM13)

## Inputs
- master_dataset.csv (n=519)
- ADMET model outputs (CYPs, hERG, AMES, PGP, PAMPA, BBB, Solubility). Lipophilicity model failed (see below).

## ADMET merge coverage
- ADMET predictions successfully merged for **424 / 519** compounds (matched on canonical_smiles).
- **95** compounds had missing ADMET predictions across most endpoints (likely missing/invalid SMILES).
- Missing list saved: `admet_missing_compounds.csv`

## Scoring
### Mechanistic evidence score (prioritized)
- Raw score: `3*n_protein_links + 1*n_pathway_links + 2*n_moa_links`
- Normalized to 0–1 (`mech_score`) across dataset.

### ADMET risk score (penalty)
- Components (higher = worse):
  - hERG (weight 2)
  - AMES (weight 1)
  - P-gp (weight 0.5)
  - CYP inhibition mean across 5 isoforms (weight 0.3)
  - PAMPA low permeability risk: (1 - pampa) (weight 0.5)
  - Solubility risk from predicted logS: <=-4 high (1.0), -4 to -2 medium (0.5), >-2 low (0.0) (weight 0.7)
  - Missing ADMET flag adds 0.2
- Normalized to 0–1 (`admet_risk`).

### Combined score (for ranking)
- **combined_score = 0.85*mech_score - 0.15*admet_risk**

## IC50 assay concentration recommendations (MOLM13)
- Default: **10-point 3-fold dilution series**.
- Top concentration rule:
  - If predicted **logS <= -4** OR **PAMPA=0** OR **PGP=1**, set top concentration to **30 µM** (to avoid missing activity due to exposure/solubility constraints).
  - Otherwise top concentration **10 µM**.
- The exported column `ic50_series_uM` lists 10 concentrations (µM) from top down by 3-fold dilution.

## Outputs
- `ADMET_predictions.csv`
- `ranked_master_list.csv` (full ranking)
- `top20_candidates.csv` (top 20 + IC50 series)
- Plots: `mech_score_distribution.png`, `combined_score_distribution.png`

## Known limitations
- Lipophilicity model failed due to RDKit NoneType mol (likely invalid SMILES among inputs). Lipophilicity not included in scoring.
- ADMET model outputs appear to contain only ~424 predictions; remaining ~95 candidates missing predictions. These are flagged and penalized via `admet_missing`.
