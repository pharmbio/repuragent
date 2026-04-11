# Step 4 – Ranking + top-20 extraction (mechanism-prioritized)

## Inputs
- merged_candidates.csv (n=601)
- ADMET prediction outputs (n=494 with SMILES)

## Integration
- Joined ADMET outputs to candidates by `smiles_canonical` (left join)
- 107/601 candidates had missing SMILES → no ADMET predictions; ADMET subscore uses neutral defaults where missing.

## Scoring (weights per supervisor)
- Mechanistic plausibility: 0.50
  - proxy: log1p(protein_genes) + 0.5*log1p(pathways) + 0.75*log1p(moa_terms) + 0.5*evidence_sources_count
- Experimental support: 0.20
  - proxy: evidence_sources_count + presence of known_targets_chembl / known_mechanisms_chembl
- Clinical/safety status: 0.15
  - proxy: preferred_name_chembl/trade_names_chembl; penalize aml_known_drug_flag (repurposing novelty)
- ADMET: 0.10
  - solubility (higher better), logP preference (centered at 2), penalty for predicted hERG/AMES/CYP/PGP positives, PAMPA
- Tractability: 0.05
  - proxy: solubility + SMILES availability

## Outputs
- master_ranked.csv: full ranked list (n=601)
- top_20_candidates.csv: top 20 repurposing candidates (AML-known drugs excluded) with suggested IC50 dilution series
- top30_repurposing_barplot.png

## Notes / limitations
- The AML-known-drug exclusion is based on `aml_known_drug_flag` from KG known_drugs.csv, not regulatory label.
- Experimental support is a proxy; MOLM13-specific evidence requires literature extraction in the report step.
- IC50 concentration series are *experimentally testable starting ranges* (10-point 3-fold). Adjust top concentration if solubility/toxicity limits observed.
