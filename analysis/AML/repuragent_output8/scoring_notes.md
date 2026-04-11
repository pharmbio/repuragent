# Integrated scoring & ranking (AML repurposing candidates)

## Inputs
- `master_candidates_filtered.csv` (n=433; AML-indicated drugs removed; SMILES required)
- ADMET prediction outputs: Solubility (logS), Lipophilicity (logP), PAMPA, PGP, CYP1A2/2C19/2C9/2D6/3A4, hERG, AMES.

## Key integration note (important fix)
ADMET result CSVs in this run **do not contain `chembl_id`**; they are keyed by `smiles` only.
Therefore, all ADMET tables were merged to the master candidate table using **`smiles`**.

## Scoring
Weights (per supervisor guidance):
- Mechanistic plausibility: 0.70
- MoA evidence: 0.15
- ADMET/practical: 0.15

### Mechanistic score (proxy)
Because the filtered master file contains aggregated evidence but not separate evidence-count fields, mechanistic support was proxied by:
- `n_target_genes` (targets linked to AML KG)
- `n_pathways` (AML-relevant pathways linked)

Raw = 1.0*n_target_genes + 0.5*n_pathways.
Normalized by 95th percentile and clipped to [0,1].

### MoA score (proxy)
MoA evidence proxied by `n_moa_terms` (count of MoA terms). Normalized by its 95th percentile and clipped.

### ADMET score
Started at 1.0 then:
- penalties: AMES(+): -0.35, hERG(+): -0.35, each CYP inhibition flag: -0.05, PGP: -0.05
- bonuses: PAMPA: +0.05
- small continuous adjustments: logS scaled from [-6,0], logP penalty if outside ~[1,3]
Clipped to [0,1].

### Final score
`final_score = 0.70*mechanism_score + 0.15*moa_score + 0.15*admet_score`

## MOLM13 assay concentration series
No potency prediction was performed. A standard 10-point series was assigned heuristically by mechanism_score quartiles:
- very_high: 0.3 nM → 0.3 µM
- high: 1 nM → 1 µM
- medium: 3 nM → 3 µM
- exploratory: 10 nM → 10 µM

## Outputs
- `master_ranked.csv`
- `top20_candidates.csv`
- `final_score_distribution.png`
- `mechanism_vs_admet.png`
