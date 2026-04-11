# Integration, normalization, ranking, and confounder-flagging notes
## Inputs
- compound annotations: `/home/repuragent/app/persistence/data/7b87215a-cefd-46b5-be7b-ab162989611b/df4a7195-237e-46e3-a2e7-dde9aa2e9f59/compound_annotations_20260130_222855.csv` (n=5159)
- CP/AB raw: `/home/repuragent/app/persistence/data/7b87215a-cefd-46b5-be7b-ab162989611b/df4a7195-237e-46e3-a2e7-dde9aa2e9f59/CP_AB_raw_20260130_222855.xlsx` (n=5275)
- CPE raw: `/home/repuragent/app/persistence/data/7b87215a-cefd-46b5-be7b-ab162989611b/df4a7195-237e-46e3-a2e7-dde9aa2e9f59/CPE_raw_20260130_222855.xlsx` (n=26471)

## Identifier harmonization
- Primary integration key: `chembl_id` from annotation file.
- Raw assay files contained `Compound_name`; we mapped to `chembl_id` by lowercased/trimmed name exact match.
- Simple alternate key (remove non-alphanumerics) was attempted for unmatched names; remaining unmatched assay rows were excluded from per-chembl summaries but all annotated compounds remain in final ranking.
- Unmapped fraction (CP/AB): 0.0040; (CPE): 0.0079.

## Assay directionality
- Cell Painting (CP) morphology score: higher = better rescue.
- Antibody-based (AB) infection rate %: lower = better antiviral; converted to activity by negating infection rate before normalization.
- CPE inhibition of cytopathicity %: higher = better rescue.
- Non-infected cell viability %: lower indicates cytotoxicity; converted to *toxicity* metric by negating viability before normalization (so higher toxicity_z = more toxic).

## Normalization
- Because plate/replicate metadata were not present, normalization was performed *within each tested concentration* using a robust z-score: `z = 0.6745*(x - median)/MAD` (fallback to standard z if MAD=0).
- CP/AB appear to be single concentration (10 µM) for most compounds; CPE covers a dose range (nM–µM).

## Per-compound aggregation
- For each compound (chembl_id), we computed per-assay summary statistics across concentrations: max and mean of the normalized z-scores.
- For ranking we used `z_max` as a liberal “best observed activity” summary (keeps dose-response positives from being averaged away).

## Composite score and ranking
- Composite score uses weighted average of available `z_max` values:
  - AB activity (`ab_activity_z_max`) weight 0.4
  - CPE activity (`cpe_activity_z_max`) weight 0.4
  - CP morphology (`cp_morph_z_max`) weight 0.2 (downweighted because typically single concentration and morphology rescue can be non-antiviral)
- If an assay is missing for a compound, weights are renormalized over the assays present.
- All compounds in the annotation list are ranked (n=5159). Compounds missing all assay data receive the lowest scores.

## Confounder flags (potential false positives)
Flags are heuristic and intended for review, not exclusion:
- `flag_ab_toxicity`: strong AB activity (ab_activity_z_max ≥ 1.5) **and** high toxicity (toxicity_z_max ≥ 1.5) → reduced infection may be driven by cell loss.
- `flag_cpe_toxicity`: strong CPE rescue and high toxicity → possible assay interference / non-specific effects; also indicates narrow therapeutic window.
- `flag_cp_only`: strong CP morphology rescue but no AB/CPE support → morphology rescue may be unrelated to viral suppression.
- `flag_ab_no_cpe`: AB strong but mean CPE negative → inconsistent antiviral vs cytoprotection readouts; warrants follow-up.

## Outputs
- Ranked integrated table: `/home/repuragent/app/persistence/results/7b87215a-cefd-46b5-be7b-ab162989611b/df4a7195-237e-46e3-a2e7-dde9aa2e9f59/integrated_ranked_compounds.csv`
- Confounder flags table: `/home/repuragent/app/persistence/results/7b87215a-cefd-46b5-be7b-ab162989611b/df4a7195-237e-46e3-a2e7-dde9aa2e9f59/confounder_flags.csv`
