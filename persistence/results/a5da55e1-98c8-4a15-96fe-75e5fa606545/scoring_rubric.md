# MSD repurposing candidate scoring rubric (SPECS → MSD matches)

## Inputs used
- Match-level evidence: `/app/persistence/results/a5da55e1-98c8-4a15-96fe-75e5fa606545/specs_msd_matches.csv` (n=843 rows)
- High-confidence direct target matches: `/app/persistence/results/a5da55e1-98c8-4a15-96fe-75e5fa606545/specs_msd_matches_high_confidence.csv` (n=38 rows)
- Compound metadata: `/app/persistence/results/a5da55e1-98c8-4a15-96fe-75e5fa606545/specs_annotated_unified.csv` (n=5159 rows)

## Match-level scoring
Each match row (compound ↔ MSD node) is scored as:

**match_score = match_type_weight × node_specificity_weight**

### 1) match_type weights (higher = stronger evidence)
| match_type | weight |
|---|---:|
| target | 5.0 |
| assay | 3.5 |
| moa | 3.0 |
| pathway | 2.0 |
| keyword / other | 0.8 |

Notes:
- Missing/unknown `match_type` is treated as **0 contribution**.
- This dataset contained `match_type` in {pathway, moa, target} only (no assay rows).

### 2) node specificity downweighting
Broad/low-specificity nodes are downweighted; mechanistically proximal lysosome/autophagy nodes are upweighted.

| msd_node_matched | specificity weight |
|---|---:|
| oxidative_stress | 0.25 |
| ER_proteostasis | 0.9 |
| mTORC1 | 1.2 |
| autophagy_flux | 1.1 |
| trehalose_AKT_TFEB | 1.3 |
| TRPML1/MCOLN1 | 1.4 |
| lysosomal_trafficking_M6P | 1.3 |

Any node not listed above receives a default specificity weight of **1.0**.

## Compound-level aggregation
For each compound (`chembl_id`), we compute:
- **total_score**: sum of `match_score` across all its match rows.
- Counts by match type (`n_target`, `n_moa`, `n_pathway`) and node category (`n_specific` = non-oxidative_stress).
- **matched_nodes**: unique list of MSD nodes matched.
- **top match** fields: the single highest-scoring match row (`top_node`, `top_match_type`, `top_match_score`) and its `rationale_draft`/`supporting_evidence`.

## Tier assignment (High / Medium / Low)
Tiering is rule-based to keep it transparent and to avoid treating missing evidence as positive:

### High
Assigned if **any** of the following holds:
1) `n_target ≥ 1` (direct/high-confidence target match exists), OR
2) `n_moa ≥ 1` AND `n_specific ≥ 1` (mechanistic MoA + at least one specific node), OR
3) `total_score ≥ 6` AND `n_specific ≥ 1`.

### Medium
Assigned if **any** of the following holds:
1) `total_score ≥ 2` AND `n_specific ≥ 1`, OR
2) `n_moa ≥ 1` AND `total_score ≥ 1`.

### Low
Everything else (typically dominated by `oxidative_stress` pathway matches only).

## Ranking
Compounds are ranked by:
1) `total_score` (descending)
2) `n_target` (descending)
3) `n_specific` (descending)
4) `max_match_score` (descending)

## Output columns
The ranked table includes identifiers/metadata plus score breakdown fields:
- `chembl_id`, `compound_name`, `canonical_smiles`
- `matched_nodes`, `top_node`, `top_match_type`, `short_rationale`
- `total_score`, `tier`, match-type counts, and per-node score contributions (`score_node_*`).

