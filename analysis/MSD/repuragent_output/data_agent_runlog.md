# MSD repurposing screen — Data Agent run log

## Objective
Run the canonical MSD repurposing screen on the uploaded SPECS library (~5k compounds) using the upstream annotation XLSX files, producing a 22-column output with High/Medium/Low tiers.

## Resolved inputs
- SPECS_CSV: `/home/repuragent/app/persistence/data/c8deab08-f02a-482f-a59b-fe0c7e0f5f36/e41790f4-f00e-4edb-90a5-da84ba49d953/SPECS-library_20260326_102746.csv`
- ANNOT_DIR: `/home/repuragent/app/persistence/results/c8deab08-f02a-482f-a59b-fe0c7e0f5f36/e41790f4-f00e-4edb-90a5-da84ba49d953`
- KG_PKL: `/home/repuragent/app/persistence/results/c8deab08-f02a-482f-a59b-fe0c7e0f5f36/e41790f4-f00e-4edb-90a5-da84ba49d953/kg_Orphanet_585.pkl`

## Knowledge graph quick landscape (Step 0.5)
- Nodes: 35 | Edges: 44
- Proteins in KG: SUMF1, SUMF2, ARSA, ARSB
- Processes/pathways observed include autophagy, lysosomal transport/organization, glycosphingolipid catabolism, and Reactome pathways around arylsulfatase activation.

## Annotation files used
- annotations_drugs_assay.xlsx
- annotations_drugs_assay_targets_info.xlsx
- annotations_drugs_info.xlsx
- annotations_drugs_moa.xlsx
- annotations_pathway_info.xlsx
- annotations_targets_info.xlsx

## Data shapes
- specs: (5159, 2)
- drugs_info: (25778, 38)
- drugs_moa: (5613, 7)
- assay_targets: (95798, 39)

## Primary target coverage
- Compounds with primary_target assigned (non-null): 3517
- Without primary_target: 1642

## Tier hits (regex-based)
- Tier1 hits: 94
- Tier2 hits: 103
- Tier3 hits: 412

## Confidence tiers
| confidence_tier   |   count |
|:------------------|--------:|
| Low               |    4963 |
| Medium            |     115 |
| High              |      81 |

## Output
- OUTPUT_FILE: `/home/repuragent/app/persistence/results/c8deab08-f02a-482f-a59b-fe0c7e0f5f36/e41790f4-f00e-4edb-90a5-da84ba49d953/MSD_repurposing_screen_all.csv`
- Output shape: (5159, 22) (must be (5159, 22))
- Row count equals SPECS compounds: 5159 == 5159

## Notes / fixes applied
- The initial `determine_primary_target` function produced extremely verbose prints due to tool logging; no data were changed, only continued execution.
- SPECS has 1 missing chembl_id; retained as a row with NaNs in annotations.

