# Inputs QC report

## Sheets used
- **targets_info** (annotations_targets_info_20260902_135420.xlsx): used sheet(s): Sheet1
- **pathway_info** (annotations_pathway_info_20260902_135420.xlsx): used sheet(s): Sheet1
- **drugs_moa** (annotations_drugs_moa_20260902_135420.xlsx): used sheet(s): Sheet1
- **drugs_info** (annotations_drugs_info_20260902_135419.xlsx): used sheet(s): Sheet1
- **drugs_assay** (annotations_drugs_assay_20260902_135419.xlsx): used sheet(s): Sheet1
- **assay_targets_info** (annotations_drugs_assay_targets_info_20260902_135419.xlsx): used sheet(s): Sheet1

## Per-input row/ID QC
|input|rows|unique chembl_id (or target_chembl_id)|missing chembl_id %|duplicate rows by id|
|---|---:|---:|---:|---:|
|SPECS|5159|5158|0.019|0|
|targets_info|4097|4096|0.024|0|
|pathway_info|4097|4096|0.024|0|
|drugs_moa|5613|5097|0.267|515|
|drugs_info|25778|5112|0.000|20666|
|drugs_assay|95798|5112|0.000|90686|
|assay_targets_info|95798|5112|0.000|90686|

## Join coverage on SPECS (chembl_id)
SPECS total rows: 5,159; unique chembl_id: 5,158
|annotation type|SPECS compounds with >=1 record|% of SPECS|
|---|---:|---:|
|SPECS_with_drug_info|5097|98.80|
|SPECS_with_moa|5095|98.76|
|SPECS_with_assay|5112|99.09|
|SPECS_with_any_target_info|3752|72.73|
|SPECS_with_any_pathway_info|1434|27.80|

## Unified table schema (data dictionary)
Rows: 5,159; Columns: 41

|column|dtype|non-null|description/source|
|---|---|---:|---|
|specs_compound_name|object|5159|SPECS-library CSV|
|chembl_id|object|5158|standardized ChEMBL ID used for joins|
|pref_name|object|3351|annotations_drugs_info|
|synonyms|object|2964|annotations_drugs_info (aggregated)|
|max_phase|float64|2941|annotations_drugs_info|
|first_approval|object|1248|annotations_drugs_info|
|therapeutic_flag|float64|5097|annotations_drugs_info|
|withdrawn_flag|float64|5097|annotations_drugs_info|
|canonical_smiles|object|5097|annotations_drugs_info|
|standard_inchi_key|object|5097|annotations_drugs_info|
|full_mwt|float64|5097|annotations_drugs_info|
|alogp|float64|5018|annotations_drugs_info|
|hba|float64|5018|annotations_drugs_info|
|hbd|float64|5018|annotations_drugs_info|
|aromatic_rings|float64|5018|annotations_drugs_info|
|num_ro5_violations|float64|5018|annotations_drugs_info|
|indication_class|float64|0|annotations_drugs_info (aggregated)|
|drug_schembl|object|5008|annotations_drugs_info (aggregated)|
|drug_cid|object|5112|annotations_drugs_info (aggregated)|
|n_moa_rows|float64|5095|annotations_drugs_moa (count rows per compound)|
|moa_actions|object|1305|annotations_drugs_moa (aggregated)|
|moa_text|object|1414|annotations_drugs_moa (aggregated)|
|moa_target_chembl_ids|object|1305|annotations_drugs_moa (aggregated target_chembl_id)|
|n_assay_rows|float64|5112|annotations_drugs_assay (count activity rows)|
|n_unique_assays|float64|5112|annotations_drugs_assay|
|n_unique_targets_assayed|float64|5112|annotations_drugs_assay|
|best_pchembl|float64|3517|annotations_drugs_assay (max pChEMBL)|
|median_pchembl|float64|3517|annotations_drugs_assay (median pChEMBL)|
|has_drug_info|bool|5159|derived boolean|
|has_moa|bool|5159|derived boolean|
|has_assay|bool|5159|derived boolean|
|n_unique_targets|float64|3752|targets from MoA+assay joined to targets_info|
|target_chembl_ids|object|3752|targets from MoA+assay|
|target_descriptions|object|3625|annotations_targets_info (aggregated)|
|target_uniprot_ids|object|3625|annotations_targets_info (aggregated)|
|n_unique_pathways|float64|3752|pathways from targets joined to pathway_info|
|pathways|object|1434|annotations_pathway_info (aggregated)|
|kegg_ids|object|1434|annotations_pathway_info|
|ec_numbers|object|2394|annotations_pathway_info|
|has_target_info|bool|5159|derived boolean|
|has_pathway_info|bool|5159|derived boolean|