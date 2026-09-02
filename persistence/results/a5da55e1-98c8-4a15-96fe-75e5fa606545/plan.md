# Execution plan

## Run 1 · complex · 2026-09-02 13:55 UTC
**Goal:** Summarize MSD biology and screen the ~5k-compound SPECS library to rank repurposing candidates with evidence-backed High/Medium/Low tiers.

- [x] **1.** Build an MSD disease profile (targets, pathways, MoA hypotheses) · @research_agent · `completed` · 2026-09-02 13:57 UTC
      Details: Resolve MSD identifiers (OMIM/Orphanet/DOID/EFO), summarize core biology (SUMF1/FGE, sulfatase network, lysosomal storage, neuroinflammation, ER/lysosome proteostasis), and compile an “actionable” list of intervention nodes (e.g., lysosome biogenesis/TFEB axis, autophagy-lysosome flux, substrate reduction, inflammation modulation, proteostasis/chaperones), with citations from Europe PMC/OpenTargets/Reactome/KEGG. Output `msd_landscape.md` + `msd_actionable_nodes.csv` (node, type, rationale, key refs).
      Depends on: none
      Note: MSD disease profile + 17 actionable nodes delivered (incl SUMF1, TFEB, TRPML1, autophagy/mTOR, substrate reduction, inflammation) with PMIDs; key IDs: Orphanet_585, MONDO_0010088.
- [x] **2.** Ingest and QC the uploaded SPECS + annotation files · @data_agent · `completed` · 2026-09-02 14:04 UTC
      Details: Load `SPECS-library_*.csv` and all `annotations_*.xlsx`; detect encodings/sheets; standardize keys (CHEMBL IDs, compound_name), deduplicate, and produce a unified table of compounds with any available target/pathway/MoA/assay annotations. Output `specs_annotated_unified.parquet` + a data dictionary `inputs_qc_report.md` (row counts, missingness, join coverage).
      Depends on: none
      Note: Unified SPECS+annotations table built: specs_annotated_unified.csv (5159 rows, 41 cols); QC report at inputs_qc_report.md; join coverage: drug info 98.8%, MoA 98.8%, assay 99.1%, target 72.7%, pathway 27.8%.
- [x] **3.** Map SPECS compounds to the MSD profile and generate mechanistic rationales · @research_agent · `completed` · 2026-09-02 14:07 UTC
      Details: Using `msd_actionable_nodes.csv` and the unified annotations, map each compound’s targets/pathways/MoA to MSD-relevant nodes; pull supporting evidence snippets (OpenTargets evidence where available; Europe PMC passages for target↔lysosome/LSD/MSD relevance; Reactome pathway context). Output `specs_msd_matches.csv` (compound, target, pathway, match_type, rationale_draft, evidence_links/citations).
      Depends on: [1], [2]
      Note: MSD node↔compound match tables created from specs_annotated_unified.csv: specs_msd_matches.csv (843 match rows; 800 unique compounds) and specs_msd_matches_high_confidence.csv (direct target matches).
- [ ] **4.** Compute a composite repurposing score and assign High/Medium/Low tiers · @data_agent · `pending`
      Details: Define a transparent scoring rubric (e.g., strength of target-to-disease relevance, mechanism fit to MSD hypotheses, assay/annotation support, polypharmacology risk flags, novelty vs known LSD approaches). Apply it to `specs_msd_matches.csv`, aggregate per compound, and select top candidates (configurable N, default 20–50). Output `msd_candidate_ranked_table.csv` + `scoring_rubric.md`.
      Depends on: [3]
- [ ] **5.** (Optional but recommended) Run in-silico ADMET and CNS/lysosome-relevant filters · @prediction_agent · `pending`
      Details: For shortlisted compounds, resolve/derive SMILES (from ChEMBL IDs) and run the CPSign ADMET panel (CYPs, hERG, Ames, P-gp, PAMPA, BBB, solubility, lipophilicity). Add flags consistent with MSD needs (e.g., CNS penetration if targeting neurodegeneration; safety liabilities). Output `msd_candidates_admet.csv`.
      Depends on: [4]
- [ ] **6.** Assemble the final deliverables (summary + table) · @data_agent · `pending`
      Details: Generate a publication-style table with: compound ID (CHEMBL and any SPECS identifier if present), primary target, mechanistic rationale, supporting evidence (citations/links), confidence tier (High/Medium/Low). Also render the MSD biological landscape summary as a concise narrative. Output `final_msd_repurposing_report.md` + `final_candidates_table.xlsx`.
      Depends on: [4] (and [5] if executed)

_Progress: 3/6 steps resolved._
