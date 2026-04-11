# biperiden (Rank 2 of 73)

## Summary recommendation
This compound ranks **#2** by an equal-weight composite score (**0.807**) integrating: multi-assay antiviral phenotypes (CPE rescue, infection inhibition, morphology rescue), cross-assay consistency, phospholipidosis confounding risk, proxy safety/drug-likeness, clinical readiness from ChEMBL max phase, and mechanistic annotation availability.

**Key decision drivers (data-backed):**
- **Efficacy phenotype:** strong aggregated performance across available assay readouts (see Evidence table and assay plot).
- **Human-cell validation support:** present.
- **PLD risk:** not flagged / lower relative PLD in this set based on the PLD counter-screen.
- **Clinical readiness:** ChEMBL max phase = 4.0.

## Evidence from provided assays (aggregated across concentrations)
| Metric                                   |   Value |
|:-----------------------------------------|--------:|
| Composite score                          |   0.807 |
| CPE inhibition mean (%)                  |  -0.924 |
| CPE viability mean (%)                   |  96.268 |
| Primary CP morphology mean               |   0.48  |
| Primary infection inhibition mean (%)    |  24.5   |
| Validation CP morphology mean            |   0.495 |
| Validation infection inhibition mean (%) |  29.75  |
| PLD 24h DIPL mean (%)                    |  10.458 |

### Visual evidence
![Key assay readouts](figures/biperiden_assay_readouts.png)

## Sub-score profile (0–1; equal weight)
| Sub-score         |   Value |
|:------------------|--------:|
| A_assay_efficacy  |   0.432 |
| B_consistency     |   0.936 |
| C_PLD             |   0.888 |
| D_safety_proxy    |   1     |
| E_clinical        |   1     |
| F_mechanism_proxy |   0.7   |
| G_druglikeness    |   1     |
| H_novelty         |   0.5   |

![Sub-score radar](figures/biperiden_radar.png)

## Mechanistic / annotation context (ChEMBL-derived)
- **Preferred name:** BIPERIDEN
- **MoA (if available):** nan: nan; target=nan
- **Top annotated targets:** Muscarinic acetylcholine receptor M1 | Muscarinic acetylcholine receptor M3 | Muscarinic acetylcholine receptor M2 | Muscarinic acetylcholine receptor M4 | Prostaglandin G/H synthase 1
- **UniProt IDs (if available):** NA

## Confounders & risks (interpretation)
- **Phospholipidosis:** DIPL is a known confounder in SARS-CoV-2 repurposing screens; compounds with strong PLD signals should be treated cautiously and prioritized only if antiviral effects are clearly separable from PLD.
- **Cell-line divergence:** not flagged by the simple heuristic used.

## Suggested next experiments
1. Confirm antiviral potency with full dose–response in A549-ACE2 and a second human airway model (e.g., Calu-3), and compare antiviral window vs cytotoxicity.
2. If PLD-high-risk: demonstrate antiviral effect persists under conditions controlling for lysosomotropic PLD mechanisms (timing, counterscreens, orthogonal readouts).
3. If clinically advanced/approved: evaluate exposure feasibility (lung-relevant concentrations) and drug–drug interaction risk.
