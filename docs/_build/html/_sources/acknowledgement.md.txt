# Acknowledgement

Repuragent depends on a handful of open-source scientific tools and data. This page summarizes how we use
them and points to the original sources for attribution.

## Knowledge Graph Generator (KGG)

We rely on the [Knowledge Graph Generator (KGG)](https://github.com/Fraunhofer-ITMP/kgg)
from Fraunhofer ITMP to create disease-specific knowledge graphs and extract information from it. Its
automated pipeline ingests curated sources (OpenTargets, UniProt, ChEMBL, etc.) and lets us
pick the disease ID, protein threshold, and clinical-stage filters.

## REMEDI4ALL Chemical Annotator

For compound annotations we use the
[REMEDI4ALL Chemical Annotator](https://github.com/REMEDI4ALL/chemical_annotator), which
queries ChEMBL, UniChem, PubChem, and KEGG from SMILES/InChI inputs.

## REMEDI4ALL Technical Note

Our SOPs are from the REMEDi4ALL technical notes, for example 

## LitSense

Literature grounding relies on [LitSense](https://academic.oup.com/nar/article/53/W1/W361/8133630),
a PubMed-scale neural search engine described by Salatino et al. (Nucleic Acids Research,
2025). LitSense indexes titles, abstracts, and full text where available, and combines
semantic representations with curated entity tagging to surface mechanism-, target-, and
phenotype-level passages that drive Repuragent's research agent prompts.

## CPSign

Predictive modeling runs through
[CPSign](https://cpsign.readthedocs.io/en/latest/), which provides conformal QSAR models,
Signatures/ECFP/CDK descriptors, and LIBLINEAR/LIBSVM back ends on the JVM. We follow the
official CLI/API guidance for training and deployment and direct users to the CPSign docs
for advanced topics such as encryption, explainability, and Serve publishing.
