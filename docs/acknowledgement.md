# Acknowledgement

Repuragent depends on a handful of open-source scientific tools and data. This page summarizes how we use
them and points to the original sources for attribution.

## Knowledge Graph Generator (KGG)

We rely on the [Knowledge Graph Generator (KGG)](https://github.com/Fraunhofer-ITMP/kgg)
from Fraunhofer ITMP to create disease-specific knowledge graphs and extract information from them.

## REMEDI4ALL Chemical Annotator

For compound annotations, we use the
[REMEDI4ALL Chemical Annotator](https://github.com/REMEDI4ALL/chemical_annotator), which
queries ChEMBL, UniChem, PubChem, and KEGG from SMILES/InChI inputs.

## REMEDI4ALL Standard Operating Procedures

The Standard Operating Procedures used in our system are provided by REMEDi4ALL. Visit [REMEDi4ALL](https://remedi4all.org) home apge for more details.

## LitSense

Literature grounding relies on [LitSense](https://academic.oup.com/nar/article/53/W1/W361/8133630),
a PubMed-scale neural search engine described by Salatino et al. (Nucleic Acids Research,
2025). LitSense indexes titles, abstracts, and full text where available, and combines
semantic representations with curated entity tagging to surface mechanism-, target-, and
phenotype-level passages that drive Repuragent's research agent prompts.

## Hugging Face Local Python Executor

The `python_executor` in the data agent reuses Hugging Face's Apache-2.0 licensed local Python executor to keep code runs safely scoped
to a curated import list.

## Unstructured

Our SOP RAG system depends on [Unstructured](https://github.com/Unstructured-IO/unstructured) for PDF parsing plus OpenAI's embedding models. Unstructured handles table/image-aware chunking, while ChatGPT/Embeddings power the summaries and Chroma vectors that back the SOP RAG experience. 

## CPSign

Predictive models were trained and evaluated using [CPSign](https://cpsign.readthedocs.io/en/latest/).