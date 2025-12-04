# Acknowledgement

Repuragent relies on several open-source scientific tools, data, and guidelines. This page summarizes these resources and links to the original sources for proper acknowledgment.


## REMEDI4ALL Standard Operating Procedures

The Standard Operating Procedures used in our system are provided by REMEDi4ALL. Visit [REMEDi4ALL](https://remedi4all.org) home page for more details.

## REMEDI4ALL Chemical Annotator

For compound annotations, we use the
[REMEDI4ALL Chemical Annotator](https://github.com/REMEDI4ALL/chemical_annotator), which
queries ChEMBL, UniChem, PubChem, and KEGG from SMILES/InChI inputs.


## Knowledge Graph Generator (KGG)

We rely on the [Knowledge Graph Generator (KGG)](https://github.com/Fraunhofer-ITMP/kgg)
from Fraunhofer ITMP to create disease-specific knowledge graphs and extract information from them.


## LitSense

Literature grounding relies on [LitSense](https://academic.oup.com/nar/article/53/W1/W361/8133630),
a PubMed semantic search engine. LitSense indexes titles, abstracts, and full text where available, and combines their
semantic representations.

## Hugging Face Local Python Executor

The `python_executor` tool in the data agent was built on the [Smolagents](https://huggingface.co/docs/smolagents/index) Python executor. It helps keep code running safely scoped to a curated import list.

## Unstructured

Our SOP RAG system relies on [Unstructured](https://github.com/Unstructured-IO/unstructured) for parsing PDF files. Unstructured processes text, tables, and images before chunking and embedding them with the OpenAI embedding model, which is then stored in ChromaDB for semantic search later.


## CPSign package and TDC data

Predictive models were trained and evaluated using [CPSign](https://cpsign.readthedocs.io/en/latest/). The data for training the models was downloaded from [Therapeutics Data Commons](https://tdcommons.ai)