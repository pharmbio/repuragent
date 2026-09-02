import asyncio
import json
import math
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import httpx
import pandas as pd
import pybel
import requests
from chembl_webresource_client.new_client import new_client
from langchain_core.tools import tool
from pybel.dsl import Abundance, BiologicalProcess
from backend.utils.cancellation import cancellable_tqdm as tqdm  # every tqdm loop is a Stop checkpoint

from app.config import logger
from backend.kgg.api_utils import createKG, searchDisease
from backend.utils.output_paths import (
    resolve_output_folder,
    task_file_path,
)
from backend.utils.storage_paths import get_data_root

DATA_ROOT = get_data_root()


def _output_dir():
    '''Resolve the writable directory for KGG artifacts.

    Returns:
    ----------
    directory (Path): the writable directory for this conversation's KGG artifacts.
    '''

    return resolve_output_folder()


def _output_file(filename: str):
    '''Build a file path scoped to the active task directory.

    Parameters:
    ---------
    filename (str): the file name to place inside the active task directory.

    Returns:
    ----------
    path (Path): the scoped absolute path.
    '''

    return task_file_path(filename)


# ========== Tool 1: Disease Search ==========

@tool
def search_disease_id(disease_name: str) -> Dict[str, Any]:
    '''Search for disease IDs (EFO/MONDO) using disease name.
    
    Parameters:
    ---------
    disease_name (str): Human-readable disease name (e.g., "Alzheimer's disease")

    Returns:
    ----------
    result (dict): Standardized dictionary with:
    - success (bool): Whether the operation succeeded
    - data (dict): Contains 'matches' list and 'top_match_id'
    - message (str): Human-readable operation result
    - metadata (dict): Query parameters and result counts
    '''

    try:
        results = searchDisease(disease_name)
        if results is not None and not results.empty:
            # Return top matches
            matches = []
            for _, row in results.head(5).iterrows():
                matches.append({
                    'id': row['id'],
                    'name': row['name'],
                    'description': row.get('description', '')
                })

            return {
                "success": True,
                "data": {
                    "matches": matches,
                },
                "message": f"Found {len(matches)} disease matches for '{disease_name}'",
                "metadata": {
                    "query": disease_name,
                    "total_matches": len(results),
                    "returned_matches": len(matches)
                }
            }
        else:
            return {
                "success": False,
                "data": None,
                "message": f"No matching diseases found for '{disease_name}'",
                "metadata": {"query": disease_name, "total_matches": 0}
            }
    except Exception as e:
        logger.error(f"Error searching disease: {e}")
        return {
            "success": False,
            "data": None,
            "message": f"Error searching for disease '{disease_name}': {str(e)}",
            "metadata": {"query": disease_name, "error_type": type(e).__name__}
        }


# ========== Tool 2: Create Knowledge Graph ==========

@tool 
def create_knowledge_graph(
    disease_id: str, 
    clinical_trial_phase: int = 3,
    protein_threshold: float = 0.3,
) -> Dict[str, Any]:
    '''Create a disease-specific knowledge graph using KGG.
    This is the FIRST step - create the graph before extracting information.
    
    Parameters:
    ---------
    disease_id (str): disease ID (provided id or first match id from search)
    clinical_trial_phase (int): Minimum clinical trial phase (1-4)
    protein_threshold (float): Minimum protein association score (0.0-1.0)

    Returns:
    ----------
    result (dict): Standardized dictionary with:
    - success (bool): Whether the knowledge graph was created successfully
    - data (dict): Contains graph statistics (nodes, edges, etc.)
    - output_file (str): Path to the saved knowledge graph file
    - message (str): Human-readable operation result
    - metadata (dict): Input parameters and creation details
    '''

    try:
        logger.info(f"Creating KG for {disease_id}, phase={clinical_trial_phase}, threshold={protein_threshold}")

        # Create the knowledge graph
        kg = createKG(
            disease_id=disease_id,
            clinical_trial_phase=clinical_trial_phase,
            protein_threshold=protein_threshold
        )

        if kg is None:
            return {
                "success": False,
                "data": None,
                "output_file": None,
                "message": f"Failed to create knowledge graph for disease '{disease_id}'",
                "metadata": {
                    "disease_id": disease_id,
                    "clinical_trial_phase": clinical_trial_phase,
                    "protein_threshold": protein_threshold,
                    "error_reason": "KG creation returned None"
                }
            }

        import pickle
        output_dir = _output_dir()
        kg_path = output_dir / f"kg_{disease_id.replace(':', '_')}.pkl"

        with open(kg_path, 'wb') as f:
            pickle.dump(kg, f)

        # Calculate graph statistics
        num_nodes = len(kg.nodes())
        num_edges = len(kg.edges())

        return {
            "success": True,
            "data": {
                "disease_id": disease_id,
                "graph_statistics": {
                    "nodes": num_nodes,
                    "edges": num_edges,
                }
            },
            "output_file": str(kg_path),
            "message": f"Knowledge graph created successfully for '{disease_id}' with {num_nodes} nodes and {num_edges} edges",
            "metadata": {
                "disease_id": disease_id,
                "clinical_trial_phase": clinical_trial_phase,
                "protein_threshold": protein_threshold,
                "file_path": str(kg_path),
            }
        }

    except Exception as e:
        logger.error(f"Error creating knowledge graph: {e}")
        return {
            "success": False,
            "data": None,
            "output_file": None,
            "message": f"Error creating knowledge graph for '{disease_id}': {str(e)}",
            "metadata": {
                "disease_id": disease_id,
                "clinical_trial_phase": clinical_trial_phase,
                "protein_threshold": protein_threshold,
                "error_type": type(e).__name__
            }
        }


# ========== Tool 3: Extract Drugs from KG ==========

@tool
def extract_drugs_from_kg(kg_path: str) -> Dict[str, Any]:
    '''Extract drug information from a previously created knowledge graph.
    
    Parameters:
    ---------
    kg_path (str): Path to the saved knowledge graph

    Returns:
    ----------
    result (dict): Standardized dictionary with:
    - success (bool): Whether drug extraction succeeded
    - data (dict): Contains extracted drug information
    - output_file (str): Path to the CSV file with drug data
    - message (str): Human-readable operation result
    - metadata (dict): Extraction parameters and statistics
    '''

    try:
        # Load the knowledge graph
        import pickle
        with open(kg_path, 'rb') as f:
            kg = pickle.load(f)

        drugs = []

        for node in kg.nodes():
            if isinstance(node, pybel.dsl.Abundance) and node.namespace == 'ChEMBL':
                attrs = kg.nodes[node]
                drug_info = {
                    'chembl_id': node.name,
                    'preferred_name': attrs.get('PreferredName', ''),
                    'trade_names': attrs.get('TradeName', []),
                    'url': attrs.get('ChEMBL', '')
                }

                # Find related information (targets, mechanisms)
                drug_targets = []
                drug_mechanisms = []

                for source, target, data in kg.edges(data=True):
                    if source == node:
                        if isinstance(target, pybel.dsl.Protein):
                            drug_targets.append(str(target))
                        elif isinstance(target, pybel.dsl.BiologicalProcess) and target.namespace == 'MOA':
                            drug_mechanisms.append(str(target))

                drug_info['targets'] = drug_targets[:5]  # Limit targets
                drug_info['mechanisms'] = drug_mechanisms[:3]  # Limit mechanisms

                drugs.append(drug_info)

        # Convert drugs list to pandas DataFrame and export as CSV
        if drugs:
            # Create DataFrame from drugs list
            df = pd.DataFrame(drugs)

            # Handle list columns by converting them to string representation
            if 'trade_names' in df.columns:
                df['trade_names'] = df['trade_names'].apply(lambda x: '; '.join(x) if isinstance(x, list) else x)
            if 'targets' in df.columns:
                df['targets'] = df['targets'].apply(lambda x: '; '.join(x) if isinstance(x, list) else x)
            if 'mechanisms' in df.columns:
                df['mechanisms'] = df['mechanisms'].apply(lambda x: '; '.join(x) if isinstance(x, list) else x)

            output_dir = _output_dir()
            csv_path = output_dir / "known_drugs.csv"
            df.to_csv(csv_path, index=False)
            csv_str = str(csv_path)

            return {
                "success": True,
                "data": {
                    "summary": {
                        "total_drugs": len(drugs),
                        "showing_in_data": min(len(drugs), 10),
                        "data_truncated": len(drugs) > 10,
                        "complete_data_location": csv_str
                    },
                    "analysis_recommendation": f"For complete analysis, use the full dataset at {csv_str} which contains all {len(drugs)} drugs"
                },
                "output_file": csv_str,
                "message": f"Successfully extracted {len(drugs)} known drugs. Showing 10 sample records in response data, complete dataset saved to {csv_str}",
                "metadata": {
                    "kg_path": kg_path,
                    "drugs_found": len(drugs),
                    "csv_exported": csv_str,
                    "data_completeness": "sample_only_use_csv_for_full_analysis"
                }
            }
        else:
            return {
                "success": True,
                "data": {
                    "summary": {
                        "total_drugs": 0,
                        "showing_in_data": 0,
                        "data_truncated": False,
                        "complete_data_location": None
                    },
                    "analysis_recommendation": "No drugs found in the knowledge graph - consider different extraction parameters or verify the knowledge graph contains drug information"
                },
                "output_file": None,
                "message": "No drugs found in the knowledge graph",
                "metadata": {
                    "kg_path": kg_path,
                    "drugs_found": 0,
                }
            }

    except FileNotFoundError:
        return {
            "success": False,
            "data": None,
            "output_file": None,
            "message": f"Error: Knowledge graph file not found at path '{kg_path}'",
            "metadata": {
                "kg_path": kg_path,
                "error_type": "FileNotFoundError"
            }
        }
    except Exception as e:
        logger.error(f"Error extracting drugs: {e}")
        return {
            "success": False,
            "data": None,
            "output_file": None,
            "message": f"Error extracting drugs from knowledge graph: {str(e)}",
            "metadata": {
                "kg_path": kg_path,
                "error_type": type(e).__name__
            }
        }


# ========== Tool 3: Extract Mechanism of Action from KG ==========
@tool
def extract_mechanism_of_actions_from_kg(kg_path: str) -> Dict[str, Any]:
    '''Extract mechanism of actions from a knowledge graph.
    
    Parameters:
    ---------
    kg_path (str): Path to the saved knowledge graph

    Returns:
    ----------
    result (dict): Dictionary containing side effect information
    '''

    import pickle
    with open(kg_path, 'rb') as f:
        kg = pickle.load(f)

    rows = []

    for u, v, data in kg.edges(data=True):
        # normalize direction: identify which side is drug, which is MOA
        a, b = u, v
        if isinstance(a, BiologicalProcess) and getattr(a, "namespace", None) == "MOA" \
           and isinstance(b, Abundance) and getattr(b, "namespace", None) == "ChEMBL":
            moa_node, drug_node = a, b
        elif isinstance(b, BiologicalProcess) and getattr(b, "namespace", None) == "MOA" \
             and isinstance(a, Abundance) and getattr(a, "namespace", None) == "ChEMBL":
            moa_node, drug_node = b, a
        else:
            continue

        rows.append({
            "chembl_id": drug_node.name,
            "drug_name": kg.nodes.get(drug_node, {}).get("PreferredName"),
            "mechanism_of_action": moa_node.name,
        })

    if not rows:
        return pd.DataFrame(columns=["chembl_id", "drug_name", "mechanism_of_action"])

    df = pd.DataFrame(rows).drop_duplicates().sort_values(["chembl_id", "mechanism_of_action"]).reset_index(drop=True)

    csv_path = _output_file("mechanism_of_actions.csv")
    df.to_csv(csv_path, index=False)
    csv_str = str(csv_path)

    return {
        "success": True,
        "data": {
            "summary": {
                "total_mechanism_of_actions": len(df['mechanism_of_action'].value_counts()),
                "complete_data_location": csv_str
            },
            "analysis_recommendation": f"For complete analysis, use the full dataset at {csv_str} which contains all {len(df['mechanism_of_action'].value_counts())} mechanism of actions"
        },
        "output_file": csv_str,
        "message": f"Successfully extracted {len(df['mechanism_of_action'].value_counts())} side effects.",
        "metadata": {
            "kg_path": kg_path,
            "mechanism_of_actions_found": len(df['mechanism_of_action'].value_counts()),
            "csv_exported": csv_str
                }
            }

# ========== Tool 4: Extract Proteins from KG ==========

@tool
def extract_proteins_from_kg(
    kg_path: str,
    druggable_only: bool = False,
) -> Dict[str, Any]:
    '''Extract protein/target information from a previously created knowledge graph.
    
    Parameters:
    ---------
    kg_path (str): Path to the saved knowledge graph
    druggable_only (boolean): If True, only return druggable proteins

    Returns:
    ----------
    result (dict): Standardized dictionary with:
    - success (bool): Whether protein extraction succeeded
    - data (dict): Contains extracted protein information
    - output_file (str): Path to the CSV file with protein data
    - message (str): Human-readable operation result
    - metadata (dict): Extraction parameters and statistics
    '''

    try:
        # Load the knowledge graph
        import pickle
        with open(kg_path, 'rb') as f:
            kg = pickle.load(f)

        # Initialize lists for building DataFrame
        protein_info = {
            'gene_symbol': [],
            'druggability': [],
            'uniprot_url': [],
            'opentargets_url': [],
            'disease_association_score': []
        }

        proteins_processed = 0

        for node in kg.nodes():
            if isinstance(node, pybel.dsl.Protein) and node.namespace == 'HGNC':
                attrs = kg.nodes[node]

                druggability = attrs.get('Druggability', 'Unknown')

                # Skip if filtering for druggable only
                if druggable_only and druggability in ['No', 'Unknown']:
                    continue

                # Add information to lists
                protein_info['gene_symbol'].append(node.name)
                protein_info['druggability'].append(druggability)
                protein_info['uniprot_url'].append(attrs.get('UniProt', ''))
                protein_info['opentargets_url'].append(attrs.get('OpenTargets', ''))
                protein_info['disease_association_score'].append(attrs.get('DiseaseAssociationScore'))

                proteins_processed += 1


        # Create pandas DataFrame
        df = pd.DataFrame(protein_info)

        output_path = _output_file('associated_genes.csv')
        df.to_csv(output_path, index=False)
        output_str = str(output_path)

        return {
            "success": True,
            "data": {
                "summary": {
                    "total_proteins": len(df),
                    "showing_in_data": min(len(df), 10),
                    "data_truncated": len(df) > 10,
                    "complete_data_location": output_str,
                    "druggable_filter_applied": druggable_only
                },
                "analysis_recommendation": f"For complete analysis, use the full dataset at {output_str} which contains all {len(df)} proteins"
            },
            "output_file": output_str,
            "message": f"Successfully extracted {len(df)} proteins. Showing {min(len(df), 10)} sample records in response data, complete dataset saved to {output_str}",
            "metadata": {
                "kg_path": kg_path,
                "druggable_only": druggable_only,
                "total_proteins": len(df),
                "csv_exported": output_str,
            }
        }

    except FileNotFoundError:
        return {
            "success": False,
            "data": None,
            "output_file": None,
            "message": f"Error: Knowledge graph file not found at path '{kg_path}'",
            "metadata": {
                "kg_path": kg_path,
                "error_type": "FileNotFoundError"
            }
        }
    except Exception as e:
        logger.error(f"Error extracting proteins: {e}")
        return {
            "success": False,
            "data": None,
            "output_file": None,
            "message": f"Error extracting proteins from knowledge graph: {str(e)}",
            "metadata": {
                "kg_path": kg_path,
                "error_type": type(e).__name__
            }
        }


# ========== Tool 6: Extract Pathways from KG ==========

@tool
def extract_pathways_from_kg(kg_path: str) -> Dict[str, Any]:
    '''Extract pathway information from a knowledge graph.
    
    Parameters:
    ---------
    kg_path (str): Path to the saved knowledge graph

    Returns:
    ----------
    result (dict): Dictionary containing pathway information
    '''

    try:
        # Load the knowledge graph
        import pickle
        with open(kg_path, 'rb') as f:
            kg = pickle.load(f)

        pathways = []

        for node in kg.nodes():
            if isinstance(node, pybel.dsl.BiologicalProcess) and node.namespace == 'Reactome':
                attrs = kg.nodes[node]

                pathway_info = {
                    'name': node.name,
                    'url': attrs.get('Reactome', ''),
                    'associated_proteins': [],
                    'associated_drugs': []
                }

                # Find associated proteins and drugs
                for source, target, data in kg.edges(data=True):
                    if target == node:
                        if isinstance(source, pybel.dsl.Protein):
                            pathway_info['associated_proteins'].append(str(source))
                        elif isinstance(source, pybel.dsl.Abundance) and source.namespace == 'ChEMBL':
                            pathway_info['associated_drugs'].append(str(source))

                pathways.append(pathway_info)

        # Convert pathways list to pandas DataFrame and export as CSV
        if pathways:
            # Create DataFrame from pathways list
            df = pd.DataFrame(pathways)

            # Handle list columns by converting them to string representation
            if 'associated_proteins' in df.columns:
                df['associated_proteins'] = df['associated_proteins'].apply(lambda x: '; '.join(x) if isinstance(x, list) else x)
            if 'associated_drugs' in df.columns:
                df['associated_drugs'] = df['associated_drugs'].apply(lambda x: '; '.join(x) if isinstance(x, list) else x)

            csv_path = _output_file("pathways.csv")
            df.to_csv(csv_path, index=False)
            csv_str = str(csv_path)

            return {
                "success": True,
                "data": {
                    "summary": {
                        "total_pathways": len(pathways),
                        "showing_in_data": min(len(pathways), 10),
                        "data_truncated": len(pathways) > 10,
                        "complete_data_location": csv_str
                    },
                    "analysis_recommendation": f"For complete analysis, use the full dataset at {csv_str} which contains all {len(pathways)} pathways"
                },
                "output_file": csv_str,
                "message": f"Successfully extracted {len(pathways)} pathways. Showing {min(len(pathways), 10)} sample records in response data, complete dataset saved to {csv_str}",
                "metadata": {
                    "kg_path": kg_path,
                    "pathways_found": len(pathways),
                    "csv_exported": csv_str,
                }
            }
        else:
            return {
                "success": True,
                "data": {
                    "summary": {
                        "total_pathways": 0,
                        "showing_in_data": 0,
                        "data_truncated": False,
                        "complete_data_location": None
                    },
                    "analysis_recommendation": "No pathways found in the knowledge graph - consider different extraction parameters or verify the knowledge graph contains pathway information"
                },
                "output_file": None,
                "message": "No pathways found in knowledge graph",
                "metadata": {
                    "kg_path": kg_path,
                    "pathways_found": 0,
                    "extraction_time": datetime.now().isoformat()
                }
            }

    except FileNotFoundError:
        return {
            "success": False,
            "data": None,
            "output_file": None,
            "message": f"Error: Knowledge graph file not found at path '{kg_path}'",
            "metadata": {
                "kg_path": kg_path,
                "error_type": "FileNotFoundError"
            }
        }
    except Exception as e:
        logger.error(f"Error extracting pathways: {e}")
        return {
            "success": False,
            "data": None,
            "output_file": None,
            "message": f"Error extracting pathways from knowledge graph: {str(e)}",
            "metadata": {
                "kg_path": kg_path,
                "error_type": type(e).__name__
            }
        }


# ========== Tool 7: Extract Side Effects from KG ==========

@tool
def extract_side_effects_from_kg(
    kg_path: str,
    drug_id: Optional[str] = None,
) -> Dict[str, Any]:
    '''Extract side effect information from a knowledge graph.
    
    Parameters:
    ---------
    kg_path (str): Path to the saved knowledge graph
    drug_id (str): Optional ChEMBL ID to filter side effects for specific drug

    Returns:
    ----------
    result (dict): Dictionary containing side effect information
    '''

    try:
        # Load the knowledge graph
        import pickle
        with open(kg_path, 'rb') as f:
            kg = pickle.load(f)

        side_effects = []

        for source, target, data in kg.edges(data=True):
            if isinstance(target, pybel.dsl.Pathology) and target.namespace == 'SideEffect':
                if isinstance(source, pybel.dsl.Abundance) and source.namespace == 'ChEMBL':

                    # Apply filter if specified
                    if drug_id and source.name != drug_id:
                        continue

                    effect_info = {
                        'drug': source.name,
                        'drug_name': kg.nodes[source].get('PreferredName', source.name),
                        'side_effect': target.name,
                        'relation': data.get('relation', 'hasSideEffect')
                    }

                    side_effects.append(effect_info)

        # Convert side_effects list to pandas DataFrame and export as CSV
        if side_effects:
            # Create DataFrame from side_effects list
            df = pd.DataFrame(side_effects)

            csv_path = _output_file("side_effects.csv")
            df.to_csv(csv_path, index=False)
            csv_str = str(csv_path)

            return {
                "success": True,
                "data": {
                    "summary": {
                        "total_side_effects": len(side_effects),
                        "showing_in_data": min(len(side_effects), 10),
                        "data_truncated": len(side_effects) > 10,
                        "complete_data_location": csv_str,
                        "drug_filter_applied": drug_id is not None
                    },
                    "analysis_recommendation": f"For complete analysis, use the full dataset at {csv_str} which contains all {len(side_effects)} side effects"
                },
                "output_file": csv_str,
                "message": f"Successfully extracted {len(side_effects)} side effects. Showing {min(len(side_effects), 10)} sample records in response data, complete dataset saved to {csv_str}",
                "metadata": {
                    "kg_path": kg_path,
                    "filter_drug": drug_id,
                    "side_effects_found": len(side_effects),
                    "csv_exported": csv_str,
                    "extraction_time": datetime.now().isoformat()
                }
            }
        else:
            return {
                "success": True,
                "data": {
                    "summary": {
                        "total_side_effects": 0,
                        "showing_in_data": 0,
                        "data_truncated": False,
                        "complete_data_location": None,
                        "drug_filter_applied": drug_id is not None
                    },
                    "drugs_with_side_effects": [],
                    "analysis_recommendation": f"No side effects found in the knowledge graph{' for the specified drug' if drug_id else ''} - consider different extraction parameters or verify the knowledge graph contains side effect information"
                },
                "output_file": None,
                "message": "No side effects found in knowledge graph",
                "metadata": {
                    "kg_path": kg_path,
                    "filter_drug": drug_id,
                    "side_effects_found": 0,
                    "extraction_time": datetime.now().isoformat()
                }
            }

    except FileNotFoundError:
        return {
            "success": False,
            "data": None,
            "output_file": None,
            "message": f"Error: Knowledge graph file not found at path '{kg_path}'",
            "metadata": {
                "kg_path": kg_path,
                "error_type": "FileNotFoundError"
            }
        }
    except Exception as e:
        logger.error(f"Error extracting side effects: {e}")
        return {
            "success": False,
            "data": None,
            "output_file": None,
            "message": f"Error extracting side effects from knowledge graph: {str(e)}",
            "metadata": {
                "kg_path": kg_path,
                "error_type": type(e).__name__
            }
        }

CHEMBL_BATCH_URL = "https://www.ebi.ac.uk/chembl/api/data/molecule.json"
CHEMBL_MOLECULE_SEARCH_URL = "https://www.ebi.ac.uk/chembl/api/data/molecule/search"
OPENFDA_EVENT_ENDPOINT = "https://api.fda.gov/drug/event.json"

HTTPX_TIMEOUT = httpx.Timeout(60.0, read=60.0, connect=30.0)
OPENFDA_MAX_CONCURRENCY = 6
CHEMBL_MAX_CONCURRENCY = 6
RETRY_BACKOFF_SECONDS = 2.0


def fetch_smiles_batch(chembl_ids, chunk_size=200):
    '''Batch-fetch canonical_smiles for many ChEMBL IDs using a persistent HTTP client.
    Returns {chembl_id: smiles or None}

    Parameters:
    ---------
    chembl_ids (list): ChEMBL molecule identifiers to look up.
    chunk_size (int): how many identifiers to request per call, to stay inside the API's limits.

    Returns:
    ----------
    smiles (dict): ChEMBL identifier mapped to its canonical SMILES, omitting the ones the API had none for.
    '''

    chembl_ids = [cid for cid in dict.fromkeys(chembl_ids)]  # dedupe, keep order
    if not chembl_ids:
        return {}

    out = {cid: None for cid in chembl_ids}

    with httpx.Client(timeout=HTTPX_TIMEOUT) as client:
        for i in tqdm(range(0, len(chembl_ids), chunk_size), desc="ChEMBL batch"):
            chunk = chembl_ids[i:i + chunk_size]
            params = {
                "molecule_chembl_id__in": ",".join(chunk),
                "limit": 1000,
                "format": "json",
            }
            response = client.get(CHEMBL_BATCH_URL, params=params)
            response.raise_for_status()
            payload = response.json()

            for molecule in payload.get("molecules", []):
                cid = molecule.get("molecule_chembl_id")
                smiles = (molecule.get("molecule_structures") or {}).get("canonical_smiles")
                if cid in out and smiles:
                    out[cid] = smiles

            next_url = (payload.get("page_meta") or {}).get("next")
            while next_url:
                response = client.get(next_url)
                response.raise_for_status()
                payload = response.json()
                for molecule in payload.get("molecules", []):
                    cid = molecule.get("molecule_chembl_id")
                    smiles = (molecule.get("molecule_structures") or {}).get("canonical_smiles")
                    if cid in out and smiles:
                        out[cid] = smiles
                next_url = (payload.get("page_meta") or {}).get("next")

    return out


_chembl_lookup_cache: Dict[str, Dict[str, Optional[str]]] = {}


def _normalize_match_string(value: str) -> str:
    cleaned = re.sub(r"[^A-Z0-9]", "", value.upper())
    return cleaned


def _generate_drug_query_variants(drug_term: str) -> List[str]:
    base = (drug_term or "").strip()
    if not base:
        return []

    variants: List[str] = []
    def _add_variant(candidate: str) -> None:
        candidate = candidate.strip()
        if candidate and candidate not in variants:
            variants.append(candidate)

    _add_variant(base)

    without_paren = re.split(r"[\\[(]", base)[0]
    _add_variant(without_paren)

    tokens = [tok for tok in re.split(r"\\s+", without_paren) if tok]
    if tokens:
        _add_variant(tokens[0])
    if len(tokens) >= 2:
        _add_variant(" ".join(tokens[:2]))

    stripped = re.sub(
        r"\\b(extended|delayed|immediate|modified|controlled|sustained|release|tablet|tablets|tab|tabs|capsule|capsules|cap|caps|injection|solution|suspension|lotion|cream|pen|kit|patch|spray|drops|syrup|syringe|ophthalmic|topical|oral|intravenous|subcutaneous|intramuscular|nasal|gel|ointment|powder|vial|mg|ml|xr|sr|er|dr|cr|xl|iv|sc)\\b",
        "",
        without_paren,
        flags=re.IGNORECASE,
    )
    stripped = re.sub(r"\\b\\d+[A-Z/%]*\\b", "", stripped, flags=re.IGNORECASE)
    stripped = re.sub(r"\\s+", " ", stripped)
    _add_variant(stripped)

    return [variant for variant in variants if variant]


async def _retrieve_chembl_match(
    drug_term: str,
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
) -> Optional[Dict[str, Optional[str]]]:
    cache_key = drug_term.upper()
    cached = _chembl_lookup_cache.get(cache_key)
    if cached is not None:
        return cached

    variants = _generate_drug_query_variants(drug_term)
    target_norm = _normalize_match_string(drug_term) if drug_term else ""

    for variant in variants:
        try:
            async with semaphore:
                response = await client.get(
                    CHEMBL_MOLECULE_SEARCH_URL,
                    params={"q": variant, "format": "json", "limit": 25},
                )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            logger.debug("ChEMBL search failed for '%s': %s", variant, exc)
            await asyncio.sleep(RETRY_BACKOFF_SECONDS)
            continue

        payload = response.json()
        molecules = payload.get("molecules") or []
        if not molecules:
            continue

        for mol in molecules:
            pref_name = (mol.get("pref_name") or "").strip()
            candidates = [pref_name] if pref_name else []
            for syn in mol.get("molecule_synonyms") or []:
                syn_val = syn.get("molecule_synonym")
                if syn_val:
                    candidates.append(syn_val.strip())
            normalized_candidates = {_normalize_match_string(name) for name in candidates if name}
            if target_norm and target_norm in normalized_candidates:
                result = {
                    "chembl_id": mol.get("molecule_chembl_id"),
                    "preferred_name": pref_name or mol.get("molecule_chembl_id"),
                    "match_type": "exact_name",
                }
                _chembl_lookup_cache[cache_key] = result
                return result

        for mol in molecules:
            pref_name = (mol.get("pref_name") or "").strip()
            candidates = [pref_name] if pref_name else []
            for syn in mol.get("molecule_synonyms") or []:
                syn_val = syn.get("molecule_synonym")
                if syn_val:
                    candidates.append(syn_val.strip())
            normalized_candidates = [_normalize_match_string(name) for name in candidates if name]
            if any(target_norm and target_norm in cand for cand in normalized_candidates):
                result = {
                    "chembl_id": mol.get("molecule_chembl_id"),
                    "preferred_name": pref_name or mol.get("molecule_chembl_id"),
                    "match_type": "partial_name",
                }
                _chembl_lookup_cache[cache_key] = result
                return result

        fallback = next((mol for mol in molecules if mol.get("pref_name")), molecules[0])
        result = {
            "chembl_id": fallback.get("molecule_chembl_id"),
            "preferred_name": (fallback.get("pref_name") or fallback.get("molecule_chembl_id")),
            "match_type": "search_top",
        }
        _chembl_lookup_cache[cache_key] = result
        return result

    result = {
        "chembl_id": None,
        "preferred_name": None,
        "match_type": None,
    }
    _chembl_lookup_cache[cache_key] = result
    return result


REACTOME_SEARCH_URL = "https://reactome.org/ContentService/search/query"
REACTOME_BROWSER_BASE = "https://reactome.org/PathwayBrowser/#/"
OPENTARGETS_GRAPHQL_URL = "https://api.platform.opentargets.org/api/v4/graphql"

KNOWN_DRUGS_ROWS_QUERY = """
    query DrugAndClinicalCandidatesQuery($ensgId: String!) {
      target(ensemblId: $ensgId) {
        id
        drugAndClinicalCandidates {
          count
          rows {
            id
            maxClinicalStage
            drug { id name }
            diseases { disease { id name } }
            clinicalReports { id source clinicalStage phaseFromSource }
          }
        }
      }
    }
"""

ASSOCIATED_DISEASES_QUERY = """
    query AssociatedDiseases($ensgId: String!, $index: Int!, $size: Int!) {
      target(ensemblId: $ensgId) {
        associatedDiseases(page: {index: $index, size: $size}) {
          count
          rows {
            score
            disease { id }
          }
        }
      }
    }
"""

MAP_TARGET_IDS_QUERY = """
    query MapTargets($terms: [String!]!) {
      mapIds(queryTerms: $terms, entityNames: ["target"]) {
        mappings {
          term
          hits {
            id
            name
            entity
          }
        }
      }
    }
"""

_target_association_score_cache: Dict[Tuple[str, str], Optional[float]] = {}


def _opentargets_graphql(query: str, variables: Dict[str, Any]) -> Dict[str, Any]:
    response = requests.post(
        OPENTARGETS_GRAPHQL_URL,
        json={"query": query, "variables": variables},
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()
    errors = payload.get("errors") or []
    if errors:
        raise ValueError(errors[0].get("message") or "Open Targets GraphQL query failed")
    data = payload.get("data")
    if data is None:
        raise ValueError("Open Targets GraphQL response did not include data")
    return data


def _clinical_stage_to_phase(value: Any) -> Optional[int]:
    if value is None:
        return None
    stage = str(value).strip().upper()
    if not stage:
        return None
    if "APPROVED" in stage:
        return 4
    matches = [int(match) for match in re.findall(r"PHASE[_ ]?(\d)", stage)]
    if not matches:
        return None
    return max(matches)


def _normalize_drug_and_clinical_candidate_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for row in rows or []:
        drug = row.get("drug") or {}
        drug_id = drug.get("id")
        if not drug_id:
            continue
        disease_entries = []
        for disease_entry in row.get("diseases") or []:
            disease = (disease_entry or {}).get("disease") or {}
            disease_entries.append(
                {
                    "disease.id": disease.get("id"),
                    "disease.name": disease.get("name"),
                }
            )
        if not disease_entries:
            disease_entries = [{"disease.id": None, "disease.name": None}]
        for disease_entry in disease_entries:
            normalized.append(
                {
                    "phase": _clinical_stage_to_phase(row.get("maxClinicalStage")),
                    "status": row.get("maxClinicalStage"),
                    "drug.id": drug_id,
                    "drug.name": drug.get("name"),
                    **disease_entry,
                }
            )
    return normalized


def _fetch_target_association_score_for_disease(
    ensg_id: str,
    disease_id: str,
    page_size: int = 500,
) -> Optional[float]:
    score_value: Optional[float] = None
    index = 0
    while True:
        variables = {"ensgId": ensg_id, "index": index, "size": page_size}
        try:
            response = requests.post(
                OPENTARGETS_GRAPHQL_URL,
                json={"query": ASSOCIATED_DISEASES_QUERY, "variables": variables},
                timeout=60,
            )
            response.raise_for_status()
            payload = response.json()
            if payload.get("errors"):
                raise ValueError(payload["errors"][0].get("message") or "Open Targets associatedDiseases query failed")
        except Exception as exc:
            logger.debug(
                "OpenTargets associatedDiseases query failed for %s: %s", ensg_id, exc
            )
            return score_value

        assoc = (((payload.get("data") or {}).get("target") or {}).get("associatedDiseases") or {})
        rows = assoc.get("rows") or []
        count = assoc.get("count")
        for row in rows:
            row_disease_id = (row.get("disease") or {}).get("id")
            if row_disease_id != disease_id:
                continue
            score = row.get("score")
            if score is None:
                continue
            if score_value is None or score > score_value:
                score_value = score

        if not rows:
            break
        index += 1
        if count is not None and index * page_size >= count:
            break
        if count is None and len(rows) < page_size:
            break
    return score_value


def _get_target_association_score_for_disease(
    ensg_id: str,
    disease_id: str,
) -> Optional[float]:
    cache_key = (ensg_id, disease_id)
    cached = _target_association_score_cache.get(cache_key)
    if cache_key in _target_association_score_cache:
        return cached
    score_value = _fetch_target_association_score_for_disease(ensg_id, disease_id)
    _target_association_score_cache[cache_key] = score_value
    return score_value


def _normalize_pathway_key(name: str) -> str:
    if not isinstance(name, str):
        return ""
    return re.sub(r"\s+", " ", name.strip().lower())


def _parse_protein_symbols(raw: Any) -> List[str]:
    '''Extract HGNC gene symbols from raw pathway protein representations.
    Supports strings like 'p(HGNC:KCNK10); p(HGNC:KCNK2)' or comma/semicolon separated lists.

    Parameters:
    ---------
    raw (Any): a pathway's protein field, whose shape varies by source.

    Returns:
    ----------
    symbols (list): the HGNC gene symbols found in it.
    '''

    if raw is None or (isinstance(raw, float) and math.isnan(raw)):  # type: ignore[arg-type]
        return []
    if isinstance(raw, list):
        values = raw
    else:
        values = re.split(r"[;,]", str(raw))
    symbols: List[str] = []
    for value in values:
        match = re.search(r"HGNC:([A-Za-z0-9\-]+)", value)
        if match:
            symbols.append(match.group(1).strip())
        else:
            stripped = value.strip()
            if stripped and not stripped.startswith("p(") and len(stripped) <= 12:
                symbols.append(stripped)
    seen: set[str] = set()
    deduped: List[str] = []
    for sym in symbols:
        if sym and sym not in seen:
            deduped.append(sym)
            seen.add(sym)
    return deduped


def _extract_pathway_id_from_url(url: Optional[str]) -> Optional[str]:
    if not isinstance(url, str):
        return None
    match = re.search(r"(R-[A-Z]+-\d+)", url)
    return match.group(1) if match else None


def _resolve_reactome_pathway_id(pathway_name: str) -> Optional[str]:
    '''Resolve Reactome pathway identifiers using the Content Service search endpoint.

    Parameters:
    ---------
    pathway_name (str): a pathway name as it appears in the knowledge graph.

    Returns:
    ----------
    pathway_id (str): the Reactome stable identifier, or None when the search finds no match.
    '''

    if not pathway_name:
        return None
    try:
        resp = requests.get(
            REACTOME_SEARCH_URL,
            params={"query": pathway_name},
            timeout=30,
        )
        resp.raise_for_status()
        payload = resp.json()
    except Exception as exc:
        logger.debug("Reactome lookup failed for '%s': %s", pathway_name, exc)
        return None

    results = payload.get("results", [])
    for block in results:
        for entry in block.get("entries", []):
            if entry.get("exactType") != "Pathway":
                continue
            species = entry.get("species") or []
            if species and "Homo sapiens" not in species:
                continue
            st_id = entry.get("stId") or entry.get("id")
            if st_id:
                return st_id
    return None


def getDrugsforProteins_count(ensg):
    query_string = """
        query DrugAndClinicalCandidatesQuery($ensgId: String!) {
          target(ensemblId: $ensgId) {
            id
            drugAndClinicalCandidates {
              count
            }
          }
        }
    """
    variables = {"ensgId": ensg}
    data = _opentargets_graphql(query_string, variables)
    return (((data.get("target") or {}).get("drugAndClinicalCandidates")) or {}).get("count", 0)

@tool
def getDrugsforProteins(
    proteins: Union[str, List[str]],
    disease_id: Optional[str] = None,
) -> Dict[str, Any]:
    '''Get known drugs that are associated with given proteins.
    
    Smart Usage:
    - If the `extract_proteins_from_kg` tool has already produced an `output_file`, feed that path here directly.
    - Otherwise, provide manual protein lists or other CSV files with a `gene_symbol` column.
    
    Parameters:
    ---------
    proteins (Union[str, List[str]]): Can be: - File path returned by previous tools (preferred when available) - Comma-separated string (e.g., 'TP53,BRCA1,EGFR') - List of gene symbols
    disease_id (str): disease ID (EFO/MONDO) to score relevance against.

    Returns:
    ----------
    result (dict): Standardized dictionary with:
    - success (bool): Whether drug-protein association retrieval succeeded
    - data (dict): Contains drug-protein association data
    - output_file (str): Path to the CSV file with repurposing candidates
    - message (str): Human-readable operation result
    - metadata (dict): Query parameters and statistics
    '''

    # Handle input
    if isinstance(proteins, str) and os.path.isfile(proteins):
        ext = os.path.splitext(proteins)[-1].lower()
        if ext not in [".csv", ".tsv"]:
            return {
                "success": False,
                "data": None,
                "output_file": None,
                "message": "Error: Only CSV or TSV files are supported",
                "metadata": {"input_file": proteins, "supported_formats": [".csv", ".tsv"]}
            }
        sep = "\t" if ext == ".tsv" else ","
        df = pd.read_csv(proteins, sep=sep)
        df.columns = [col.lower() for col in df.columns]
        if "gene_symbol" not in df.columns:
            return {
                "success": False,
                "data": None,
                "output_file": None,
                "message": "Error: No 'gene_symbol' column found in the input file",
                "metadata": {"input_file": proteins, "available_columns": list(df.columns)}
            }
        prot_list = df["gene_symbol"].dropna().astype(str).tolist()
    elif isinstance(proteins, str):
        prot_list = [s.strip() for s in proteins.split(",") if s.strip()]
    elif isinstance(proteins, list):
        prot_list = [s.strip() for s in proteins if isinstance(s, str) and s.strip()]
    else:
        return {
            "success": False,
            "data": None,
            "output_file": None,
            "message": "Error: Input must be a gene symbol string, a list of gene symbols, or a file path",
            "metadata": {"input_type": type(proteins).__name__}
        }

    if not prot_list:
        return {
            "success": False,
            "data": None,
            "output_file": None,
            "message": "Error: No valid gene symbol strings provided",
            "metadata": {"input": proteins}
        }

    # Main
    api_response = pd.DataFrame()
    df = pd.read_csv(DATA_ROOT / "api_related_data" / "DruggableProtein_annotation_OT.csv")
    mapping_dict_id_symbol = dict(df[['approvedSymbol','ENSG']].values)
    unresolved_proteins: List[str] = []
    proteins_without_annotated_drugs: List[str] = []
    request_failures: List[Dict[str, str]] = []

    for prot in tqdm(prot_list):
        ensg_id = mapping_dict_id_symbol.get(prot)
        if not ensg_id:
            unresolved_proteins.append(prot)
            continue

        try:
            count = getDrugsforProteins_count(ensg_id)
            if not count:
                proteins_without_annotated_drugs.append(prot)
                continue

            variables = {"ensgId": ensg_id}
            data = _opentargets_graphql(KNOWN_DRUGS_ROWS_QUERY, variables)
            rows = _normalize_drug_and_clinical_candidate_rows(
                (((data.get("target") or {}).get("drugAndClinicalCandidates")) or {}).get("rows") or []
            )
            if not rows:
                proteins_without_annotated_drugs.append(prot)
                continue

            df_temp = pd.json_normalize(rows)
            if df_temp.empty:
                proteins_without_annotated_drugs.append(prot)
                continue

            df_temp['Protein'] = prot
            api_response = pd.concat([api_response, df_temp], ignore_index=True)
        except Exception as exc:
            request_failures.append(
                {
                    "protein": prot,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

    api_response.reset_index(drop=True, inplace=True)
    for col in ["Protein", "drug.id", "drug.name", "disease.id", "disease.name", "phase", "status"]:
        if col not in api_response.columns:
            api_response[col] = None
    api_response.drop_duplicates(subset=['Protein', 'drug.id'], inplace=True)

    unique_ids = api_response["drug.id"].dropna().unique().tolist()
    chembl_to_smiles = fetch_smiles_batch(unique_ids, chunk_size=200)
    api_response["smiles"] = api_response["drug.id"].map(chembl_to_smiles)

    api_response = api_response.rename(
        columns={
            "disease.id": "disease_id",
            "disease.name": "disease_name",
            "drug.id": "chembl_id",
            "drug.name": "drug_name",
            "Protein": "gene_symbol",
        }
    )

    if disease_id and not api_response.empty:
        score_cache: Dict[str, Optional[float]] = {}
        for gene_symbol in api_response["gene_symbol"].dropna().unique():
            ensg_id = mapping_dict_id_symbol.get(gene_symbol)
            if not ensg_id:
                continue
            score_cache[gene_symbol] = _get_target_association_score_for_disease(
                ensg_id,
                disease_id,
            )
        api_response["associated_score"] = api_response["gene_symbol"].map(score_cache)

    output_columns = [
        "phase",
        "status",
        "disease_id",
        "disease_name",
        "chembl_id",
        "drug_name",
        "gene_symbol",
    ]
    if disease_id:
        output_columns.append("associated_score")
    output_columns.append("smiles")
    for col in output_columns:
        if col not in api_response.columns:
            api_response[col] = None
    api_response = api_response[output_columns]

    output_file = _output_file('protein_drug_candidates.csv')
    api_response.to_csv(output_file, index=False)
    output_str = str(output_file)

    total_candidates = len(api_response)
    unique_drugs_found = api_response["chembl_id"].nunique(dropna=True)
    unique_proteins_found = api_response["gene_symbol"].nunique(dropna=True)

    if total_candidates == 0:
        if proteins_without_annotated_drugs and not unresolved_proteins and not request_failures:
            message = "There are no annotated drugs for this protein."
            if len(prot_list) > 1:
                message = "There are no annotated drugs for the provided proteins."
        elif unresolved_proteins and not request_failures:
            message = "No drug annotations could be retrieved for the provided proteins."
        else:
            message = "No drug-protein pairs were retrieved."
    else:
        message = (
            f"Successfully found {total_candidates} drug-protein pairs. "
        )

    return {
        "success": len(request_failures) == 0,
        "data": {
            "summary": {
                "total_candidates": total_candidates,
                "showing_in_data": min(total_candidates, 10),
                "data_truncated": total_candidates > 10,
                "complete_data_location": output_str,
                "unique_drugs": unique_drugs_found,
                "unique_proteins": unique_proteins_found
            },
            "analysis_recommendation": f"For complete analysis, use the full dataset at {output_str} which contains all {total_candidates} drug-protein pairs"
        },
        "output_file": output_str,
        "message": message,
        "metadata": {
            "total_input_proteins": len(prot_list),
            "total_drug_protein_pairs": total_candidates,
            "unique_drugs_found": unique_drugs_found,
            "proteins_without_annotated_drugs": proteins_without_annotated_drugs,
            "unresolved_proteins": unresolved_proteins,
            "request_failures": request_failures[:10],
            "request_failures_truncated": len(request_failures) > 10,
            "associated_score_disease_id": disease_id,
            "csv_exported": output_str,
        }
    }




def safe_int(x, default=0):
    '''Convert x to int safely, accepting float-like strings.

    Parameters:
    ---------
    x (any): a value that may be an int, a float, or a float-like string.
    default (int): what to return when the value cannot be converted.

    Returns:
    ----------
    value (int): the converted integer, or `default`.
    '''

    try:
        return int(x)
    except (TypeError, ValueError):
        try:
            return int(float(x))
        except (TypeError, ValueError):
            return default

@tool
def getDrugsforMechanisms(
    moas: Union[List[str], str],
    batch_size: int = 10,
    only_small_molecule: bool = True,   # set False to include biologics too
    min_phase: int = 4,              # use 4 for approved drugs only
) -> Dict[str, Any]:
    '''Retrieve drugs associated with mechanism of action.

    Parameters:
    ---------
    moas (Union[List[str], str]): Mechanism of Action specification that can be: - Path to CSV/TSV file containing mechanism_of_action names (preferred columns: mechanism_of_action) - Comma-separated string of mechanism_of_action names - List of mechanism_of_action name strings

    Returns:
    ----------
    result (dict): Standardized dictionary with summary statistics.
    '''

    # --- normalize input ---
    if isinstance(moas, str) and os.path.isfile(moas):
        ext = os.path.splitext(moas)[-1].lower()
        sep = "\t" if ext == ".tsv" else ","
        df_in = pd.read_csv(moas, sep=sep)
        df_in.columns = [c.lower() for c in df_in.columns]
        if "mechanism_of_action" not in df_in.columns:
            raise ValueError("No 'mechanism_of_action' column found in the file.")
        moa_list = list(df_in["mechanism_of_action"].dropna().unique())
    elif isinstance(moas, str):
        moa_list = [s.strip() for s in moas.split(",") if s.strip()]
    elif isinstance(moas, list):
        moa_list = [s.strip() for s in moas if isinstance(s, str) and s.strip()]
    else:
        raise ValueError("Input must be a string, list[str], or a CSV/TSV file path.")

    if not moa_list:
        return pd.DataFrame(columns=["drug_name", "chembl_id", "mechanism_of_actions", "SMILES"])

    mechanism = new_client.mechanism
    molecule = new_client.molecule

    # --- Step 1: gather all molecule IDs + their MoAs ---
    moa_map = {}  # chembl_id -> set of moa texts
    for moa_name in tqdm(moa_list, desc="Fetching mechanisms"):
        for entry in mechanism.filter(mechanism_of_action__icontains=moa_name):
            mid = entry.get("molecule_chembl_id")
            if not mid:
                continue
            moa_text = entry.get("mechanism_of_action") or moa_name
            moa_map.setdefault(mid, set()).add(moa_text)

    chembl_ids = list(moa_map.keys())
    if not chembl_ids:
        return pd.DataFrame(columns=["drug_name", "chembl_id", "mechanism_of_actions", "SMILES"])

    # --- Step 2: batch-fetch molecule info ---
    records = []
    total_batches = math.ceil(len(chembl_ids) / batch_size)

    stats = {
        "input_ids": len(chembl_ids),
        "molecules_fetched": 0,
        "missing_phase": 0,
        "non_numeric_phase": 0,
        "below_phase_threshold": 0,
        "non_small_molecule": 0,
    }

    for i in tqdm(range(total_batches), desc="Fetching molecules"):
        batch_ids = chembl_ids[i * batch_size : (i + 1) * batch_size]

        mol_iter = molecule.filter(
            molecule_chembl_id__in=batch_ids
        ).only(
            "molecule_chembl_id",
            "pref_name",
            "molecule_type",
            "max_phase",
            "molecule_structures",
        )

        for mol in mol_iter:
            stats["molecules_fetched"] += 1

            raw_phase = mol.get("max_phase")
            max_phase = safe_int(raw_phase, default=None)
            if raw_phase in (None, ""):
                stats["missing_phase"] += 1
                continue
            if max_phase is None:
                stats["non_numeric_phase"] += 1
                continue
            if max_phase < min_phase:
                stats["below_phase_threshold"] += 1
                continue  # not a drug per chosen threshold

            mol_type = mol.get("molecule_type")
            if only_small_molecule and mol_type != "Small molecule":
                stats["non_small_molecule"] += 1
                continue

            mid = mol.get("molecule_chembl_id")
            if not mid:
                continue

            smiles = None
            mstruct = mol.get("molecule_structures")
            if mstruct:
                smiles = mstruct.get("canonical_smiles")

            drug_name = mol.get("pref_name") or mid
            moa_joined = " | ".join(sorted(moa_map.get(mid, [])))

            records.append({
                "drug_name": drug_name,
                "chembl_id": mid,
                "mechanism_of_actions": moa_joined,
                "smiles": smiles
            })

    df = pd.DataFrame(
        records,
        columns=["drug_name", "chembl_id", "mechanism_of_actions", "smiles"],
    ).reset_index(drop=True)

    output_path = _output_file('mechanism_drug_candidate.csv')
    df.to_csv(output_path, index=False)
    output_str = str(output_path)

    return {
            "success": True,
            "data": {
                "summary": {
                    "total_mechanism_of_actions": len(moa_list),
                    "total_MoA_drug_pairs": len(df),
                    "complete_data_location": output_str,
                },
                "analysis_recommendation": f"For complete analysis, use the full dataset at {output_str} which contains all {len(df)} MoA-drug pairs"
            },
            "output_file": output_str,
            "message": f"Successfully query {len(df)} MoA - Drug pairs. Complete dataset saved to {output_str}",
        }


@tool
def getDrugsforPathways(
    pathways: Union[str, List[str]],
) -> Dict[str, Any]:
    '''Retrieve drugs associated with biological pathways.

    Parameters:
    ---------
    pathways (Union[str, List[str]]): Pathway specification that can be: - Path to CSV/TSV file containing pathway names (preferred columns: pathway, name, pathway_id) - Comma-separated string of pathway names - List of pathway name strings

    Returns:
    ----------
    result (dict): Standardized dictionary with summary statistics.
    '''

    def _normalize_input_names(raw: Union[str, List[str]]) -> List[str]:
        if isinstance(raw, list):
            names = [str(item).strip() for item in raw if isinstance(item, str) and item.strip()]
        else:
            names = [frag.strip() for frag in str(raw).split(",") if frag.strip()]
        return [name for name in names if name]

    def _load_pathway_frame(path: str) -> Tuple[pd.DataFrame, Dict[str, str]]:
        ext = os.path.splitext(path)[-1].lower()
        if ext not in [".csv", ".tsv"]:
            raise ValueError("Only CSV or TSV files are supported for pathway inputs.")
        sep = "\t" if ext == ".tsv" else ","
        df = pd.read_csv(path, sep=sep)
        col_map = {col.lower(): col for col in df.columns}
        return df, col_map

    pathway_map: Dict[str, Dict[str, Any]] = {}
    source_names: List[str] = []

    if isinstance(pathways, str) and os.path.isfile(pathways):
        try:
            df, col_map = _load_pathway_frame(pathways)
        except Exception as exc:
            return {
                "success": False,
                "data": None,
                "output_file": None,
                "message": f"Error reading pathways file '{pathways}': {exc}",
                "metadata": {
                    "input_file": pathways,
                    "error_type": type(exc).__name__,
                },
            }

        if df.empty:
            return {
                "success": False,
                "data": None,
                "output_file": None,
                "message": f"Error: Pathway file '{pathways}' is empty",
                "metadata": {"input_file": pathways},
            }

        name_col = None
        for candidate in ("pathway", "pathway_name", "name", "pathways"):
            if candidate in col_map:
                name_col = col_map[candidate]
                break
        if not name_col:
            return {
                "success": False,
                "data": None,
                "output_file": None,
                "message": "Error: No pathway name column found. Expected one of ['pathway', 'pathway_name', 'name'].",
                "metadata": {
                    "input_file": pathways,
                    "available_columns": list(df.columns),
                },
            }

        protein_col = None
        for candidate in ("associated_proteins", "proteins", "associated_genes", "genes"):
            if candidate in col_map:
                protein_col = col_map[candidate]
                break

        id_col = None
        for candidate in ("pathway_id", "reactome_id"):
            if candidate in col_map:
                id_col = col_map[candidate]
                break

        url_col = col_map.get("url")

        for _, row in df.iterrows():
            pathway_name = str(row[name_col]).strip()
            if not pathway_name or pathway_name.lower() == "nan":
                continue
            key = _normalize_pathway_key(pathway_name)
            entry = pathway_map.setdefault(
                key,
                {
                    "name": pathway_name,
                    "proteins": set(),
                    "pathway_id": None,
                    "url": None,
                    "source": pathways,
                },
            )
            if protein_col:
                entry["proteins"].update(_parse_protein_symbols(row[protein_col]))
            if id_col and pd.notna(row[id_col]):
                entry["pathway_id"] = str(row[id_col]).strip()
            if not entry.get("pathway_id") and url_col:
                entry["pathway_id"] = _extract_pathway_id_from_url(row[url_col])
            if url_col and pd.notna(row[url_col]):
                entry["url"] = str(row[url_col]).strip()
            if not entry.get("url") and entry.get("pathway_id"):
                entry["url"] = f"{REACTOME_BROWSER_BASE}{entry['pathway_id']}"
        source_names = [entry["name"] for entry in pathway_map.values()]

    else:
        names = _normalize_input_names(pathways if pathways is not None else [])
        source_names = names
        for name in names:
            key = _normalize_pathway_key(name)
            pathway_map.setdefault(
                key,
                {
                    "name": name,
                    "proteins": set(),
                    "pathway_id": None,
                    "url": None,
                    "source": "manual",
                },
            )

    if not pathway_map:
        return {
            "success": False,
            "data": None,
            "output_file": None,
            "message": "Error: No valid pathway names provided.",
            "metadata": {"input": pathways},
        }

    # Fall back to this conversation's own pathway extract. This used to read
    # `RESULTS_ROOT/pathways.csv` — the shared root rather than the conversation
    # scope — which is not where `extract_pathways_from_kg` writes, so the fallback
    # never fired; and had a file ever landed there it would have been read across
    # conversations.
    fallback_pathways: Dict[str, Dict[str, Any]] = {}
    fallback_path = _output_file("pathways.csv")
    if fallback_path.is_file():
        try:
            fb_df, fb_cols = _load_pathway_frame(fallback_path)
            fb_name_col = None
            for candidate in ("pathway", "pathway_name", "name"):
                if candidate in fb_cols:
                    fb_name_col = fb_cols[candidate]
                    break
            protein_col = fb_cols.get("associated_proteins")
            url_col = fb_cols.get("url")
            id_col = fb_cols.get("pathway_id")
            if fb_name_col:
                for _, row in fb_df.iterrows():
                    name = str(row[fb_name_col]).strip()
                    if not name:
                        continue
                    key = _normalize_pathway_key(name)
                    proteins = _parse_protein_symbols(row[protein_col]) if protein_col else []
                    pathway_id = None
                    if id_col and pd.notna(row[id_col]):
                        pathway_id = str(row[id_col]).strip()
                    if not pathway_id and url_col:
                        pathway_id = _extract_pathway_id_from_url(row[url_col])
                    url = str(row[url_col]).strip() if url_col and pd.notna(row[url_col]) else None
                    if not url and pathway_id:
                        url = f"{REACTOME_BROWSER_BASE}{pathway_id}"
                    fallback_pathways[key] = {
                        "proteins": set(proteins),
                        "pathway_id": pathway_id,
                        "url": url,
                    }
        except Exception as exc:
            logger.debug("Failed to load fallback pathways from %s: %s", fallback_path, exc)

    # Enrich pathway entries with fallback info and Reactome lookup
    for key, entry in pathway_map.items():
        fallback = fallback_pathways.get(key)
        if not entry["proteins"] and fallback:
            entry["proteins"].update(fallback.get("proteins", []))
        if not entry.get("pathway_id"):
            if fallback and fallback.get("pathway_id"):
                entry["pathway_id"] = fallback["pathway_id"]
            else:
                entry["pathway_id"] = _resolve_reactome_pathway_id(entry["name"])
        if not entry.get("url"):
            if fallback and fallback.get("url"):
                entry["url"] = fallback["url"]
            elif entry.get("pathway_id"):
                entry["url"] = f"{REACTOME_BROWSER_BASE}{entry['pathway_id']}"

    if all(len(entry["proteins"]) == 0 for entry in pathway_map.values()):
        return {
            "success": False,
            "data": None,
            "output_file": None,
            "message": "Error: Unable to identify associated proteins for any pathway.",
            "metadata": {
                "pathways_requested": source_names,
                "fallback_used": bool(fallback_pathways),
            },
        }

    # Load gene symbol to ENSG mapping
    try:
        protein_map_df = pd.read_csv(DATA_ROOT / "api_related_data" / "DruggableProtein_annotation_OT.csv")
        symbol_to_ensg = dict(protein_map_df[["approvedSymbol", "ENSG"]].values)
    except Exception as exc:
        return {
            "success": False,
            "data": None,
            "output_file": None,
            "message": f"Error loading protein mapping file: {exc}",
            "metadata": {
                "mapping_file": str(DATA_ROOT / "api_related_data" / "DruggableProtein_annotation_OT.csv"),
                "error_type": type(exc).__name__,
            },
        }

    gene_ensg_cache: Dict[str, Optional[str]] = {k: v for k, v in symbol_to_ensg.items()}
    gene_drug_cache: Dict[str, pd.DataFrame] = {}

    def _lookup_ensg(symbol: str) -> Optional[str]:
        if symbol in gene_ensg_cache:
            return gene_ensg_cache[symbol]
        variables = {"terms": [symbol]}
        try:
            response = requests.post(
                OPENTARGETS_GRAPHQL_URL,
                json={"query": MAP_TARGET_IDS_QUERY, "variables": variables},
                timeout=60,
            )
            response.raise_for_status()
            payload = response.json()
            mappings = (((payload.get("data") or {}).get("mapIds") or {}).get("mappings") or [])
            for mapping in mappings:
                if mapping.get("term") != symbol:
                    continue
                for hit in mapping.get("hits", []):
                    if hit.get("entity") == "target" and hit.get("id", "").startswith("ENSG"):
                        ensg_id = hit.get("id")
                        gene_ensg_cache[symbol] = ensg_id
                        return ensg_id
        except Exception as exc:
            logger.debug("mapIds lookup failed for %s: %s", symbol, exc)
        gene_ensg_cache[symbol] = None
        return None

    def _fetch_drug_rows(symbol: str) -> pd.DataFrame:
        if symbol in gene_drug_cache:
            return gene_drug_cache[symbol].copy()
        ensg_id = _lookup_ensg(symbol)
        if not ensg_id:
            gene_drug_cache[symbol] = pd.DataFrame()
            return pd.DataFrame()
        try:
            count = getDrugsforProteins_count(ensg_id)
        except Exception as exc:
            logger.debug("Failed to fetch known drug count for %s (%s): %s", symbol, ensg_id, exc)
            gene_drug_cache[symbol] = pd.DataFrame()
            return pd.DataFrame()
        if not count:
            gene_drug_cache[symbol] = pd.DataFrame()
            return pd.DataFrame()
        variables = {"ensgId": ensg_id}
        try:
            data = _opentargets_graphql(KNOWN_DRUGS_ROWS_QUERY, variables)
            rows = _normalize_drug_and_clinical_candidate_rows(
                (((data.get("target") or {}).get("drugAndClinicalCandidates")) or {}).get("rows") or []
            )
            df_gene = pd.json_normalize(rows)
            if not df_gene.empty:
                df_gene["gene_symbol"] = symbol
                df_gene["ensg_id"] = ensg_id
            gene_drug_cache[symbol] = df_gene
            return df_gene.copy()
        except Exception as exc:
            logger.debug("Failed to fetch known drugs for %s (%s): %s", symbol, ensg_id, exc)
            gene_drug_cache[symbol] = pd.DataFrame()
            return pd.DataFrame()

    pathway_rows: List[pd.DataFrame] = []
    pathway_stats: Dict[str, Dict[str, Any]] = {}

    for entry in pathway_map.values():
        proteins = sorted(entry["proteins"])
        pathway_stats[entry["name"]] = {
            "protein_candidates": len(proteins),
            "resolved_proteins": 0,
            "resolved_drugs": 0,
        }
        for symbol in proteins:
            df_gene = _fetch_drug_rows(symbol)
            if df_gene.empty:
                continue
            pathway_stats[entry["name"]]["resolved_proteins"] += 1
            pathway_stats[entry["name"]]["resolved_drugs"] += len(df_gene)
            df_gene = df_gene.copy()
            df_gene["pathway_name"] = entry["name"]
            df_gene["pathway_id"] = entry.get("pathway_id")
            df_gene["pathway_url"] = entry.get("url")
            pathway_rows.append(df_gene)

    if not pathway_rows:
        return {
            "success": True,
            "data": {
                "summary": {
                    "total_pathways": len(pathway_map),
                    "total_drug_pathway_pairs": 0,
                    "unique_drugs": 0,
                },
                "analysis_recommendation": "No drug associations were found for the supplied pathways.",
            },
            "output_file": None,
            "message": "No drug associations resolved for the supplied pathways.",
        }

    combined = pd.concat(pathway_rows, ignore_index=True)
    combined.drop_duplicates(subset=["pathway_name", "gene_symbol", "drug.id"], inplace=True)

    unique_drugs = combined["drug.id"].dropna().unique().tolist()
    chembl_to_smiles = fetch_smiles_batch(unique_drugs, chunk_size=200) if unique_drugs else {}
    combined["smiles"] = combined["drug.id"].map(chembl_to_smiles)

    rename_map = {
        "disease.id": "disease_id",
        "disease.name": "disease_name",
        "drug.id": "chembl_id",
        "drug.name": "drug_name",
    }
    combined.rename(columns=rename_map, inplace=True)

    column_order = [
        "pathway_name",
        "pathway_id",
        "pathway_url",
        "gene_symbol",
        "ensg_id",
        "chembl_id",
        "drug_name",
        "phase",
        "status",
        "disease_id",
        "disease_name",
        "mechanismOfAction",
        "smiles",
    ]
    for col in column_order:
        if col not in combined.columns:
            combined[col] = None
    combined = combined[column_order]

    output_file = _output_file("pathway_drug_candidates.csv")
    combined.to_csv(output_file, index=False)
    output_str = str(output_file)

    summary = {
        "total_pathways": len(pathway_map),
        "pathways_with_proteins": sum(1 for entry in pathway_map.values() if entry["proteins"]),
        "total_drug_pathway_pairs": len(combined),
        "unique_drugs": combined["chembl_id"].nunique(dropna=True),
        "unique_genes": combined["gene_symbol"].nunique(dropna=True),
    }

    sample_records = combined.head(10)[
        ["pathway_name", "gene_symbol", "chembl_id", "drug_name", "phase"]
    ].to_dict("records")

    return {
        "success": True,
        "data": {
            "summary": summary,
            "analysis_recommendation": f"Full pathway-drug associations are available in {output_str}.",
        },
        "output_file": output_str,
        "message": f"Resolved {len(combined)} pathway-drug associations across {summary['total_pathways']} pathway inputs.",
    }

# ========== Export all tools ==========

__all__ = [
    'search_disease_id',
    'getDrugsforProteins',
    'getDrugsforMechanisms'
    'getDrugsforPathways',
    'create_knowledge_graph',
    'extract_mechanism_of_actions_from_kg',
    'extract_drugs_from_kg',
    'extract_proteins_from_kg',
    'extract_pathways_from_kg',
    'extract_side_effects_from_kg',
]
