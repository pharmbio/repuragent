import asyncio
import json
import math
import os
import re
import sys
import threading
from datetime import datetime
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple, Union

import httpx
import pandas as pd
import pybel
import requests
from chembl_webresource_client.new_client import new_client
from langchain_core.tools import tool
from pybel.dsl import Abundance, BiologicalProcess
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from app.config import logger  # noqa E402
from kgg.kgg_apiutils import createKG, searchDisease  # noqa E402


# ========== Tool 1: Disease Search ==========

@tool
def search_disease_id(disease_name: str) -> Dict[str, Any]:
    """
    Search for disease IDs (EFO/MONDO) using disease name.
    
    Args:
        disease_name: Human-readable disease name (e.g., "Alzheimer's disease")
    
    Returns:
        Standardized dictionary with:
        - success (bool): Whether the operation succeeded
        - data (dict): Contains 'matches' list and 'top_match_id' 
        - message (str): Human-readable operation result
        - metadata (dict): Query parameters and result counts
    """

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
def create_knowledge_graph(disease_id: str, 
                          clinical_trial_phase: int = 1,
                          protein_threshold: float = 0.5) -> Dict[str, Any]:
    """
    Create a disease-specific knowledge graph using KGG.
    This is the FIRST step - create the graph before extracting information.
    
    Args:
        disease_id: EFO/MONDO disease ID (e.g., "EFO_0000685", "MONDO_0004975")
        clinical_trial_phase: Minimum clinical trial phase (1-4)
        protein_threshold: Minimum protein association score (0.0-1.0)
    
    Returns:
        Standardized dictionary with:
        - success (bool): Whether the knowledge graph was created successfully
        - data (dict): Contains graph statistics (nodes, edges, etc.)
        - output_file (str): Path to the saved knowledge graph file
        - message (str): Human-readable operation result
        - metadata (dict): Input parameters and creation details
    """

    
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
        
        # Store KG in a temporary location for subsequent tools to access
        # In a real system, this could be stored in agent's context/state
        import pickle
        kg_path = f"./results/kg_{disease_id.replace(':', '_')}.pkl"
        
        # Ensure results directory exists
        os.makedirs('results', exist_ok=True)
        
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
            "output_file": kg_path,
            "message": f"Knowledge graph created successfully for '{disease_id}' with {num_nodes} nodes and {num_edges} edges",
            "metadata": {
                "disease_id": disease_id,
                "clinical_trial_phase": clinical_trial_phase,
                "protein_threshold": protein_threshold,
                "file_path": kg_path,
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
def extract_drugs_from_kg(kg_path: str, limit: int = 20) -> Dict[str, Any]:
    """
    Extract drug information from a previously created knowledge graph.
    
    Args:
        kg_path: Path to the saved knowledge graph
        limit: Maximum number of drugs to return
    
    Returns:
        Standardized dictionary with:
        - success (bool): Whether drug extraction succeeded
        - data (dict): Contains extracted drug information
        - output_file (str): Path to the CSV file with drug data
        - message (str): Human-readable operation result
        - metadata (dict): Extraction parameters and statistics
    """
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
                
                if len(drugs) >= limit:
                    break
        
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
            
            # Ensure results directory exists
            os.makedirs('results', exist_ok=True)
            
            # Export to CSV in results folder
            csv_path = "results/known_drugs.csv"
            df.to_csv(csv_path, index=False)
            
            return {
                "success": True,
                "data": {
                    "summary": {
                        "total_drugs": len(drugs),
                        "showing_in_data": min(len(drugs), 10),
                        "data_truncated": len(drugs) > 10,
                        "complete_data_location": csv_path
                    },
                    "sample_drugs": [d['chembl_id'] for d in drugs[:10]] if drugs else [],
                    "analysis_recommendation": f"For complete analysis, use the full dataset at {csv_path} which contains all {len(drugs)} drugs"
                },
                "output_file": csv_path,
                "message": f"Successfully extracted {len(drugs)} known drugs. Showing 10 sample records in response data, complete dataset saved to {csv_path}",
                "metadata": {
                    "kg_path": kg_path,
                    "limit_requested": limit,
                    "drugs_found": len(drugs),
                    "csv_exported": csv_path,
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
                    "sample_drugs": [],
                    "analysis_recommendation": "No drugs found in the knowledge graph - consider different extraction parameters or verify the knowledge graph contains drug information"
                },
                "output_file": None,
                "message": "No drugs found in the knowledge graph",
                "metadata": {
                    "kg_path": kg_path,
                    "limit_requested": limit,
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
def extract_mechanism_of_actions_from_kg(kg_path: str) -> pd.DataFrame:
    """
    Extract mechanism of actions from a knowledge graph.
    
    Args:
        kg_path: Path to the saved knowledge graph
    
    Returns:
        Dictionary containing side effect information
    """

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
    
    csv_path = "results/mechanism_of_actions.csv"
    df.to_csv(csv_path, index=False)
    
    return {
        "success": True,
        "data": {
            "summary": {
                "total_mechanism_of_actions": len(df['mechanism_of_action'].value_counts()),
                "complete_data_location": csv_path
            },
            "analysis_recommendation": f"For complete analysis, use the full dataset at {csv_path} which contains all {len(df['mechanism_of_action'].value_counts())} mechanism of actions"
        },
        "output_file": csv_path,
        "message": f"Successfully extracted {len(df['mechanism_of_action'].value_counts())} side effects.",
        "metadata": {
            "kg_path": kg_path,
            "mechanism_of_actions_found": len(df['mechanism_of_action'].value_counts()),
            "csv_exported": csv_path
                }
            }

# ========== Tool 4: Extract Proteins from KG ==========

@tool
def extract_proteins_from_kg(kg_path: str, 
                            druggable_only: bool = False,
                            ) -> Dict[str, Any]:
    """
    Extract protein/target information from a previously created knowledge graph.
    
    Args:
        kg_path: Path to the saved knowledge graph
        druggable_only: If True, only return druggable proteins
    
    Returns:
        Standardized dictionary with:
        - success (bool): Whether protein extraction succeeded
        - data (dict): Contains extracted protein information
        - output_file (str): Path to the CSV file with protein data
        - message (str): Human-readable operation result
        - metadata (dict): Extraction parameters and statistics
    """
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
            'opentargets_url': []
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
                
                proteins_processed += 1
                
        
        # Create pandas DataFrame
        df = pd.DataFrame(protein_info)
        
        # Ensure results directory exists
        os.makedirs('results', exist_ok=True)
        
        # Export to CSV
        output_path = 'results/associated_genes.csv'
        df.to_csv(output_path, index=False)
        
        return {
            "success": True,
            "data": {
                "summary": {
                    "total_proteins": len(df),
                    "showing_in_data": min(len(df), 10),
                    "data_truncated": len(df) > 10,
                    "complete_data_location": output_path,
                    "druggable_filter_applied": druggable_only
                },
                "sample_proteins": protein_info['gene_symbol'][:10] if 'gene_symbol' in protein_info else [],
                "analysis_recommendation": f"For complete analysis, use the full dataset at {output_path} which contains all {len(df)} proteins"
            },
            "output_file": output_path,
            "message": f"Successfully extracted {len(df)} proteins. Showing {min(len(df), 10)} sample records in response data, complete dataset saved to {output_path}",
            "metadata": {
                "kg_path": kg_path,
                "druggable_only": druggable_only,
                "total_proteins": len(df),
                "csv_exported": output_path,
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
def extract_pathways_from_kg(kg_path: str, limit: int = 20) -> Dict[str, Any]:
    """
    Extract pathway information from a knowledge graph.
    
    Args:
        kg_path: Path to the saved knowledge graph
        limit: Maximum number of pathways to return
    
    Returns:
        Dictionary containing pathway information
    """
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
                
                if len(pathways) >= limit:
                    break
        
        # Convert pathways list to pandas DataFrame and export as CSV
        if pathways:
            # Create DataFrame from pathways list
            df = pd.DataFrame(pathways)
            
            # Handle list columns by converting them to string representation
            if 'associated_proteins' in df.columns:
                df['associated_proteins'] = df['associated_proteins'].apply(lambda x: '; '.join(x) if isinstance(x, list) else x)
            if 'associated_drugs' in df.columns:
                df['associated_drugs'] = df['associated_drugs'].apply(lambda x: '; '.join(x) if isinstance(x, list) else x)
            
            # Export to CSV in results folder
            csv_path = "results/pathways.csv"
            df.to_csv(csv_path, index=False)
            
            return {
                "success": True,
                "data": {
                    "summary": {
                        "total_pathways": len(pathways),
                        "showing_in_data": min(len(pathways), 10),
                        "data_truncated": len(pathways) > 10,
                        "complete_data_location": csv_path
                    },
                    "sample_pathways": [p['name'] for p in pathways[:10]] if pathways else [],
                    "analysis_recommendation": f"For complete analysis, use the full dataset at {csv_path} which contains all {len(pathways)} pathways"
                },
                "output_file": csv_path,
                "message": f"Successfully extracted {len(pathways)} pathways. Showing {min(len(pathways), 10)} sample records in response data, complete dataset saved to {csv_path}",
                "metadata": {
                    "kg_path": kg_path,
                    "limit_requested": limit,
                    "pathways_found": len(pathways),
                    "csv_exported": csv_path,
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
                    "sample_pathways": [],
                    "pathway_types": [],
                    "analysis_recommendation": "No pathways found in the knowledge graph - consider different extraction parameters or verify the knowledge graph contains pathway information"
                },
                "output_file": None,
                "message": "No pathways found in knowledge graph",
                "metadata": {
                    "kg_path": kg_path,
                    "limit_requested": limit,
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
def extract_side_effects_from_kg(kg_path: str, 
                                drug_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Extract side effect information from a knowledge graph.
    
    Args:
        kg_path: Path to the saved knowledge graph
        drug_id: Optional ChEMBL ID to filter side effects for specific drug
    
    Returns:
        Dictionary containing side effect information
    """
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
            
            # Export to CSV in results folder
            csv_path = "results/side_effects.csv"
            df.to_csv(csv_path, index=False)
            
            return {
                "success": True,
                "data": {
                    "summary": {
                        "total_side_effects": len(side_effects),
                        "showing_in_data": min(len(side_effects), 10),
                        "data_truncated": len(side_effects) > 10,
                        "complete_data_location": csv_path,
                        "drug_filter_applied": drug_id is not None
                    },
                    "sample_side_effects": side_effects[:10],
                    "analysis_recommendation": f"For complete analysis, use the full dataset at {csv_path} which contains all {len(side_effects)} side effects"
                },
                "output_file": csv_path,
                "message": f"Successfully extracted {len(side_effects)} side effects. Showing {min(len(side_effects), 10)} sample records in response data, complete dataset saved to {csv_path}",
                "metadata": {
                    "kg_path": kg_path,
                    "filter_drug": drug_id,
                    "side_effects_found": len(side_effects),
                    "csv_exported": csv_path,
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
                    "sample_side_effects": [],
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




def get_smiles_from_chembl(chembl_id):
    """
    Retrieve SMILES structure from ChEMBL API using ChEMBL ID
    """
    try:
        chembl_url = f"https://www.ebi.ac.uk/chembl/api/data/molecule/{chembl_id}.json"
        r = requests.get(chembl_url, timeout=30)
        if r.status_code == 200:
            data = r.json()
            if data.get('molecule_structures'):
                return data['molecule_structures'].get('canonical_smiles')
        print(f'No SMILES found for {chembl_id}')
        return None
    except Exception as e:
        print(f"Error retrieving SMILES for {chembl_id}: {str(e)}")
        return None

CHEMBL_BATCH_URL = "https://www.ebi.ac.uk/chembl/api/data/molecule.json"
CHEMBL_MOLECULE_SEARCH_URL = "https://www.ebi.ac.uk/chembl/api/data/molecule/search"
OPENFDA_EVENT_ENDPOINT = "https://api.fda.gov/drug/event.json"

HTTPX_TIMEOUT = httpx.Timeout(60.0, read=60.0, connect=30.0)
OPENFDA_MAX_CONCURRENCY = 6
CHEMBL_MAX_CONCURRENCY = 6
RETRY_BACKOFF_SECONDS = 2.0


def _run_async_task(coro: "asyncio.Future[Any]") -> Any:
    """
    Execute an async coroutine from synchronous code with graceful fallback
    when an event loop is already running.
    """
    try:
        running_loop = asyncio.get_running_loop()
    except RuntimeError:
        running_loop = None

    if running_loop and running_loop.is_running():
        result_container: Dict[str, Any] = {}
        error_container: Dict[str, BaseException] = {}

        def _thread_runner() -> None:
            try:
                result_container["value"] = asyncio.run(coro)
            except BaseException as exc:  # propagate later
                error_container["error"] = exc

        thread = threading.Thread(target=_thread_runner, daemon=True)
        thread.start()
        thread.join()

        if "error" in error_container:
            raise error_container["error"]
        return result_container.get("value")

    return asyncio.run(coro)

def fetch_smiles_batch(chembl_ids, chunk_size=200):
    """
    Batch-fetch canonical_smiles for many ChEMBL IDs using a persistent HTTP client.
    Returns {chembl_id: smiles or None}
    """
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


async def _fetch_openfda_counts(
    side_effect: str,
    limit: int,
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
) -> List[Dict[str, Any]]:
    params = {
        "search": f'patient.reaction.reactionmeddrapt:"{side_effect.upper()}"',
        "count": "patient.drug.medicinalproduct.exact",
        "limit": max(1, min(limit, 100)),
    }
    try:
        async with semaphore:
            response = await client.get(OPENFDA_EVENT_ENDPOINT, params=params)
        if response.status_code == 404:
            return []
        response.raise_for_status()
        data = response.json()
        return data.get("results", [])
    except httpx.HTTPError as exc:
        logger.debug("OpenFDA query failed for side effect '%s': %s", side_effect, exc)
        await asyncio.sleep(RETRY_BACKOFF_SECONDS)
        return []


def _parse_side_effect_inputs(side_effects: Union[str, List[str]]) -> Tuple[List[str], Dict[str, Any]]:
    metadata: Dict[str, Any] = {}

    if isinstance(side_effects, str) and os.path.isfile(side_effects):
        path = side_effects
        ext = os.path.splitext(path)[-1].lower()
        if ext not in [".csv", ".tsv"]:
            raise ValueError("Only CSV or TSV files are supported for side effect inputs.")
        sep = "\\t" if ext == ".tsv" else ","
        df = pd.read_csv(path, sep=sep)
        if df.empty:
            raise ValueError(f"Input file '{path}' is empty.")
        lowercase_cols = {col.lower(): col for col in df.columns}
        candidate_columns = [
            "side_effect",
            "side effects",
            "sideeffect",
            "side_effects",
            "side effect",
            "adverse_event",
            "adverse events",
            "event",
            "events",
            "reaction",
            "reactions",
            "meddra_term",
            "meddra",
        ]
        chosen_col: Optional[str] = None
        for candidate in candidate_columns:
            if candidate in lowercase_cols:
                chosen_col = lowercase_cols[candidate]
                break
        if not chosen_col:
            raise ValueError(
                "No side effect column found. Expected one of "
                "['side_effect', 'adverse_event', 'event', 'reaction', 'meddra_term']."
            )
        values = (
            df[chosen_col]
            .dropna()
            .astype(str)
            .map(lambda x: x.strip())
        )
        side_effect_list = [val for val in values if val]
        metadata["input_file"] = path
        metadata["detected_column"] = chosen_col
    elif isinstance(side_effects, str):
        fragments = re.split(r"[\\n;,]", side_effects)
        side_effect_list = [frag.strip() for frag in fragments if frag.strip()]
        metadata["input_mode"] = "string"
    elif isinstance(side_effects, list):
        side_effect_list = [str(item).strip() for item in side_effects if isinstance(item, str) and item.strip()]
        metadata["input_mode"] = "list"
    else:
        raise ValueError("Input must be a file path, a string of side effects, or a list of strings.")

    seen: set[str] = set()
    deduped: List[str] = []
    for effect in side_effect_list:
        key = effect.lower()
        if key and key not in seen:
            deduped.append(effect)
            seen.add(key)

    if not deduped:
        raise ValueError("No valid side effect terms found in the input.")

    metadata["unique_side_effects"] = len(deduped)
    metadata["total_raw_entries"] = len(side_effect_list)
    return deduped, metadata


REACTOME_SEARCH_URL = "https://reactome.org/ContentService/search/query"
REACTOME_BROWSER_BASE = "https://reactome.org/PathwayBrowser/#/"
OPENTARGETS_GRAPHQL_URL = "https://api.platform.opentargets.org/api/v4/graphql"

KNOWN_DRUGS_ROWS_QUERY = """
    query KnownDrugsQuery(
      $ensgId: String!
      $cursor: String
      $freeTextQuery: String
      $size: Int!
    ) {
      target(ensemblId: $ensgId) {
        id
        knownDrugs(cursor: $cursor, freeTextQuery: $freeTextQuery, size: $size) {
          count
          cursor
          rows {
            phase
            status
            disease { id name }
            drug { id name }
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


def _normalize_pathway_key(name: str) -> str:
    if not isinstance(name, str):
        return ""
    return re.sub(r"\s+", " ", name.strip().lower())


def _parse_protein_symbols(raw: Any) -> List[str]:
    """
    Extract HGNC gene symbols from raw pathway protein representations.
    Supports strings like 'p(HGNC:KCNK10); p(HGNC:KCNK2)' or comma/semicolon separated lists.
    """
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
    """
    Resolve Reactome pathway identifiers using the Content Service search endpoint.
    """
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


def _ensure_results_dir() -> None:
    os.makedirs("results", exist_ok=True)


def getDrugsforProteins_count(ensg):
    query_string = """
        query KnownDrugsQuery(
          $ensgId: String!
          $cursor: String
          $freeTextQuery: String
          $size: Int = 10
        ) {
          target(ensemblId: $ensgId) {
            id
            knownDrugs(cursor: $cursor, freeTextQuery: $freeTextQuery, size: $size) {
              count
            }
          }
        }
    """
    variables = {"ensgId": ensg}
    base_url = "https://api.platform.opentargets.org/api/v4/graphql"
    r = requests.post(base_url, json={"query": query_string, "variables": variables}, timeout=60)
    api_response = r.json()
    return api_response['data']['target']['knownDrugs']['count']

@tool
def getDrugsforProteins(proteins: Union[str, List[str]]) -> Dict[str, Any]:
    """Get known drugs that are associated with given proteins.
    
    Smart Usage:
    - If results/associated_genes.csv exists (from extract_proteins_from_kg), use it directly
    - Otherwise, provide manual protein lists or other CSV files with gene_symbol column
    
    Args:
        proteins: Can be:
            - File path (e.g., 'results/associated_genes.csv') - PREFERRED when available
            - Comma-separated string (e.g., 'TP53,BRCA1,EGFR')  
            - List of gene symbols
            
    Returns:
        Standardized dictionary with:
        - success (bool): Whether drug-protein association retrieval succeeded
        - data (dict): Contains drug-protein association data
        - output_file (str): Path to the CSV file with repurposing candidates
        - message (str): Human-readable operation result
        - metadata (dict): Query parameters and statistics
    """
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
    df = pd.read_csv('data/api_related_data/DruggableProtein_annotation_OT.csv')
    mapping_dict_id_symbol = dict(df[['approvedSymbol','ENSG']].values)

    for prot in tqdm(prot_list):
        try:
            count = getDrugsforProteins_count(mapping_dict_id_symbol[prot])
            variables = {"ensgId": mapping_dict_id_symbol[prot], "size": count}
            r = requests.post(
                OPENTARGETS_GRAPHQL_URL,
                json={"query": KNOWN_DRUGS_ROWS_QUERY, "variables": variables},
                timeout=60,
            )
            rows = r.json()['data']['target']['knownDrugs']['rows']
            df_temp = pd.json_normalize(rows)
            df_temp['Protein'] = prot
            api_response = pd.concat([api_response, df_temp])
        except Exception:
            continue

    api_response.reset_index(drop=True, inplace=True)
    api_response.drop_duplicates(subset=['Protein', 'drug.id'], inplace=True)

    unique_ids = api_response["drug.id"].dropna().unique().tolist()
    chembl_to_smiles = fetch_smiles_batch(unique_ids, chunk_size=200)
    api_response["smiles"] = api_response["drug.id"].map(chembl_to_smiles)

    api_response.columns = ['phase','status','disease_id','disease_name', 'chembl_id','drug_name','gene_symbol','smiles']
    
    # Ensure results directory exists
    os.makedirs('results', exist_ok=True)
    output_file = 'results/protein_drug_candidates.csv'
    api_response.to_csv(output_file, index=False)

    return {
        "success": True,
        "data": {
            "summary": {
                "total_candidates": len(api_response),
                "showing_in_data": min(len(api_response), 10),
                "data_truncated": len(api_response) > 10,
                "complete_data_location": output_file,
                "unique_drugs": len(api_response['chembl_id'].unique()),
                "unique_proteins": len(api_response['gene_symbol'].unique())
            },
            "sample_drug_protein_pairs": api_response[['gene_symbol','chembl_id']].head(10).to_dict('records'),
            "analysis_recommendation": f"For complete analysis, use the full dataset at {output_file} which contains all {len(api_response)} drug-protein pairs"
        },
        "output_file": output_file,
        "message": f"Successfully found {len(api_response)} drug-protein pairs. Showing 10 sample records in response data, complete dataset saved to {output_file}",
        "metadata": {
            "total_input_proteins": len(prot_list),
            "total_drug_protein_pairs": len(api_response),
            "unique_drugs_found": len(api_response['chembl_id'].unique()),
            "csv_exported": output_file,
            "usage_note": "This filtered dataset should be used for subsequent ADMET predictions instead of the original drug database"
        }
    }




def safe_int(x, default=0):
    """Convert x to int safely, accepting float-like strings."""
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
    min_phase: int = 4               # use 4 for approved drugs only

) -> pd.DataFrame:
    """
    Retrieve drugs associated with mechanism of action.

    Args:
        moas: Mechanism of Action specification that can be:
            - Path to CSV/TSV file containing mechanism_of_action names (preferred columns: mechanism_of_action)
            - Comma-separated string of mechanism_of_action names
            - List of mechanism_of_action name strings

    Returns:
        Standardized dictionary with summary statistics.
    """

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

    output_path = 'results/mechanism_drug_candidate.csv'
    df.to_csv(output_path, index=False)

    return {
            "success": True,
            "data": {
                "summary": {
                    "total_mechanism_of_actions": len(moa_list),
                    "total_MoA_drug_pairs": len(df),
                    "complete_data_location": output_path,
                },
                "analysis_recommendation": f"For complete analysis, use the full dataset at {output_path} which contains all {len(df)} MoA-drug pairs"
            },
            "output_file": output_path,
            "message": f"Successfully query {len(df)} MoA - Drug pairs. Complete dataset saved to {output_path}",
        }


@tool
def getDrugsforPathways(pathways: Union[str, List[str]]) -> Dict[str, Any]:
    """
    Retrieve drugs associated with biological pathways.

    Args:
        pathways: Pathway specification that can be:
            - Path to CSV/TSV file containing pathway names (preferred columns: pathway, name, pathway_id)
            - Comma-separated string of pathway names
            - List of pathway name strings

    Returns:
        Standardized dictionary with summary statistics.
    """

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

    # Load fallback pathways from canonical extraction output if available
    fallback_pathways: Dict[str, Dict[str, Any]] = {}
    fallback_path = os.path.join("results", "pathways.csv")
    if os.path.isfile(fallback_path):
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
        protein_map_df = pd.read_csv("data/api_related_data/DruggableProtein_annotation_OT.csv")
        symbol_to_ensg = dict(protein_map_df[["approvedSymbol", "ENSG"]].values)
    except Exception as exc:
        return {
            "success": False,
            "data": None,
            "output_file": None,
            "message": f"Error loading protein mapping file: {exc}",
            "metadata": {
                "mapping_file": "data/api_related_data/DruggableProtein_annotation_OT.csv",
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
        variables = {"ensgId": ensg_id, "size": int(count)}
        try:
            response = requests.post(
                OPENTARGETS_GRAPHQL_URL,
                json={"query": KNOWN_DRUGS_ROWS_QUERY, "variables": variables},
                timeout=90,
            )
            response.raise_for_status()
            payload = response.json()
            rows = (((payload.get("data") or {}).get("target") or {}).get("knownDrugs") or {}).get("rows", [])
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

    _ensure_results_dir()
    output_file = "results/pathway_drug_candidates.csv"
    combined.to_csv(output_file, index=False)

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
            "analysis_recommendation": f"Full pathway-drug associations are available in {output_file}.",
        },
        "output_file": output_file,
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
