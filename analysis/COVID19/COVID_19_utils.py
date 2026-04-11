import pandas as pd 
import requests
from tqdm import tqdm
import numpy as np
from sklearn.metrics import auc

def get_chembl_id(drug_name):
    """
    Query PubChem database first, then ChEMBL database to get ChEMBL ID for a given drug name or synonym.
    
    Parameters:
    drug_name (str): Name or synonym of the drug to search for
    
    Returns:
    str: ChEMBL ID if found, None otherwise
    """
    
    # Try PubChem first
    chembl_id = _get_chembl_id_from_pubchem(drug_name)
    
    if chembl_id:
        return chembl_id
    
    # If PubChem returns None, fall back to ChEMBL API
    print(f"PubChem search failed for '{drug_name}', trying ChEMBL API...")
    chembl_id = _get_chembl_id_from_chembl(drug_name)
    
    return chembl_id


def _get_chembl_id_from_pubchem(drug_name):
    """
    Query PubChem database to get ChEMBL ID.
    
    Parameters:
    drug_name (str): Name or synonym of the drug to search for
    
    Returns:
    str: ChEMBL ID if found, None otherwise
    """
    base_url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name"
    
    try:
        # Step 1: Get PubChem CID from compound name/synonym
        search_url = f"{base_url}/{requests.utils.quote(drug_name)}/cids/JSON"
        response = requests.get(search_url)
        response.raise_for_status()
        
        data = response.json()
        
        # Check if any results were found
        if 'IdentifierList' in data and 'CID' in data['IdentifierList']:
            cid = data['IdentifierList']['CID'][0]  # Get first CID
            
            # Step 2: Get ChEMBL ID from the compound's cross-references
            xrefs_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/xrefs/RegistryID/JSON"
            xrefs_response = requests.get(xrefs_url)
            xrefs_response.raise_for_status()
            
            xrefs_data = xrefs_response.json()
            
            # Look for ChEMBL ID in the registry IDs
            if 'InformationList' in xrefs_data and 'Information' in xrefs_data['InformationList']:
                registry_ids = xrefs_data['InformationList']['Information'][0].get('RegistryID', [])
                
                # Find ChEMBL ID (starts with 'CHEMBL')
                for reg_id in registry_ids:
                    if reg_id.startswith('CHEMBL'):
                        return reg_id
                
                return None  # No ChEMBL ID found in cross-references
            else:
                return None
        else:
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"PubChem error for '{drug_name}': {e}")
        return None
    except (KeyError, IndexError) as e:
        print(f"PubChem parsing error for '{drug_name}': {e}")
        return None


def _get_chembl_id_from_chembl(drug_name):
    """
    Query ChEMBL database directly to get ChEMBL ID.
    
    Parameters:
    drug_name (str): Name of the drug to search for
    
    Returns:
    str: ChEMBL ID if found, None otherwise
    """
    base_url = "https://www.ebi.ac.uk/chembl/api/data/molecule/search"
    
    params = {
        'q': drug_name,
        'format': 'json'
    }
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        
        data = response.json()
        
        # Check if any results were found
        if 'molecules' in data and len(data['molecules']) > 0:
            return data['molecules'][0].get('molecule_chembl_id')
        else:
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"ChEMBL error for '{drug_name}': {e}")
        return None
    except (KeyError, IndexError) as e:
        print(f"ChEMBL parsing error for '{drug_name}': {e}")
        return None


# ROC calculation
def roc_from_score(df, score_col, hit_col, ascending):
    """
    Experimental ROC: sort by score, then walk row-by-row (each additional row becomes predicted positive).
    ascending=True means "smaller score = better"; ascending=False means "larger score = better".
    """
    d = df.copy().sort_values(score_col, ascending=ascending).reset_index(drop=True)

    total_pos = int(d[hit_col].sum())
    total_neg = len(d) - total_pos

    tpr, fpr = [0.0], [0.0]
    tp = fp = 0

    for i in range(len(d)):
        if bool(d.loc[i, hit_col]):
            tp += 1
        else:
            fp += 1

        tpr.append(tp / total_pos if total_pos else 0.0)
        fpr.append(fp / total_neg if total_neg else 0.0)

    tpr.append(1.0); fpr.append(1.0)
    return np.array(fpr), np.array(tpr), auc(fpr, tpr)

def roc_from_rank(df, rank_col, hit_col):
    """
    Repuragent ROC: smaller rank = better, thresholds include ties.
    """
    d = df.copy().sort_values(rank_col, ascending=True).reset_index(drop=True)

    total_pos = int(d[hit_col].sum())
    total_neg = len(d) - total_pos

    thresholds = np.sort(d[rank_col].dropna().unique())

    tpr, fpr = [0.0], [0.0]
    for r in thresholds:
        top = d[d[rank_col] <= r]
        tp = int(top[hit_col].sum())
        fp = len(top) - tp

        tpr.append(tp / total_pos if total_pos else 0.0)
        fpr.append(fp / total_neg if total_neg else 0.0

                   )

    tpr.append(1.0); fpr.append(1.0)
    return np.array(fpr), np.array(tpr), auc(fpr, tpr)