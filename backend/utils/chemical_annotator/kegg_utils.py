"""
chembl_utils.py

Utility functions for retrieving and processing data from the KEGG database.

Author: Flavio Ballante

Contact: flavio.ballante@ki.se, flavioballante@gmail.com

Institution: CBCS-SciLifeLab-Karolinska Institutet

Year: 2025
"""

import requests
import re
import pandas as pd

def get_pathways_from_ec(ec_number):
    """
    Retrieve KEGG pathway information for a given EC (Enzyme Commission) number.

    Parameters
    ----------
    ec_number : str
        The EC number for which to retrieve associated KEGG pathways.

    Returns
    -------
    DataFrame
        A DataFrame with columns ['EC Numbers', 'KEGG_ID', 'Pathway'] containing pathway information.
        Returns an empty DataFrame if no pathways are found or if an error occurs.
    """
    # Define column names
    COLUMNS = ['EC Numbers', 'KEGG_ID', 'Pathway']
    # Check if ec_number is empty, None, or NaN
    if pd.isna(ec_number) or not ec_number: 
        #print(f"Warning: EC Number is empty, None, or NaN")
        return pd.DataFrame(columns=COLUMNS)

    link_url = f"https://rest.kegg.jp/link/pathway/enzyme:{ec_number}"
    try:
        response = requests.get(link_url)
        response.raise_for_status()
        
        pathways = []
        for line in response.text.strip().split("\n"):
            if line:
                try:
                    pathways.append(line.split("\t")[1])
                except IndexError:
                    #print(f"Warning: Unexpected format in line: {line}")
                    continue
        
        if not pathways:
            #print(f"No pathways found for EC number {ec_number}")
            return pd.DataFrame(columns=COLUMNS)
        
        results = []
        for pathway in pathways:
            pathway_url = f"https://rest.kegg.jp/get/{pathway}"
            pathway_response = requests.get(pathway_url)
            pathway_response.raise_for_status()
            
            lines = pathway_response.text.split('\n')
            pathway_maps = [line.replace('PATHWAY_MAP', '').strip() for line in lines if line.startswith('PATHWAY_MAP') and re.match(r'^PATHWAY_MAP\s+map\d+', line)]
            
            for pathway_map in pathway_maps:
                results.append({'EC Numbers': ec_number, 'KEGG_ID': pathway, 'Pathway': pathway_map})
        
        if not results:
            print(f"No pathway information found for EC number {ec_number}")
            return pd.DataFrame(columns=COLUMNS)
        
        # Create DataFrame
        df = pd.DataFrame(results)
        return df
    
    except requests.RequestException as e:
        print(f"Network error occurred: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    
    return pd.DataFrame(columns=COLUMNS)
