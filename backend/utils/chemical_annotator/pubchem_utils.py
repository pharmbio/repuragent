'''pubchem_utils.py

Utility functions for retrieving and processing data from the PubChem database.

Author: Flavio Ballante

Contact: flavio.ballante@ki.se, flavioballante@gmail.com

Institution: CBCS-SciLifeLab-Karolinska Institutet

Year: 2025
'''

from functools import lru_cache

import pubchempy as pcp 

@lru_cache(maxsize=50_000)
def pubchem_get_cid(query, identifier):
    '''Get the PubChem CID for a query structure.

    Parameters:
    ---------
    query (str): compound's structure (SMILES, InChI, or InChIKey).
    identifier (str): type of identifier (SMILES, InChI, or InChIKey).

    Returns:
    ----------
    cids (list): PubChem CID.
    '''

    try:
        PubChem_cid = pcp.get_properties(['MolecularFormula'],query, identifier.lower(),as_dataframe=False)[0]['CID']
    except IndexError:
        PubChem_cid = None  # If no CID is found, return None
    return PubChem_cid
