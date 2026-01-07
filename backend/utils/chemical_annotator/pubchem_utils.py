"""
pubchem_utils.py

Utility functions for retrieving and processing data from the PubChem database.

Author: Flavio Ballante

Contact: flavio.ballante@ki.se, flavioballante@gmail.com

Institution: CBCS-SciLifeLab-Karolinska Institutet

Year: 2025
"""

import pubchempy as pcp 

def pubchem_get_cid(query, identifier):
    """
    Get the PubChem CID for a query structure.

    Parameters
    ----------
    query: string
        compound's structure (SMILES, InChI, or InChIKey).
    identifier: string
        type of identifier (SMILES, InChI, or InChIKey).

    Returns
    -------
    list
        PubChem CID.
    """
    try:
        PubChem_cid = pcp.get_properties(['MolecularFormula'],query, identifier.lower(),as_dataframe=False)[0]['CID']
    except IndexError:
        PubChem_cid = None  # If no CID is found, return None
    return PubChem_cid
