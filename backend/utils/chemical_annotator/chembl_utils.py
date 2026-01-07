"""
chembl_utils.py

Utility functions for retrieving and processing data from the ChEMBL database.

Author: Flavio Ballante

Contact: flavio.ballante@ki.se, flavioballante@gmail.com

Institution: CBCS-SciLifeLab-Karolinska Institutet

Year: 2025
"""

import requests
import pandas as pd
from chembl_webresource_client.new_client import new_client
from chembl_webresource_client.utils import utils
import math
from tqdm import tqdm
import json
molecule = new_client.molecule
activity = new_client.activity

def fetch_chembl_status():
    """
    Check the current status of the ChEMBL web service.

    Returns
    -------
    dict or None: The parsed JSON response from the ChEMBL status endpoint if successful,
                  otherwise None if the request fails or returns a non-200 status code.
    """
    try:
        response = requests.get('https://www.ebi.ac.uk/chembl/api/data/status?format=json', timeout=35)
        if response.status_code == 200:
            data = response.json()
            return data
        else:
            return None
    except requests.RequestException:
        return None

chembl_status = fetch_chembl_status()

# %%
def append_empty_rows(dataframe, n):
    """
    Append n empty rows to a DataFrame.

    Parameters
    ----------
    dataframe: dataframe
    n: number of empty rows to append
    """
    for _ in range(n):
        dataframe.loc[len(dataframe)] = pd.Series(dtype='float64')

# %%
def chembl_get_id(query, identifier):
    """
    Get the ChEMBL ID for a query structure.

    Parameters
    ----------
    query: string
        compound's structure (SMILES, InChI, or InChIKey).
    identifier: string
        type of identifier (SMILES, InChI, or InChIKey).
        
    Returns
    -------
    list
        ChEMBL ID.
    """
    if identifier.lower() == "smiles":
        ChEMBL_mol_std = utils.smiles2inchiKey(query)
    elif identifier.lower() == "inchi":
        ChEMBL_mol_std = utils.inchi2inchiKey(query)
    elif identifier.lower() == "inchikey":
        ChEMBL_mol_std = query
        
    ChEMBL_id=molecule.filter(molecule_structures__standard_inchi_key=ChEMBL_mol_std).only(['molecule_chembl_id'])
    ChEMBL_id=[item['molecule_chembl_id'] for item in ChEMBL_id]
    if len(ChEMBL_id) == 0:
        ChEMBL_id=float("NaN")
        return ChEMBL_id
    else:
        return ' '.join(ChEMBL_id)

# %%
def chembl_drug_annotations(chembl_id):
    """
    Get the drug annotations for a query structure.

    Parameters
    ----------
    chembl_id : str
        ChEMBL compound ID.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the compound's annotations or an empty DataFrame with predefined columns if an error occurs.
    """
    # Define the column names
    columns = [
        'molecule_chembl_id', 'active_chembl_id', 'parent_chembl_id', 'canonical_smiles',
        'standard_inchi', 'standard_inchi_key', 'full_molformula', 'full_mwt', 'heavy_atoms',
        'alogp', 'hba', 'hbd', 'aromatic_rings', 'num_ro5_violations', 'indication_class',
        'first_approval', 'max_phase', 'molecule_synonym', 'syn_type', 'synonyms', 'pref_name',
        'therapeutic_flag', 'withdrawn_flag'
    ]
    try:
        # Attempt to filter molecule data by ChEMBL ID
        ChEMBL_drug_annotation = molecule.filter(chembl_id=chembl_id).only([
            'molecule_structures', 'molecule_properties', 'molecule_chembl_id',
            'molecule_hierarchy', 'pref_name', 'molecule_synonyms', 'xref_id',
            'indication_class', 'therapeutic_flag', 'first_approval', 'max_phase',
            'withdrawn_flag'
        ])
        
        # Return an empty DataFrame with predefined columns if no data found
        if not ChEMBL_drug_annotation:
            return pd.DataFrame({col: [None] for col in columns})  # Return a DataFrame with NaN values if no data found
        
        ChEMBL_drug_annotation = ChEMBL_drug_annotation[0]

        # Handle potential empty 'molecule_synonyms'
        if 'molecule_synonyms' in ChEMBL_drug_annotation and ChEMBL_drug_annotation['molecule_synonyms']:
            molecule_synonym_info = ChEMBL_drug_annotation['molecule_synonyms'][0]
            molecule_synonym = molecule_synonym_info.get('molecule_synonym', None)
            syn_type = molecule_synonym_info.get('syn_type', None)
            synonyms = molecule_synonym_info.get('synonyms', None)
        else:
            molecule_synonym = None
            syn_type = None
            synonyms = None
        
        # Flatten the dictionary
        flat_data = {
            'molecule_chembl_id': ChEMBL_drug_annotation.get('molecule_chembl_id'),
            'active_chembl_id': ChEMBL_drug_annotation.get('molecule_hierarchy', {}).get('active_chembl_id'),
            'parent_chembl_id': ChEMBL_drug_annotation.get('molecule_hierarchy', {}).get('parent_chembl_id'),
            'canonical_smiles': ChEMBL_drug_annotation.get('molecule_structures', {}).get('canonical_smiles'),
            'standard_inchi': ChEMBL_drug_annotation.get('molecule_structures', {}).get('standard_inchi'),
            'standard_inchi_key': ChEMBL_drug_annotation.get('molecule_structures', {}).get('standard_inchi_key'),
            'full_molformula': ChEMBL_drug_annotation.get('molecule_properties', {}).get('full_molformula'),
            'full_mwt': ChEMBL_drug_annotation.get('molecule_properties', {}).get('full_mwt'),
            'heavy_atoms': ChEMBL_drug_annotation.get('molecule_properties', {}).get('heavy_atoms'),
            'alogp': ChEMBL_drug_annotation.get('molecule_properties', {}).get('alogp'),
            'hba': ChEMBL_drug_annotation.get('molecule_properties', {}).get('hba'),
            'hbd': ChEMBL_drug_annotation.get('molecule_properties', {}).get('hbd'),
            'aromatic_rings': ChEMBL_drug_annotation.get('molecule_properties', {}).get('aromatic_rings'),
            'num_ro5_violations': ChEMBL_drug_annotation.get('molecule_properties', {}).get('num_ro5_violations'),
            'indication_class': ChEMBL_drug_annotation.get('indication_class'),
            'first_approval': ChEMBL_drug_annotation.get('first_approval'),
            'max_phase': ChEMBL_drug_annotation.get('max_phase'),
            'molecule_synonym': molecule_synonym,
            'syn_type': syn_type,
            'synonyms': synonyms,
            'pref_name': ChEMBL_drug_annotation.get('pref_name'),
            'therapeutic_flag': ChEMBL_drug_annotation.get('therapeutic_flag'),
            'withdrawn_flag': ChEMBL_drug_annotation.get('withdrawn_flag')
        }
        
        # Create DataFrame
        ChEMBL_drug_annotation_df = pd.DataFrame([flat_data])
        return ChEMBL_drug_annotation_df
    
    except Exception as e:
        # Log the exception if necessary, e.g., print(e) or log to a file
        return pd.DataFrame({col: [None] for col in columns})  # Return a DataFrame with NaN values in case of any exception
    #ChEMBL_drug_annotation['CID']= cid
    #return ChEMBL_drug_annotation

# %%
def chembl_drug_indications(chembl_id):
     """
     Get the drug indications for a query structure.

     Parameters
     ----------
     chembl : str
          ChEMBL compound ID.

     Returns
     -------
     pd.DataFrame
          DataFrame containing the compound's indications.
          
     """
     # Define the column names and initialize NaN values
     columns = [
     'indication_refs.ref_id', 'indication_refs.ref_type', 'indication_refs.ref_url',
     'drugind_id', 'efo_id', 'efo_term', 'max_phase_for_ind', 'mesh_heading',
     'mesh_id', 'molecule_chembl_id', 'parent_molecule_chembl_id'
     ]

     try:
          # Initial URL to get the total count of indications
          url = f"https://www.ebi.ac.uk/chembl/api/data/drug_indication?molecule_chembl_id={chembl_id}&offset=0&format=json"
          r = requests.get(url)
          r.raise_for_status()
          data = r.json()
          total_count = data['page_meta']['total_count']

          # If no data is found, return DataFrame with NaN values
          if total_count == 0:
               data = {col: [None] for col in columns}
               data['molecule_chembl_id'] = [chembl_id]
               data_all = pd.DataFrame(data)
               data_all.drop(columns=['parent_molecule_chembl_id'], inplace=True, errors='ignore')
               data_all['total_count'] = total_count
               return data_all
                   
          n_of_pages = (total_count + 19) // 20  # ceil division to get number of pages
        
          # Initialize an empty DataFrame with the defined columns
          data_all = pd.DataFrame(columns=columns)
     
          # Loop through all pages to get all data
          for page in range(n_of_pages):
               offset = 20 * page
               url = f"https://www.ebi.ac.uk/chembl/api/data/drug_indication?molecule_chembl_id={chembl_id}&offset={offset}&format=json"
               r = requests.get(url)
               r.raise_for_status()
               data = r.json()
               if 'drug_indications' in data:
                    page_data = pd.json_normalize(data['drug_indications'])
                    page_data.drop(columns=['indication_refs'], inplace=True, errors='ignore')
                    data_all = pd.concat([data_all, page_data], ignore_index=True)

          # Add total count column
          data_all['total_count'] = total_count

          # Drop unnecessary columns
          data_all.drop(columns=['parent_molecule_chembl_id'], inplace=True, errors='ignore')

          return data_all

     except requests.exceptions.RequestException as e:
          # Handle HTTP request errors
          #print(f"An error occurred while fetching data: {e}")
          return pd.DataFrame({col: [None] for col in columns})

     except Exception as e:
          # Handle general errors
          #print(f"An unexpected error occurred: {e}")
          return pd.DataFrame({col: [None] for col in columns})
     
# %%
def chembl_mechanism_of_action(chembl_id):
    """
    Fetch the mechanism of action for a given ChEMBL compound ID using chembl_webresource_client.

    Parameters
    ----------
    chembl_id : str
        ChEMBL compound ID.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the action type, mechanism of action, and target ChEMBL ID.
        Returns an empty DataFrame if no data is found or if an error occurs.
    """
    # Define the column names and initialize NaN values
    columns = [
    'molecule_chembl_id', 'action_type', 'mechanism_of_action', 'target_chembl_id'
    ]
    mechanism = new_client.mechanism
    try:
        mechanisms = mechanism.filter(molecule_chembl_id=chembl_id)

        # Convert the results to a list of dictionaries
        mechanisms_list = list(mechanisms)

        if not mechanisms_list:
            return pd.DataFrame({col: [None] for col in columns})

        # Extract relevant fields
        MOA = []
        for m in mechanisms_list:
            MOA.append({
                'molecule_chembl_id': m.get('molecule_chembl_id'),
                'action_type': m.get('action_type'),
                'mechanism_of_action': m.get('mechanism_of_action'),
                'target_chembl_id': m.get('target_chembl_id')
            })

        return pd.DataFrame(MOA)

    except Exception as e:
        #print(f"Error fetching mechanism of action for {chembl_id}: {e}")
        return pd.DataFrame({col: [None] for col in columns})

# %%
def chembl_assay_information(chembl_id, confidence_threshold=8, assay_type_in=['B', 'F'], pchembl_value_gte=6):
    """
    Get the assay information for a query structure with a minimum confidence score.

    Parameters
    ----------
    chembl_id : str
        ChEMBL compound ID.
    confidence_threshold : int, optional
        Minimum confidence score for assays (default is 8).
    assay_type_in : list of str, optional
        List of assay types to include (default: ['B', 'F']).
    pchembl_value_gte : float, optional
        Minimum pChEMBL value to include (default: 6).    

    Returns
    -------
    pd.DataFrame
        DataFrame containing the compound's assay information with confidence scores above the threshold.
    """
    # Define the column names and initialize NaN values
    columns = [
        'molecule_chembl_id', 'activity_id', 'assay_chembl_id', 'assay_description', 
        'assay_type', 'bao_endpoint', 'bao_format', 'bao_label', 'document_chembl_id',  
        'document_year', 'pchembl_value', 'relation', 'standard_type', 'standard_units',
        'standard_value', 'target_pref_name', 'target_chembl_id', 'target_organism',
        'confidence_score'  # Add confidence_score to the columns
    ]
     
    document_columns = [
        'doc_type', 'doi', 'journal', 'pubmed_id',
        'title'
    ]

    def fetch_document_info(chembl_doc_id):
        if pd.isna(chembl_doc_id) or chembl_doc_id == '':
            return pd.DataFrame(document_columns)
        try:
            return next(iter(document_client.filter(chembl_id=chembl_doc_id)), None)
        except Exception as e:
            #print(f"Error fetching document {chembl_doc_id}: {str(e)}")
            return pd.DataFrame(document_columns)
            
    activity = new_client.activity
    document_client = new_client.document
    assay_client = new_client.assay
       
    try:
        activities = activity.filter(
            molecule_chembl_id = chembl_id, 
            assay_type__in = assay_type_in,
            standard_type__in = ['AC50','EC50','IC50','Ki','MIC','GI50','TGI','Km','Kd','CC50','LC50'],
            pchembl_value__gte = pchembl_value_gte
        )

        # Convert the results to a list of dictionaries
        activities_list = list(activities)

        if not activities_list:
            empty_data = pd.DataFrame(columns=columns)
            empty_data['molecule_chembl_id'] = [chembl_id]
            return empty_data
            

        # Extract relevant fields and fetch confidence scores
        activity_data = []
        for m in activities_list:
            assay_info = next(iter(assay_client.filter(assay_chembl_id=m['assay_chembl_id'])), None)
            if assay_info and assay_info.get('confidence_score', 0) >= confidence_threshold:
                activity_dict = {field: m.get(field) for field in columns if field != 'confidence_score'}
                activity_dict['confidence_score'] = assay_info.get('confidence_score', None)
                activity_data.append(activity_dict)

        if not activity_data:  # If no activities meet the confidence threshold
            empty_data = pd.DataFrame(columns=columns)
            empty_data['molecule_chembl_id'] = [chembl_id]
            return empty_data
        
        activity_data = pd.DataFrame(activity_data)
        
        # Loop through all the ChEMBL documents
        document_data = [fetch_document_info(chembl_doc_id) for chembl_doc_id in activity_data['document_chembl_id']]
        document_data = pd.DataFrame(document_data)[document_columns]
        assay_information = pd.merge(activity_data, document_data, left_index=True, right_index=True)
        return assay_information
        
    except requests.exceptions.RequestException as e:
        # Handle HTTP request errors
        #print(f"An error occurred while fetching data: {e}")
        return pd.DataFrame({col: [None] for col in columns})

    except Exception as e:
        # Handle general errors
        #print(f"An unexpected error occurred: {e}")
        return pd.DataFrame({col: [None] for col in columns})

# %%
def surechembl_get_id(query, identifier):
    """
    Retrieve the SureChEMBL compound ID for a given query, using the specified identifier type.

    This function converts the query to an InChIKey if necessary, then queries the UniChem API
    to fetch the corresponding SureChEMBL compound ID.

    Parameters
    ----------
    query : str
        The compound identifier (SMILES, InChI, or InChIKey).
    identifier : str
        The type of the query, e.g., "smiles", "inchi", or "inchikey".

    Returns
    -------
        str or None:
        The SureChEMBL compound ID if found, otherwise None if not found or if an error occurs.
    """
    if identifier.lower() == "smiles":
        ChEMBL_mol_std = utils.smiles2inchiKey(query)
    elif identifier.lower() == "inchi":
        ChEMBL_mol_std = utils.inchi2inchiKey(query)
    elif identifier.lower() == "inchikey":
        ChEMBL_mol_std = query

    headers = {
        'accept': 'application/json',
    }

    url = f'https://www.ebi.ac.uk/unichem/rest/verbose_inchikey/{ChEMBL_mol_std}'
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        data = response.json()
        for entry in data:
            if entry.get('name') == 'surechembl':
                src_compound_id=entry.get('src_compound_id')
                return (src_compound_id)[0]
        return None  # If 'surechembl' is not found
    else:
        print(f"Failed to retrieve data: {response.status_code}")
        return None

# %%
def get_target_data(target_chembl_id):
    """
    Retrieve target information from ChEMBL for a given target ChEMBL ID.

    This function fetches the target's description, UniProt ID, and EC numbers using the ChEMBL web service client.
    If the target_chembl_id is empty, None, or NaN, or if the target is not found in ChEMBL, or if an error occurs,
    it returns a dictionary with error messages.

    Parameters
    ----------
    target_chembl_id : str, float, or None
        The ChEMBL ID of the target. If empty, None, or NaN, the function returns
        an error message without querying ChEMBL.

    Returns
    -------
    dict: A dictionary containing:
        - 'target_chembl_id': The input target ChEMBL ID 
        - 'Description': The target's preferred name (if found)
        - 'UniProt ID': The first UniProt accession found among the target components (if available)
        - 'EC Numbers': A semicolon-separated string of EC numbers (if available)
    """
    # Check if target_chembl_id is empty, None, or NaN
    if not target_chembl_id or (isinstance(target_chembl_id, float) and math.isnan(target_chembl_id)):
        #print(f"Warning: target_chembl_id is empty, None, or NaN")
        return {
            "target_chembl_id": str(target_chembl_id),
            "Description": "No target ChEMBL ID, skipping",
            "UniProt ID": "",
            "EC Numbers": ""
        }
    target = new_client.target
    
    try:
        result = target.get(target_chembl_id)
        
        if not result:
            print(f"No data found for {target_chembl_id}")
            return {
                "target_chembl_id": target_chembl_id,
                "Description": "No data found",
                "UniProt ID": "",
                "EC Numbers": ""
            }
        
        description = result.get('pref_name', '')
        components = result.get('target_components', [])
        
        uniprot_id = ""
        ec_numbers = []
        
        for component in components:
            accession = component.get('accession', [])
            if accession:
                uniprot_id = accession
            
            for synonym in component.get('target_component_synonyms', []):
                if synonym.get('syn_type') == "EC_NUMBER":
                    ec_numbers.append(synonym.get('component_synonym', ''))
        
        return {
            "target_chembl_id": target_chembl_id,
            "Description": description,
            "UniProt ID": uniprot_id,
            "EC Numbers": "; ".join(ec_numbers)
        }
    
    except Exception as e:
        #print(f"Error fetching data for {target_chembl_id}: {str(e)}")
        return {
            "target_chembl_id": target_chembl_id,
            "Description": "Error fetching data",
            "UniProt ID": "",
            "EC Numbers": ""
        }

# %%
def process_targets(targets_list):
    """
    Process a list of target ChEMBL IDs and collect their associated data.

    Parameters
    ----------
    targets_list : DataFrame
        DataFrame containing a column named 'target_chembl_id' with ChEMBL IDs of the targets to process.

    Returns
    -------
    DataFrame
        DataFrame where each row contains the processed data for a target.
    """
    target_data = []
    #Get the total number of targets
    total_targets = len(targets_list)
   
    #Initialize tqdm with position on the left
    pbar = tqdm(total=total_targets, desc="Processing targets", position=0, bar_format="{percentage:3.0f}%|{bar}|{desc}")
    
    #Iterate through each target in the list with a progress bar
    for i, (index, row) in enumerate (targets_list.iterrows(), start=1):
        target_chembl_id = row["target_chembl_id"]
        #print(drug_cid,drug_schembl)
        pbar.set_description(f"Processing target n.: {i}")
        single_target_data = get_target_data(target_chembl_id)
        target_data.append(single_target_data)
        
         #Update the progress bar
        pbar.update(1)

    return pd.DataFrame(target_data)

# %%
def get_protein_classifications(target_chembl_id):
    """
    Fetch protein classifications for a given ChEMBL target ID.

    Parameters
    ----------
    target_chembl_id : str
        ChEMBL Target ID (e.g., CHEMBL2933).

    Returns
    -------
    str
        A comma-separated string of protein classifications.
    """
    try:
        # Step 1: Fetch component_id
        target_url = f"https://www.ebi.ac.uk/chembl/api/data/target/{target_chembl_id}.json"
        target_data = requests.get(target_url).json()
        component_id = target_data['target_components'][0]['component_id']

        # Step 2: Get protein_classification_ids
        component_url = f"https://www.ebi.ac.uk/chembl/api/data/target_component/{component_id}.json"
        component_data = requests.get(component_url).json()
        protein_class_ids = [
            pc['protein_classification_id'] for pc in component_data.get('protein_classifications', [])
        ]

        # Step 3: Fetch protein classifications
        #classifications = []
        #for pc_id in protein_class_ids:
            #pc_url = f"https://www.ebi.ac.uk/chembl/api/data/protein_classification/{pc_id}.json"
            #pc_data = requests.get(pc_url).json()
            #classifications.append(pc_data['protein_class_desc'])  # Take the first word of pref_name

        # Combine classifications into a single string separated by commas
        #return ', '.join(classifications)
        return protein_class_ids[0] #Now I take the first value, need to implement another function for a list

    except Exception as e:
        #print(f"An error occurred while fetching protein classifications for {target_chembl_id}: {e}")
        return None

# %%
def trace_hierarchy(protein_class_id, hierarchy=None):
    """
    Recursively traces the hierarchy of a given protein_class_id.

    Parameters
    ----------
    protein_class_id : int
        The ID of the protein class to trace.
    hierarchy : list
        Accumulator for the hierarchy path.

    Returns
    -------
    list: 
        A list representing the traced protein hierarchy.
    """
    protein_class = new_client.protein_classification
    pclass=pd.DataFrame(protein_class)

    if hierarchy is None:
        hierarchy = []

    # Find the row corresponding to the given protein_class_id
    row = pclass[pclass["protein_class_id"] == protein_class_id]

    if row.empty:
        return hierarchy  # Return the accumulated hierarchy if no match is found

    # Add the current protein_class_id to the hierarchy
    current_name = row.iloc[0]["pref_name"]
    hierarchy.insert(0, current_name)  # Insert at the beginning to maintain root-to-leaf order

    # Get the parent_id of the current protein_class_id
    parent_id = row.iloc[0]["parent_id"]

    # If there's a parent_id, recursively trace it
    if not pd.isna(parent_id):
        return trace_hierarchy(int(parent_id), hierarchy)
    hierarchy = " > ".join(hierarchy[1:])
    return hierarchy

# %%
def trace_hierarchy_for_list(protein_class_ids):
    """
    Traces hierarchies for a list of protein_class_ids.

    Parameters
    ----------
    protein_class_ids : list
        A list of protein_class_ids (numbers) to trace.

    Returns
    -------
    dict: 
        A dictionary where keys are protein_class_ids and values are their hierarchies.
    """
    results = {}
    for protein_class_id in protein_class_ids:
        try:
            # Ensure the ID is treated as an integer
            protein_class_id = int(protein_class_id)
            hierarchy = trace_hierarchy(protein_class_id)
            results[protein_class_id] = " > ".join(hierarchy[1:])
        except ValueError:
            results[protein_class_id] = "Invalid ID"
    return results
