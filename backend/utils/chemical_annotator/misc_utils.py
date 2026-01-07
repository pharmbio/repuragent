"""
misc_utils.py

Miscellaneous helper functions used throughout the Chemical Annotator tool.

Author: Flavio Ballante

Contact: flavio.ballante@ki.se, flavioballante@gmail.com

Institution: CBCS-SciLifeLab-Karolinska Institutet

Year: 2025
"""

import pandas as pd
import re
from tqdm import tqdm
from .chembl_utils import chembl_get_id
from .chembl_utils import chembl_drug_annotations
from .chembl_utils import chembl_drug_indications
from .chembl_utils import chembl_assay_information
from .chembl_utils import chembl_mechanism_of_action
from .chembl_utils import surechembl_get_id
from .pubchem_utils import pubchem_get_cid

# %%
def _normalize_header(value) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def _infer_identifier_type(value: str) -> str | None:
    """
    Infer the identifier type ('smiles', 'inchi', 'inchikey') from a column-like string.
    """
    normalized = _normalize_header(value)
    if not normalized:
        return None
    if "smiles" in normalized:
        return "smiles"
    if "inchikey" in normalized or ("inchi" in normalized and "key" in normalized):
        return "inchikey"
    if "inchi" in normalized:
        return "inchi"
    return None


def resolve_identifier_column(compounds_list: pd.DataFrame, identifier: str) -> tuple[str, str]:
    """
    Resolve which DataFrame column contains the requested compound identifiers.

    - Case-insensitive for common identifier types ('SMILES', 'InChI', 'InChIKey').
    - Flexible for SMILES-like columns (e.g., 'SMILES', 'canonical_smiles').
    - Returns (column_name, identifier_type) where identifier_type is one of:
      'smiles', 'inchi', 'inchikey'.
    """
    if not isinstance(identifier, str) or not identifier.strip():
        raise ValueError("identifier must be a non-empty string")

    requested = identifier.strip()
    requested_type = requested.lower()
    if requested_type not in {"smiles", "inchi", "inchikey"}:
        inferred = _infer_identifier_type(requested)
        if inferred is None:
            raise ValueError(
                f"Unsupported identifier '{identifier}'. Expected one of: SMILES, InChI, InChIKey "
                f"(or a column name containing one of those)."
            )
        requested_type = inferred

    columns = list(compounds_list.columns)

    # Prefer exact case-insensitive match first.
    exact_matches = [col for col in columns if str(col).strip().lower() == requested_type]
    if exact_matches:
        return exact_matches[0], requested_type

    # Otherwise fall back to a "contains" match (e.g., canonical_smiles).
    def is_candidate(col) -> bool:
        normalized = _normalize_header(col)
        if requested_type == "smiles":
            return "smiles" in normalized
        if requested_type == "inchi":
            return "inchi" in normalized and "key" not in normalized
        return "inchikey" in normalized or ("inchi" in normalized and "key" in normalized)

    candidates = [col for col in columns if is_candidate(col)]
    if not candidates:
        available = ", ".join([str(c) for c in columns])
        raise ValueError(
            f"Could not find a '{requested_type}' column (case-insensitive) in input. "
            f"Available columns: {available}"
        )

    def rank(col) -> int:
        normalized = _normalize_header(col)
        if requested_type == "smiles":
            if normalized == "smiles":
                return 0
            if normalized in {"canonicalsmiles", "isomericsmiles", "standardsmiles"}:
                return 1
            return 2
        if requested_type == "inchi":
            if normalized == "inchi":
                return 0
            if normalized == "standardinchi":
                return 1
            return 2
        # inchikey
        if normalized == "inchikey":
            return 0
        if normalized in {"standardinchikey"}:
            return 1
        return 2

    best = min(candidates, key=lambda c: (rank(c), columns.index(c)))
    best_rank = rank(best)
    tied = [c for c in candidates if rank(c) == best_rank]
    if len(tied) > 1:
        tied_list = ", ".join([str(c) for c in tied])
        raise ValueError(
            f"Multiple possible '{requested_type}' columns found: {tied_list}. "
            f"Please keep only one '{requested_type}'-like column in the input."
        )

    return best, requested_type


def process_compounds(compounds_list, identifier, confidence_threshold=8, assay_type_in=['B', 'F'], pchembl_value_gte=6):
    """
    Process a list of compounds by retrieving drug annotations, indications, assay information,
    and mechanisms of action from ChEMBL and other databases.

    Parameters
    ----------
    compounds_list : DataFrame
        DataFrame containing a list of compounds.
    identifier : str
        Identifier type ('SMILES', 'InChI', 'InChIKey') or a column name containing one of those.
        Matching is case-insensitive and will also accept SMILES-like columns such as 'canonical_smiles'.
    confidence_threshol : int, optional
        Minimum confidence threshold for assay data. Defaults to 8.
    assay_type_in : list, optional
        List of assay types to include. Defaults to ['B', 'F'].
    pchembl_value_gte : int, optional
        Minimum pChEMBL value for assay data. Defaults to 6.

    Returns
    -------
        Three DataFrames containing:
            - all_drug_info: Merged drug annotations and indications
            - all_drug_assay: Merged assay information
            - all_MoA: Merged mechanisms of action
    """
    identifier_column, identifier_type = resolve_identifier_column(compounds_list, identifier)
    # Define an empty DataFrame to store all drug information
    all_drug_info = pd.DataFrame()
    all_drug_assay = pd.DataFrame()
    all_MoA = pd.DataFrame()
    # Get the total number of compounds
    total_compounds = len(compounds_list[identifier_column])
    # Initialize progress bar for tracking compound processing
    pbar = tqdm(total=total_compounds, desc="Processing compounds", position=0, bar_format="{percentage:3.0f}%|{bar}|{desc}")
    # Iterate through each compound in the list with a progress bar
    for i, (index, row) in enumerate (compounds_list.iterrows(), start=1):
        try:
            compound = row[identifier_column]
            chembl_id = chembl_get_id(compound, identifier_type)
            drug_cid = pubchem_get_cid(compound, identifier_type)
            drug_schembl = surechembl_get_id(compound, identifier_type)
            #print(drug_cid,drug_schembl)
            pbar.set_description(f"Processing compound n.: {i}")
            # Get drug annotations and indications
            drug_annot = chembl_drug_annotations(chembl_id)
            drug_annot_selected = drug_annot[['molecule_chembl_id', 'canonical_smiles','standard_inchi', 'standard_inchi_key']] # Select specific columns from drug_annot
            drug_indic = chembl_drug_indications(chembl_id)
            drug_assay = chembl_assay_information(chembl_id, confidence_threshold, assay_type_in, pchembl_value_gte)
            drug_MoA = chembl_mechanism_of_action(chembl_id)
            # If no ChEMBL ID is found, ensure molecule_chembl_id is treated as a string to avoid type issues in merges
            # Merge the annotations and indications on 'molecule_chembl_id'
            if pd.isnull(chembl_id) == True: #if no chembl id is found convert NaN (float) to type string
                drug_indic['molecule_chembl_id'] = drug_indic['molecule_chembl_id'].astype(str)
                drug_assay['molecule_chembl_id'] = drug_assay['molecule_chembl_id'].astype(str)
                drug_MoA['molecule_chembl_id'] = drug_MoA['molecule_chembl_id'].astype(str)
            # Merge annotations and indications on molecule_chembl_id
            drug_info = drug_annot.merge(drug_indic, on='molecule_chembl_id', how='left')
            # Merge assay information with selected annotations
            drug_assay = drug_annot_selected.merge(drug_assay, on='molecule_chembl_id', how='left')
            # Merge mechanism of action with selected annotations
            drug_MoA = drug_annot_selected.merge(drug_MoA, on='molecule_chembl_id', how='left')
            # Add the drug_cid and drug_schembl to drug_info DataFrame
            drug_info['drug_cid'] = drug_cid
            drug_info['drug_schembl'] = drug_schembl
            drug_assay['drug_cid'] = drug_cid
            drug_assay['drug_schembl'] = drug_schembl
            # Merge with the current compound's row data
            merged_info = pd.concat([row.to_frame().T] * len(drug_info), ignore_index=True)
            merged_info = pd.concat([merged_info.reset_index(drop=True), drug_info.reset_index(drop=True)], axis=1)
            merged_assay = pd.concat([row.to_frame().T] * len(drug_assay), ignore_index=True)
            merged_assay = pd.concat([merged_assay.reset_index(drop=True), drug_assay.reset_index(drop=True)], axis=1)  
            merged_MoA = pd.concat([row.to_frame().T] * len(drug_MoA), ignore_index=True)
            merged_MoA = pd.concat([merged_MoA.reset_index(drop=True), drug_MoA.reset_index(drop=True)], axis=1)
            # Append the processed data to the result DataFrames
            all_drug_info = pd.concat([all_drug_info, merged_info], ignore_index=True)
            all_drug_assay = pd.concat([all_drug_assay, merged_assay], ignore_index=True)
            all_MoA = pd.concat([all_MoA, drug_MoA], ignore_index=True)
        
        except Exception as e:
            print(f"Warning: while processing compound {i}: {e}")
        # Update the progress bar
        pbar.update(1)  
    # Return the three DataFrames containing processed information
    return all_drug_info, all_drug_assay, all_MoA
