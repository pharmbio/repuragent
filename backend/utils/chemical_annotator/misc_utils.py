'''misc_utils.py

Miscellaneous helper functions used throughout the Chemical Annotator tool.

Author: Flavio Ballante

Contact: flavio.ballante@ki.se, flavioballante@gmail.com

Institution: CBCS-SciLifeLab-Karolinska Institutet

Year: 2025
'''

from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import re
import time
from urllib.parse import quote

import pandas as pd
import pubchempy as pcp
import requests
from chembl_webresource_client.new_client import new_client
from tqdm import tqdm

from .chembl_utils import chembl_assay_information
from .chembl_utils import chembl_drug_annotations
from .chembl_utils import chembl_drug_indications
from .chembl_utils import chembl_get_id
from .chembl_utils import chembl_mechanism_of_action
from .chembl_utils import surechembl_get_id
from .pubchem_utils import pubchem_get_cid

CACTUS_BASE = "https://cactus.nci.nih.gov/chemical/structure"
molecule = new_client.molecule

# %%
def _normalize_header(value) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).strip().lower())


def _infer_identifier_type(value: str) -> str | None:
    '''Infer the identifier type ('smiles', 'inchi', 'inchikey') from a column-like string.

    Parameters:
    ---------
    value (str): a column name, or a value from it.

    Returns:
    ----------
    identifier_type (str): `smiles`, `inchi` or `inchikey`, or None when it looks like none of them.
    '''

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


def find_smiles_column(compounds_list: pd.DataFrame) -> str | None:
    '''Return the first column name that contains 'smiles' (case-insensitive).

    Parameters:
    ---------
    compounds_list (Pandas DataFrame): the table to inspect.

    Returns:
    ----------
    column (str): the first column whose name contains 'smiles', case-insensitively, or None.
    '''

    for col in compounds_list.columns:
        if "smiles" in str(col).strip().lower():
            return col
    return None


def resolve_identifier_column(compounds_list: pd.DataFrame, identifier: str) -> tuple[str, str]:
    '''Resolve which DataFrame column contains the requested compound identifiers.

    - Case-insensitive for common identifier types ('SMILES', 'InChI', 'InChIKey').
    - Flexible for SMILES-like columns (e.g., 'SMILES', 'canonical_smiles').
    - Returns (column_name, identifier_type) where identifier_type is one of:
      'smiles', 'inchi', 'inchikey'.

    Parameters:
    ---------
    compounds_list (Pandas DataFrame): the table to inspect.
    identifier (str): the identifier type the caller asked for.

    Returns:
    ----------
    resolved (tuple): `(column name, identifier type)` for the requested identifiers.
    '''

    if not isinstance(identifier, str) or not identifier.strip():
        raise ValueError("identifier must be a non-empty string")

    requested = identifier.strip()
    columns = list(compounds_list.columns)

    # If identifier matches a column name exactly (case-insensitive), use it directly.
    exact_column_matches = [col for col in columns if str(col).strip().lower() == requested.lower()]
    if exact_column_matches:
        column_name = exact_column_matches[0]
        inferred = _infer_identifier_type(column_name)
        if inferred is None:
            raise ValueError(
                f"Column '{column_name}' does not look like a SMILES/InChI/InChIKey identifier."
            )
        return column_name, inferred

    requested_type = requested.lower()
    if requested_type not in {"smiles", "inchi", "inchikey"}:
        inferred = _infer_identifier_type(requested)
        if inferred is None:
            raise ValueError(
                f"Unsupported identifier '{identifier}'. Expected one of: SMILES, InChI, InChIKey "
                f"(or a column name containing one of those)."
            )
        requested_type = inferred

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


def auto_detect_identifier_column(compounds_list: pd.DataFrame) -> tuple[str, str]:
    '''Try to detect a single identifier column, preferring SMILES, then InChIKey, then InChI.
    Returns (column_name, identifier_type).

    Parameters:
    ---------
    compounds_list (Pandas DataFrame): the table to inspect.

    Returns:
    ----------
    resolved (tuple): `(column name, identifier type)`, preferring SMILES, then InChIKey, then InChI.
    '''

    for candidate in ("smiles", "inchikey", "inchi"):
        try:
            return resolve_identifier_column(compounds_list, candidate)
        except ValueError:
            continue
    # Fallback: detect other identifier-like columns by name (e.g., chembl, pubchem, cas, cid)
    def looks_like_alt_identifier(col) -> bool:
        normalized = _normalize_header(col)
        if not normalized:
            return False
        if "chembl" in normalized:
            return True
        if "pubchem" in normalized:
            return True
        if normalized in {"cid", "cas"}:
            return True
        if "casrn" in normalized or "casnumber" in normalized or "casid" in normalized:
            return True
        if "pubchemcid" in normalized or "pubchemcid" in normalized:
            return True
        if "chemblid" in normalized:
            return True
        return False

    alt_candidates = [col for col in compounds_list.columns if looks_like_alt_identifier(col)]
    if len(alt_candidates) == 1:
        col = alt_candidates[0]
        normalized = _normalize_header(col)
        if "chembl" in normalized:
            return col, "chembl"
        return col, "any"
    if len(alt_candidates) > 1:
        alt_list = ", ".join([str(c) for c in alt_candidates])
        raise ValueError(
            "Multiple possible identifier columns found: "
            f"{alt_list}. Please keep only one identifier column in the input."
        )
    available = ", ".join([str(c) for c in compounds_list.columns])
    raise ValueError(
        "Could not auto-detect an identifier column. "
        "Expected a column containing SMILES, InChIKey, or InChI. "
        f"Available columns: {available}"
    )


def _clean_identifier(value):
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text if text else None


def _is_chembl_id(value: str) -> bool:
    return bool(re.fullmatch(r"CHEMBL\d+", value.upper()))


def _looks_like_inchikey(value: str) -> bool:
    return bool(re.fullmatch(r"[A-Z]{14}-[A-Z]{10}-[A-Z]", value))


@lru_cache(maxsize=50_000)
def resolve_smiles_any(
    identifier: str,
    *,
    identifier_type: str | None = None,
    pause_s: float = 0.0,
    timeout_s: float = 15.0,
) -> str | None:
    '''Resolve many identifier types to canonical SMILES.
    Order:
      1) ChEMBL IDs via ChEMBL API
      2) CACTUS resolver
      3) PubChem (CID / InChIKey / name)

    Parameters:
    ---------
    identifier (str): the value to resolve.
    identifier_type (str): its type, or None to infer it.
    pause_s (float): seconds to wait between calls, to stay inside the service's rate limit.
    timeout_s (float): how long to wait for one lookup.

    Returns:
    ----------
    smiles (str): the canonical SMILES, or None when nothing resolved it.
    '''

    ident = _clean_identifier(identifier)
    if not ident:
        return None

    ident_u = ident.upper()

    if identifier_type == "chembl" or _is_chembl_id(ident_u):
        try:
            mol = molecule.get(ident_u)
            smiles = mol.get("molecule_structures", {}).get("canonical_smiles")
            if smiles:
                if pause_s:
                    time.sleep(pause_s)
                return smiles
        except Exception:
            pass

    try:
        url = f"{CACTUS_BASE}/{quote(ident)}/smiles"
        response = requests.get(url, timeout=timeout_s, headers={"User-Agent": "smiles-resolver/1.0"})
        if response.ok:
            text = response.text.strip()
            if text and "not found" not in text.lower() and "<html" not in text.lower():
                if pause_s:
                    time.sleep(pause_s)
                return text
    except requests.RequestException:
        pass

    try:
        if ident.isdigit():
            compound = pcp.Compound.from_cid(int(ident))
            return getattr(compound, "canonical_smiles", None)

        if _looks_like_inchikey(ident_u):
            compounds = pcp.get_compounds(ident_u, namespace="inchikey")
        else:
            compounds = pcp.get_compounds(ident, namespace="name")

        if compounds:
            return getattr(compounds[0], "canonical_smiles", None)
    except Exception:
        pass

    return None


def _normalize_merge_key(dataframe: pd.DataFrame, column_name: str) -> pd.DataFrame:
    normalized = dataframe.copy()
    if column_name in normalized.columns:
        normalized[column_name] = normalized[column_name].astype(object)
    return normalized


def _merge_compound_row(row: dict, dataframe: pd.DataFrame) -> pd.DataFrame:
    base_rows = pd.DataFrame([row] * len(dataframe)).reset_index(drop=True)
    return pd.concat([base_rows, dataframe.reset_index(drop=True)], axis=1)


def _compound_result_key(value) -> str | None:
    cleaned = _clean_identifier(value)
    return cleaned if cleaned is not None else None


def _fetch_compound_bundle(
    compound,
    identifier_type,
    confidence_threshold,
    assay_type_in,
    pchembl_value_gte,
):
    compound_key = _compound_result_key(compound)
    if compound_key is None:
        empty = pd.DataFrame()
        return {"drug_info": empty, "drug_assay": empty, "drug_moa": empty}

    chembl_id = chembl_get_id(compound_key, identifier_type)
    drug_cid = pubchem_get_cid(compound_key, identifier_type)
    drug_schembl = surechembl_get_id(compound_key, identifier_type)

    drug_annot = _normalize_merge_key(chembl_drug_annotations(chembl_id), "molecule_chembl_id")
    drug_annot_selected = drug_annot[
        ["molecule_chembl_id", "canonical_smiles", "standard_inchi", "standard_inchi_key"]
    ]
    drug_indic = _normalize_merge_key(chembl_drug_indications(chembl_id), "molecule_chembl_id")
    drug_assay = _normalize_merge_key(
        chembl_assay_information(
            chembl_id,
            confidence_threshold=confidence_threshold,
            assay_type_in=assay_type_in,
            pchembl_value_gte=pchembl_value_gte,
        ),
        "molecule_chembl_id",
    )
    drug_moa = _normalize_merge_key(
        chembl_mechanism_of_action(chembl_id),
        "molecule_chembl_id",
    )

    if pd.isna(chembl_id):
        for dataframe in (drug_indic, drug_assay, drug_moa):
            if "molecule_chembl_id" in dataframe.columns:
                dataframe["molecule_chembl_id"] = dataframe["molecule_chembl_id"].astype(str)

    drug_info = drug_annot.merge(drug_indic, on="molecule_chembl_id", how="left")
    drug_assay = drug_annot_selected.merge(drug_assay, on="molecule_chembl_id", how="left")
    drug_moa = drug_annot_selected.merge(drug_moa, on="molecule_chembl_id", how="left")

    drug_info["drug_cid"] = drug_cid
    drug_info["drug_schembl"] = drug_schembl
    drug_assay["drug_cid"] = drug_cid
    drug_assay["drug_schembl"] = drug_schembl

    return {
        "drug_info": drug_info,
        "drug_assay": drug_assay,
        "drug_moa": drug_moa,
    }


def process_compounds(compounds_list, identifier, confidence_threshold=8, assay_type_in=['B', 'F'], pchembl_value_gte=6):
    '''Process a list of compounds by retrieving drug annotations, indications, assay information,
    and mechanisms of action from ChEMBL and other databases.

    Parameters:
    ---------
    compounds_list (Pandas DataFrame): DataFrame containing a list of compounds.
    identifier (str): Identifier type ('SMILES', 'InChI', 'InChIKey') or a column name containing one of those. Matching is case-insensitive and will also accept SMILES-like columns such as 'canonical_smiles'.
    confidence_threshol (int, optional): Minimum confidence threshold for assay data. Defaults to 8.
    assay_type_in (list, optional): List of assay types to include. Defaults to ['B', 'F'].
    pchembl_value_gte (int, optional): Minimum pChEMBL value for assay data. Defaults to 6.

    Returns:
    ----------
    frames (tuple): Three DataFrames containing:
    - all_drug_info: Merged drug annotations and indications
    - all_drug_assay: Merged assay information
    - all_MoA: Merged mechanisms of action
    '''

    identifier_column, identifier_type = resolve_identifier_column(compounds_list, identifier)
    assay_type_in = tuple(assay_type_in)
    rows = compounds_list.to_dict("records")

    unique_compounds = {}
    for row in rows:
        compound = row.get(identifier_column)
        compound_key = _compound_result_key(compound)
        if compound_key not in unique_compounds:
            unique_compounds[compound_key] = compound

    pbar = tqdm(
        total=len(unique_compounds),
        desc="Processing compounds",
        position=0,
        bar_format="{percentage:3.0f}%|{bar}|{desc}",
    )

    compound_results = {}
    max_workers = min(8, len(unique_compounds)) if unique_compounds else 1
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_key = {
            executor.submit(
                _fetch_compound_bundle,
                compound,
                identifier_type,
                confidence_threshold,
                assay_type_in,
                pchembl_value_gte,
            ): compound_key
            for compound_key, compound in unique_compounds.items()
        }

        for index, future in enumerate(as_completed(future_to_key), start=1):
            compound_key = future_to_key[future]
            pbar.set_description(f"Processing compound n.: {index}")
            try:
                compound_results[compound_key] = future.result()
            except Exception as exc:
                print(f"Warning: while processing compound {index}: {exc}")
                compound_results[compound_key] = {
                    "drug_info": pd.DataFrame(),
                    "drug_assay": pd.DataFrame(),
                    "drug_moa": pd.DataFrame(),
                }
            pbar.update(1)

    all_drug_info = []
    all_drug_assay = []
    all_moa = []
    for row in rows:
        result = compound_results[_compound_result_key(row.get(identifier_column))]
        if not result["drug_info"].empty:
            all_drug_info.append(_merge_compound_row(row, result["drug_info"]))
        if not result["drug_assay"].empty:
            all_drug_assay.append(_merge_compound_row(row, result["drug_assay"]))
        if not result["drug_moa"].empty:
            all_moa.append(result["drug_moa"].copy())

    return (
        pd.concat(all_drug_info, ignore_index=True) if all_drug_info else pd.DataFrame(),
        pd.concat(all_drug_assay, ignore_index=True) if all_drug_assay else pd.DataFrame(),
        pd.concat(all_moa, ignore_index=True) if all_moa else pd.DataFrame(),
    )
