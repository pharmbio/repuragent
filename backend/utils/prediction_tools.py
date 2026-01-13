import json
import os
import re
import shlex
import subprocess
from typing import List, Union
from urllib.parse import quote

import pandas as pd
import requests
from langchain_core.tools import tool
from tqdm import tqdm

from backend.utils.chemical_annotator.chembl_utils import chembl_get_id
from backend.utils.output_paths import task_file_path


def smiles_csv(smiles_input: Union[str, List[str]]) -> Union[str, os.PathLike]:
    """
    Standardize various SMILES input formats into a single-column DataFrame and export to CSV.

    Args:
        smiles_input: A SMILES string, a comma-separated string of SMILES,
                      a list of SMILES strings, or a path to a CSV/TSV file
                      with a 'smiles' column.

    Returns:
        Path to the CSV file containing a 'smiles' column.
    """
    smiles_list = []

    # Case 1: File path input
    if isinstance(smiles_input, str) and os.path.isfile(smiles_input):
        ext = os.path.splitext(smiles_input)[-1].lower()
        if ext not in [".csv", ".tsv"]:
            return "Error: Only CSV or TSV files are supported"
        sep = "\t" if ext == ".tsv" else ","
        df = pd.read_csv(smiles_input, sep=sep)
        df.columns = [col.lower() for col in df.columns]
        smiles_columns = [col for col in df.columns if "smiles" in col]
        if not smiles_columns:
            return "Error: No 'smiles' column found in the file"
        smiles_list = df[smiles_columns[0]].dropna().astype(str).tolist()

    # Case 2: SMILES string (single or comma-separated)
    elif isinstance(smiles_input, str):
        smiles_list = [s.strip() for s in smiles_input.split(",") if s.strip()]

    # Case 3: List input
    elif isinstance(smiles_input, list):
        smiles_list = [s.strip() for s in smiles_input if isinstance(s, str) and s.strip()]

    else:
        return "Error: Input must be a SMILES string, a list of SMILES, or a file path"

    if not smiles_list:
        return "Error: No valid SMILES strings provided"

    # Create DataFrame
    df = pd.DataFrame(smiles_list, columns=["smiles"])

    # Export to temporary file
    output_path = "data/modelling_data.csv"
    df.to_csv(output_path, index=False)

    return output_path


def format_clf_label(label):
    if label == "{0}":
        return 0
    elif label == "{1}":
        return 1
    elif label == "{0, 1}":
        return 0.5
    else:
        return label


def format_clf_df(df, column):
    df["tmp"] = df[column].apply(format_clf_label)
    df[column] = df["tmp"]
    df = df.drop(["tmp"], axis=1)
    return df


def _prepare_output_file(filename: str):
    """Return the path for a classifier artifact inside the active task directory."""
    path = task_file_path(filename)
    if path.exists():
        path.unlink()
    return path


def _single_row_to_dict(df: pd.DataFrame) -> dict:
    """Convert a single-row DataFrame into a JSON-friendly dict."""
    if df.empty:
        return {}
    row = df.iloc[0].to_dict()
    normalized = {}
    for key, value in row.items():
        if pd.isna(value):
            normalized[key] = None
        elif hasattr(value, "item"):
            normalized[key] = value.item()
        else:
            normalized[key] = value
    return normalized


@tool
def CYP3A4_classifier(smiles_input: Union[str, List[str]]):
    """Pretrained classification model for CYP3A4 inhibition prediction.
    Args:
        smiles_input: A SMILES string, a comma-separated string of SMILES,
                      a list of SMILES strings, or a path to a CSV/TSV file
                      with a 'smiles' column.
    Returns:
        Absolute path to the generated `CYP3A4_results.csv` inside the task output folder."""

    output_path = _prepare_output_file("CYP3A4_results.csv")
    data_path = smiles_csv(smiles_input)
    output_str = shlex.quote(str(output_path))
    cmd = f"java -jar models/CPSign/cpsign-2.0.0-fatjar.jar predict \
    --model models/CYP3A4_clf_trained.jar \
    --predict-file CSV {data_path} \
    --confidences 0.79 \
    --output-format CSV \
    --output {output_str}"
    _ = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    # Modify output columns
    df = pd.read_csv(output_path)
    df.columns = ["smiles", "p_value_0", "p_value_1", "CYP3A4_inhibition"]
    df = format_clf_df(df, "CYP3A4_inhibition")
    df.to_csv(output_path, index=False)

    if len(df.index) == 1:
        return _single_row_to_dict(df)
    return str(output_path)


@tool
def hERG_classifier(smiles_input: Union[str, List[str]]):
    """Pretrained classification model for hERG inhibition prediction.
    Args:
        smiles_input: A SMILES string, a comma-separated string of SMILES,
                      a list of SMILES strings, or a path to a CSV/TSV file
                      with a 'smiles' column.
    Returns:
        Absolute path to `hERG_results.csv` saved for this task."""

    output_path = _prepare_output_file("hERG_results.csv")
    data_path = smiles_csv(smiles_input)
    output_str = shlex.quote(str(output_path))
    cmd = f"java -jar models/CPSign/cpsign-2.0.0-fatjar.jar predict \
    --model models/hERG_clf_trained.jar \
    --predict-file CSV {data_path} \
    --confidences 0.78 \
    --output-format CSV \
    --output {output_str}"
    _ = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    # Modify output columns
    df = pd.read_csv(output_path)
    df.columns = ["smiles", "p_value_0", "p_value_1", "hERG_inhibition"]
    df = format_clf_df(df, "hERG_inhibition")
    df.to_csv(output_path, index=False)

    if len(df.index) == 1:
        return _single_row_to_dict(df)
    return str(output_path)


@tool
def AMES_classifier(smiles_input: Union[str, List[str]]):
    """Pretrained classification model for AMES inhibition prediction.
    Args:
        smiles_input: A SMILES string, a comma-separated string of SMILES,
                      a list of SMILES strings, or a path to a CSV/TSV file
                      with a 'smiles' column.
    Returns:
        Absolute path to `AMES_results.csv` for this task."""

    output_path = _prepare_output_file("AMES_results.csv")
    data_path = smiles_csv(smiles_input)
    output_str = shlex.quote(str(output_path))
    cmd = f"java -jar models/CPSign/cpsign-2.0.0-fatjar.jar predict \
    --model models/AMES_clf_trained.jar \
    --predict-file CSV {data_path} \
    --confidences 0.81 \
    --output-format CSV \
    --output {output_str}"
    _ = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    # Modify output columns
    df = pd.read_csv(output_path)
    df.columns = ["smiles", "p_value_0", "p_value_1", "AMES_mutagenic"]
    df = format_clf_df(df, "AMES_mutagenic")
    df.to_csv(output_path, index=False)

    if len(df.index) == 1:
        return _single_row_to_dict(df)
    return str(output_path)


@tool
def PGP_classifier(smiles_input: Union[str, List[str]]):
    """Pretrained classification model for P-glycoprotein inhibition prediction.
    Args:
        smiles_input: A SMILES string, a comma-separated string of SMILES,
                      a list of SMILES strings, or a path to a CSV/TSV file
                      with a 'smiles' column.
    Returns:
        Absolute path to `PGP_results.csv` saved in the task folder."""

    output_path = _prepare_output_file("PGP_results.csv")
    data_path = smiles_csv(smiles_input)
    output_str = shlex.quote(str(output_path))
    cmd = f"java -jar models/CPSign/cpsign-2.0.0-fatjar.jar predict \
    --model models/PGP_clf_trained.jar \
    --predict-file CSV {data_path} \
    --confidences 0.83 \
    --output-format CSV \
    --output {output_str}"
    _ = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    # Modify output columns
    df = pd.read_csv(output_path)
    df.columns = ["smiles", "p_value_0", "p_value_1", "PGP_inhibition"]
    df = format_clf_df(df, "PGP_inhibition")
    df.to_csv(output_path, index=False)

    if len(df.index) == 1:
        return _single_row_to_dict(df)
    return str(output_path)


@tool
def Solubility_regressor(smiles_input: Union[str, List[str]]):
    """Pretrained regression model for solubility prediction with single point prediction and prediction interval.
    Args:
        smiles_input: A SMILES string, a comma-separated string of SMILES,
                      a list of SMILES strings, or a path to a CSV/TSV file
                      with a 'smiles' column.
    Returns:
        Path to `Solubility_results.csv` stored for this task."""

    output_path = _prepare_output_file("Solubility_results.csv")
    data_path = smiles_csv(smiles_input)
    output_str = shlex.quote(str(output_path))
    cmd = f"java -jar models/CPSign/cpsign-2.0.0-fatjar.jar predict \
    --model models/Solubility_rgs_trained.jar \
    --predict-file CSV {data_path} \
    --confidences 0.71 \
    --output-format CSV \
    --output {output_str}"
    _ = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    # Modify output columns
    df = pd.read_csv(output_path)
    df.columns = [
        "smiles",
        "logS",
        "logS_lower_bound",
        "logS_upper_bound",
        "Capped_logS_lower_bound",
        "Capped_logS_upper_bound",
    ]
    df.to_csv(output_path, index=False)

    if len(df.index) == 1:
        return _single_row_to_dict(df)
    return str(output_path)


@tool
def Lipophilicity_regressor(smiles_input: Union[str, List[str]]):
    """Pretrained regression model for lipophilicity prediction with single point prediction.
    Args:
        smiles_input: A SMILES string, a comma-separated string of SMILES,
                      a list of SMILES strings, or a path to a CSV/TSV file
                      with a 'smiles' column.
    Returns:
        Path to `Lipophilicity_results.csv` inside the task directory."""

    output_path = _prepare_output_file("Lipophilicity_results.csv")
    data_path = smiles_csv(smiles_input)

    # Generate logP
    from rdkit import Chem
    from rdkit.Chem import Crippen

    df = pd.read_csv(data_path)
    df["mol"] = df["smiles"].apply(lambda smi: Chem.MolFromSmiles(smi))
    df["logP"] = df["mol"].apply(lambda mol: Crippen.MolLogP(mol))
    df = df.drop(["mol"], axis=1)

    # Modify output columns
    df.to_csv(output_path, index=False)

    if len(df.index) == 1:
        return _single_row_to_dict(df)
    return str(output_path)


@tool
def PAMPA_classifier(smiles_input: Union[str, List[str]]):
    """Pretrained classification model for PAMPA permeability assay.
    Args:
        smiles_input: A SMILES string, a comma-separated string of SMILES,
                      a list of SMILES strings, or a path to a CSV/TSV file
                      with a 'smiles' column.
    Returns:
        Path to `PAMPA_results.csv` for the current task."""

    output_path = _prepare_output_file("PAMPA_results.csv")
    data_path = smiles_csv(smiles_input)
    output_str = shlex.quote(str(output_path))
    cmd = f"java -jar models/CPSign/cpsign-2.0.0-fatjar.jar predict \
    --model models/PAMPA_clf_trained.jar \
    --predict-file CSV {data_path} \
    --confidences 0.75 \
    --output-format CSV \
    --output {output_str}"
    _ = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    # Modify output columns
    df = pd.read_csv(output_path)
    df.columns = ["smiles", "p_value_0", "p_value_1", "PAMPA_permeability"]
    df = format_clf_df(df, "PAMPA_permeability")
    df.to_csv(output_path, index=False)

    if len(df.index) == 1:
        return _single_row_to_dict(df)
    return str(output_path)


@tool
def BBB_classifier(smiles_input: Union[str, List[str]]):
    """Pretrained classification model for assessing Blood Brain Barrier penetration ability.
    Args:
        smiles_input: A SMILES string, a comma-separated string of SMILES,
                      a list of SMILES strings, or a path to a CSV/TSV file
                      with a 'smiles' column.
    Returns:
        Path to `BBB_results.csv` stored under the task output directory."""

    output_path = _prepare_output_file("BBB_results.csv")
    data_path = smiles_csv(smiles_input)
    output_str = shlex.quote(str(output_path))
    cmd = f"java -jar models/CPSign/cpsign-2.0.0-fatjar.jar predict \
    --model models/BBB_clf_trained.jar \
    --predict-file CSV {data_path} \
    --confidences 0.80 \
    --output-format CSV \
    --output {output_str}"
    _ = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    # Modify output columns
    df = pd.read_csv(output_path)
    df.columns = ["smiles", "p_value_0", "p_value_1", "BBB_penetration"]
    df = format_clf_df(df, "BBB_penetration")
    df.to_csv(output_path, index=False)

    if len(df.index) == 1:
        return _single_row_to_dict(df)
    return str(output_path)


@tool
def CYP2C19_classifier(smiles_input: Union[str, List[str]]):
    """Pretrained classification model for CYP2C19 inhibition prediction.
    Args:
        smiles_input: A SMILES string, a comma-separated string of SMILES,
                      a list of SMILES strings, or a path to a CSV/TSV file
                      with a 'smiles' column.
    Returns:
        Path to `CYP2C19_results.csv` for this task."""

    output_path = _prepare_output_file("CYP2C19_results.csv")
    data_path = smiles_csv(smiles_input)
    output_str = shlex.quote(str(output_path))
    cmd = f"java -jar models/CPSign/cpsign-2.0.0-fatjar.jar predict \
    --model models/CYP2C19_clf_trained.jar \
    --predict-file CSV {data_path} \
    --confidences 0.80 \
    --output-format CSV \
    --output {output_str}"
    _ = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    # Modify output columns
    df = pd.read_csv(output_path)
    df.columns = ["smiles", "p_value_0", "p_value_1", "CYP2C19_inhibition"]
    df = format_clf_df(df, "CYP2C19_inhibition")
    df.to_csv(output_path, index=False)

    if len(df.index) == 1:
        return _single_row_to_dict(df)
    return str(output_path)


@tool
def CYP2D6_classifier(smiles_input: Union[str, List[str]]):
    """Pretrained classification model for CYP2D6 inhibition prediction.
    Args:
        smiles_input: A SMILES string, a comma-separated string of SMILES,
                      a list of SMILES strings, or a path to a CSV/TSV file
                      with a 'smiles' column.
    Returns:
        Path to `CYP2D6_results.csv` stored for this task."""

    output_path = _prepare_output_file("CYP2D6_results.csv")
    data_path = smiles_csv(smiles_input)
    output_str = shlex.quote(str(output_path))
    cmd = f"java -jar models/CPSign/cpsign-2.0.0-fatjar.jar predict \
    --model models/CYP2D6_clf_trained.jar \
    --predict-file CSV {data_path} \
    --confidences 0.80 \
    --output-format CSV \
    --output {output_str}"
    _ = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    # Modify output columns
    df = pd.read_csv(output_path)
    df.columns = ["smiles", "p_value_0", "p_value_1", "CYP2D6_inhibition"]
    df = format_clf_df(df, "CYP2D6_inhibition")
    df.to_csv(output_path, index=False)

    if len(df.index) == 1:
        return _single_row_to_dict(df)
    return str(output_path)


@tool
def CYP1A2_classifier(smiles_input: Union[str, List[str]]):
    """Pretrained classification model for CYP1A2 inhibition prediction.
    Args:
        smiles_input: A SMILES string, a comma-separated string of SMILES,
                      a list of SMILES strings, or a path to a CSV/TSV file
                      with a 'smiles' column.
    Returns:
        Path to `CYP1A2_results.csv` stored for this task."""

    output_path = _prepare_output_file("CYP1A2_results.csv")
    data_path = smiles_csv(smiles_input)
    output_str = shlex.quote(str(output_path))
    cmd = f"java -jar models/CPSign/cpsign-2.0.0-fatjar.jar predict \
    --model models/CYP1A2_clf_trained.jar \
    --predict-file CSV {data_path} \
    --confidences 0.84 \
    --output-format CSV \
    --output {output_str}"
    _ = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    # Modify output columns
    df = pd.read_csv(output_path)
    df.columns = ["smiles", "p_value_0", "p_value_1", "CYP1A2_inhibition"]
    df = format_clf_df(df, "CYP1A2_inhibition")
    df.to_csv(output_path, index=False)

    if len(df.index) == 1:
        return _single_row_to_dict(df)
    return str(output_path)


@tool
def CYP2C9_classifier(smiles_input: Union[str, List[str]]):
    """Pretrained classification model for CYP2C9 inhibition prediction.
    Args:
        smiles_input: A SMILES string, a comma-separated string of SMILES,
                      a list of SMILES strings, or a path to a CSV/TSV file
                      with a 'smiles' column.
    Returns:
        Path to `CYP2C9_results.csv` saved for this conversation."""

    output_path = _prepare_output_file("CYP2C9_results.csv")
    data_path = smiles_csv(smiles_input)
    output_str = shlex.quote(str(output_path))
    cmd = f"java -jar models/CPSign/cpsign-2.0.0-fatjar.jar predict \
    --model models/CYP2C9_clf_trained.jar \
    --predict-file CSV {data_path} \
    --confidences 0.80 \
    --output-format CSV \
    --output {output_str}"
    _ = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    # Modify output columns
    df = pd.read_csv(output_path)
    df.columns = ["smiles", "p_value_0", "p_value_1", "CYP2C9_inhibition"]
    df = format_clf_df(df, "CYP2C9_inhibition")
    df.to_csv(output_path, index=False)

    if len(df.index) == 1:
        return _single_row_to_dict(df)
    return str(output_path)


### ML SMILES
BASE_URL = "https://repurposedrugs.aittokallio.group"
_JSON_RE = re.compile(r"\{.*\}\s*$", re.DOTALL)


def _looks_like_smiles(s: str) -> bool:
    s = s.strip()
    if not s:
        return False
    tokens = ["=", "#", "(", ")", "[", "]", "@", "\\", "/", "+", "-", "%"]
    if any(t in s for t in tokens):
        return True
    if any(ch.isdigit() for ch in s):
        return True
    return False


def _pubchem_name_to_smiles(name: str, timeout_s: int = 30) -> str:
    name = name.strip()
    if not name:
        raise ValueError("Empty name cannot be resolved via PubChem.")
    url = (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/"
        f"{quote(name)}/property/IsomericSMILES/JSON"
    )
    r = requests.get(url, timeout=timeout_s)
    if r.status_code == 404:
        raise ValueError(f"PubChem could not find a compound for name: {name!r}")
    r.raise_for_status()
    data = r.json()
    try:
        return data["PropertyTable"]["Properties"][0]["SMILES"]
    except Exception as e:
        raise ValueError(f"Unexpected PubChem response for {name!r}: {e}")


def _call_runqc(name: str, smiles: str, timeout_s: int = 120) -> dict:
    url = f"{BASE_URL.rstrip('/')}/runQC.php"
    r = requests.get(url, params={"name": name, "smiles": smiles}, timeout=timeout_s)
    r.raise_for_status()
    text = r.text.strip()
    m = _JSON_RE.search(text)
    if not m:
        raise ValueError("Could not find JSON in response. First 1200 chars:\n" + text[:1200])
    return json.loads(m.group(0))


@tool
def predict_repurposedrugs(
    query: Union[str, List[str]],
    name: str | None = None,
    timeout_pubchem_s: int = 30,
    timeout_runqc_s: int = 120,
) -> dict | str:
    """
    Predict new indicator for given drugs.

    Parameters
    ----------
    query : str | list[str]
        A SMILES string, a compound name resolvable by PubChem, a list of SMILES
        or names, or a path to a CSV/TSV file containing a column with "smiles"
        in its name.
    name : str | None, optional
        Optional label for the drug. If not provided, the compound name (when
        `query` is a name) or the string ``"custom"`` (when `query` is SMILES)
        is used.
    timeout_pubchem_s : int, optional
        Timeout in seconds for the PubChem name-to-SMILES request.
    timeout_runqc_s : int, optional
        Timeout in seconds for the RepurposeDrugs RunQC request.

    Returns
    -------
    dict | str
        If the input is a single SMILES string, returns a dictionary with the
        SMILES, ChEMBL ID, and prediction list. Otherwise writes a CSV under the
        active user's task folder and returns the file path.
    """

    def _extract_inputs(
        query_input: Union[str, List[str]],
    ) -> tuple[list[dict], int]:
        items: list[dict] = []
        if isinstance(query_input, list):
            for value in query_input:
                if isinstance(value, str) and value.strip():
                    items.append(
                        {
                            "raw": value.strip(),
                            "label": None,
                            "force_smiles": False,
                        }
                    )
        elif isinstance(query_input, str):
            if os.path.isfile(query_input):
                ext = os.path.splitext(query_input)[-1].lower()
                if ext not in [".csv", ".tsv"]:
                    raise ValueError("Only CSV or TSV files are supported")
                sep = "\t" if ext == ".tsv" else ","
                df = pd.read_csv(query_input, sep=sep)
                df.columns = [col.lower() for col in df.columns]
                smiles_columns = [col for col in df.columns if "smiles" in col]
                if not smiles_columns:
                    raise ValueError("No 'smiles' column found in the file")
                smiles_column = smiles_columns[0]
                name_column = None
                for col in df.columns:
                    if col in {"name", "drug", "drug_name", "compound", "compound_name"}:
                        name_column = col
                        break
                for idx, value in df[smiles_column].items():
                    if pd.isna(value):
                        continue
                    label = None
                    if name_column is not None:
                        name_value = df.at[idx, name_column]
                        if pd.notna(name_value):
                            label = str(name_value).strip()
                    items.append(
                        {
                            "raw": str(value).strip(),
                            "label": label,
                            "force_smiles": True,
                        }
                    )
            else:
                parts = [part.strip() for part in query_input.split(",") if part.strip()]
                for value in parts:
                    items.append(
                        {
                            "raw": value,
                            "label": None,
                            "force_smiles": False,
                        }
                    )
        else:
            raise ValueError(
                "query must be a SMILES string, list of SMILES/names, or a CSV/TSV path"
            )
        if not items:
            raise ValueError("query is empty")
        return items, len(items)

    inputs, input_count = _extract_inputs(query)
    if name and input_count > 1:
        raise ValueError("name is only supported for a single query")

    results = []
    chembl_cache = {}

    def _lookup_chembl_id(smiles_value: str):
        cached = chembl_cache.get(smiles_value)
        if cached is not None or smiles_value in chembl_cache:
            return cached
        try:
            chembl_id = chembl_get_id(smiles_value, "smiles")
        except Exception:
            chembl_id = None
        if isinstance(chembl_id, float) and pd.isna(chembl_id):
            chembl_id = None
        chembl_cache[smiles_value] = chembl_id
        return chembl_id

    for idx, item in enumerate(tqdm(inputs, desc="Repurposedrugs predictions"), start=1):
        raw = item["raw"]
        if item["force_smiles"]:
            smiles = raw
            drug_label = item["label"] or name or f"custom_{idx}"
        else:
            if _looks_like_smiles(raw):
                smiles = raw
                drug_label = name if name else ("custom" if input_count == 1 else f"custom_{idx}")
            else:
                drug_label = name if name else raw
                smiles = _pubchem_name_to_smiles(raw, timeout_s=timeout_pubchem_s)

        chembl_id = _lookup_chembl_id(smiles)
        payload = _call_runqc(drug_label, smiles, timeout_s=timeout_runqc_s)
        diseases = payload.get("diseases", [])
        values = payload.get("values", [])
        if not diseases and not values:
            results.append(
                pd.DataFrame(
                    {
                        "smiles": [smiles],
                        "chembl_id": [chembl_id],
                        "disease": [None],
                        "prediction_score": [None],
                    }
                )
            )
            continue
        if len(diseases) != len(values):
            raise ValueError(
                f"Length mismatch for {drug_label!r}: diseases={len(diseases)} values={len(values)}"
            )
        results.append(
            pd.DataFrame(
                {
                    "smiles": [smiles] * len(diseases),
                    "chembl_id": [chembl_id] * len(diseases),
                    "disease": diseases,
                    "prediction_score": values,
                }
            )
        )

    df = pd.concat(results, ignore_index=True)
    df = df.sort_values("prediction_score", ascending=False, ignore_index=True)

    is_single_smiles = (
        input_count == 1
        and not (isinstance(query, str) and os.path.isfile(query))
        and (inputs[0]["force_smiles"] or _looks_like_smiles(inputs[0]["raw"]))
    )

    if is_single_smiles:
        smiles_value = df.at[0, "smiles"] if not df.empty else inputs[0]["raw"]
        chembl_value = df.at[0, "chembl_id"] if not df.empty else None
        predictions = []
        if not df.empty:
            for _, row in df.iterrows():
                if pd.isna(row["disease"]) and pd.isna(row["prediction_score"]):
                    continue
                predictions.append(
                    {
                        "disease": row["disease"],
                        "prediction_score": row["prediction_score"],
                    }
                )
        return {
            "smiles": smiles_value,
            "chembl_id": chembl_value,
            "predictions": predictions,
        }

    output_path = task_file_path("drugs_new_indications.csv")
    df.to_csv(output_path, index=False)
    return str(output_path)
