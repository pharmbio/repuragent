'''ADMET prediction tools, backed by pre-trained CPSign conformal models.

Twelve endpoints, all driven from `CPSIGN_MODELS` below: one Java invocation per
call, one CSV per endpoint written into the conversation's output scope.

Two things about the numbers these return:

* A classifier is **conformal**, so at the configured confidence it may return
  both labels. CPSign writes that as `{0}` / `{1}` / `{0, 1}`, mapped here to
  `0` / `1` / `0.5`. **0.5 means "both labels are plausible at this confidence",
  not "50 % probability"** — it is an abstention, and treating it as a middling
  score is a misreading.
* `Solubility_regressor` returns a point estimate plus a conformal prediction
  interval; the bounds are part of the answer, not decoration.

`predict_repurposedrugs` is different in kind: it queries an external service for
new-indication predictions rather than running a local model.
'''

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union
from urllib.parse import quote

import pandas as pd
import requests
from langchain_core.tools import tool

from app.config import (
    CPSIGN_CONFIDENCE,
    CPSIGN_JAR,
    CPSIGN_TIMEOUT_SECONDS,
    MODELS_ROOT,
    logger,
)
from backend.utils.cancellation import ExecutionCancelled, cancel_event
from backend.utils.cancellation import cancellable_tqdm as tqdm  # every tqdm loop is a Stop checkpoint
from backend.utils.chemical_annotator.chembl_utils import chembl_get_id
from backend.utils.output_paths import task_file_path


class PredictionError(RuntimeError):
    '''A prediction could not be produced, with a reason worth showing the agent.'''

# --- Input handling -----------------------------------------------------------


def smiles_csv(smiles_input: Union[str, List[str]]) -> str:
    '''Normalise any accepted SMILES input into a one-column CSV and return its path.

    Accepts a single SMILES, a comma-separated string, a list, or a CSV/TSV path
    with a column whose name contains "smiles".

    The file is written into the *conversation's* output scope. It used to go to a
    single shared `DATA_ROOT/modelling_data.csv`, so two users predicting at the
    same time silently overwrote each other's model input.

    Parameters:
    ---------
    smiles_input (Union[str, List[str]]): a single SMILES, a comma-separated string, a list, or a CSV/TSV path with a column whose name contains 'smiles'.

    Returns:
    ----------
    path (str): the one-column CSV written into this conversation's output scope.
    '''

    smiles_list: List[str] = []

    if isinstance(smiles_input, str) and os.path.isfile(smiles_input):
        ext = os.path.splitext(smiles_input)[-1].lower()
        if ext not in (".csv", ".tsv"):
            raise PredictionError("Only CSV or TSV files are supported for SMILES input.")
        frame = pd.read_csv(smiles_input, sep="\t" if ext == ".tsv" else ",")
        frame.columns = [str(col).lower() for col in frame.columns]
        matches = [col for col in frame.columns if "smiles" in col]
        if not matches:
            raise PredictionError(
                f"No column containing 'smiles' found in {smiles_input}. "
                f"Columns present: {', '.join(frame.columns)}."
            )
        smiles_list = frame[matches[0]].dropna().astype(str).tolist()
    elif isinstance(smiles_input, str):
        smiles_list = [item.strip() for item in smiles_input.split(",") if item.strip()]
    elif isinstance(smiles_input, list):
        smiles_list = [item.strip() for item in smiles_input if isinstance(item, str) and item.strip()]
    else:
        raise PredictionError(
            "Input must be a SMILES string, a comma-separated string, a list of "
            "SMILES, or a path to a CSV/TSV file with a 'smiles' column."
        )

    if not smiles_list:
        raise PredictionError("No valid SMILES strings were provided.")

    path = task_file_path("modelling_input.csv")
    pd.DataFrame(smiles_list, columns=["smiles"]).to_csv(path, index=False)
    return str(path)


def format_clf_label(label):
    '''CPSign's conformal label set -> a single value (0.5 = both labels plausible).

    Parameters:
    ---------
    label (str): one conformal label set as CPSign writes it, `{0}`, `{1}` or `{0, 1}`.

    Returns:
    ----------
    value (int or float): 0, 1, or 0.5 meaning both labels are plausible at the configured confidence.
    '''

    if label == "{0}":
        return 0
    if label == "{1}":
        return 1
    if label == "{0, 1}":
        return 0.5
    return label


def format_clf_df(df: pd.DataFrame, column: str) -> pd.DataFrame:
    df[column] = df[column].apply(format_clf_label)
    return df


def _prepare_output_file(filename: str) -> Path:
    '''Fresh path for one endpoint's results inside the active conversation scope.

    Parameters:
    ---------
    filename (str): the endpoint's results file name, from its `CPSIGN_MODELS` entry.

    Returns:
    ----------
    path (Path): the scoped output path, with any previous run's file removed.
    '''

    path = task_file_path(filename)
    if path.exists():
        path.unlink()
    return path


def _single_row_to_dict(df: pd.DataFrame) -> dict:
    '''A one-compound result reads better inline than as a path to a one-row CSV.

    Parameters:
    ---------
    df (Pandas DataFrame): a results table holding exactly one compound.

    Returns:
    ----------
    row (dict): the single row with NaN as None and numpy scalars unwrapped.
    '''

    if df.empty:
        return {}
    normalized = {}
    for key, value in df.iloc[0].to_dict().items():
        if pd.isna(value):
            normalized[key] = None
        elif hasattr(value, "item"):
            normalized[key] = value.item()
        else:
            normalized[key] = value
    return normalized


# --- Running CPSign ----------------------------------------------------------


def _run_cpsign(model_file: str, data_path: str, output_path: Path, confidence: float) -> None:
    '''Invoke CPSign for one model, or raise `PredictionError` explaining why not.

    Three things this fixes over the previous inline `subprocess.run(..., shell=True)`:

    * The jar paths are absolute (from config), so a prediction no longer depends
      on the process happening to have been started from the repository root.
    * The exit status is checked. It used to be discarded into `_`, so a failed
      Java run surfaced one line later as `FileNotFoundError: hERG_results.csv` —
      with the actual CPSign error message thrown away.
    * The child is polled rather than waited on, so Stop terminates the JVM
      instead of leaving it running after the user has moved on.

    Parameters:
    ---------
    model_file (str): the model's file name, resolved under `MODELS_ROOT`.
    data_path (str): the one-column SMILES CSV to predict on.
    output_path (Path): where CPSign should write its results.
    confidence (float): the conformal confidence level for this endpoint.
    '''

    jar = Path(CPSIGN_JAR)
    model = MODELS_ROOT / model_file
    if not jar.exists():
        raise PredictionError(
            f"The CPSign jar is missing at {jar}. ADMET prediction needs it plus a "
            "Java runtime (`java -version`)."
        )
    if not model.exists():
        raise PredictionError(f"The trained model {model} is missing.")

    command = [
        "java",
        "-jar",
        str(jar),
        "predict",
        "--model",
        str(model),
        "--predict-file",
        "CSV",
        str(data_path),
        "--confidences",
        str(confidence),
        "--output-format",
        "CSV",
        "--output",
        str(output_path),
    ]

    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except FileNotFoundError as exc:
        raise PredictionError(
            "Java was not found on PATH; CPSign models cannot run without a JRE."
        ) from exc

    event = cancel_event()
    deadline = time.monotonic() + CPSIGN_TIMEOUT_SECONDS if CPSIGN_TIMEOUT_SECONDS else None
    while True:
        try:
            stdout, stderr = process.communicate(timeout=0.5)
            break
        except subprocess.TimeoutExpired:
            if event.is_set():
                process.kill()
                process.communicate()
                raise ExecutionCancelled("CPSign prediction stopped at your request.")
            if deadline is not None and time.monotonic() > deadline:
                process.kill()
                process.communicate()
                raise PredictionError(
                    f"CPSign did not finish within {CPSIGN_TIMEOUT_SECONDS}s for {model_file}. "
                    "Predict a smaller batch of compounds."
                )

    if process.returncode != 0 or not output_path.exists():
        combined = f"{stdout or ''}\n{stderr or ''}"
        if "No molecules parsed" in combined:
            # CPSign's own message here is just "Try cpsign --help", which sends
            # the reader after a flag problem that does not exist: the real cause
            # is that none of the structures could be featurized.
            raise PredictionError(
                f"{model_file} could not featurize any of the supplied structures. "
                "These conformal models are built for drug-like molecules; very "
                "small fragments have too few signature descriptors to predict."
            )
        detail = [line for line in combined.strip().splitlines() if line.strip()]
        message = detail[-1] if detail else f"exit code {process.returncode}"
        raise PredictionError(f"CPSign failed for {model_file}: {message}")


# --- The endpoint table -------------------------------------------------------


@dataclass(frozen=True)
class CPSignModel:
    model_file: str
    output_file: str
    columns: List[str]
    label_column: Optional[str] = None  # set for conformal classifiers
    confidence: float = CPSIGN_CONFIDENCE


_CLASSIFIER_COLUMNS = ["smiles", "p_value_0", "p_value_1"]


def _classifier(name: str, label_column: str) -> CPSignModel:
    return CPSignModel(
        model_file=f"{name}_clf_trained.jar",
        output_file=f"{name}_results.csv",
        columns=[*_CLASSIFIER_COLUMNS, label_column],
        label_column=label_column,
    )


CPSIGN_MODELS: Dict[str, CPSignModel] = {
    "CYP3A4": _classifier("CYP3A4", "CYP3A4_inhibition"),
    "CYP2C19": _classifier("CYP2C19", "CYP2C19_inhibition"),
    "CYP2D6": _classifier("CYP2D6", "CYP2D6_inhibition"),
    "CYP1A2": _classifier("CYP1A2", "CYP1A2_inhibition"),
    "CYP2C9": _classifier("CYP2C9", "CYP2C9_inhibition"),
    "hERG": _classifier("hERG", "hERG_inhibition"),
    "AMES": _classifier("AMES", "AMES_mutagenic"),
    "PGP": _classifier("PGP", "PGP_inhibition"),
    "PAMPA": _classifier("PAMPA", "PAMPA_permeability"),
    "BBB": _classifier("BBB", "BBB_penetration"),
    "Solubility": CPSignModel(
        model_file="Solubility_rgs_trained.jar",
        output_file="Solubility_results.csv",
        columns=[
            "smiles",
            "logS",
            "logS_lower_bound",
            "logS_upper_bound",
            "Capped_logS_lower_bound",
            "Capped_logS_upper_bound",
        ],
        confidence=0.71,
    ),
}


def _predict(endpoint: str, smiles_input: Union[str, List[str]]):
    '''Run one CPSign endpoint end to end. Shared by every tool below.

    Parameters:
    ---------
    endpoint (str): key into `CPSIGN_MODELS` naming which model to run.
    smiles_input (Union[str, List[str]]): anything `smiles_csv` accepts.

    Returns:
    ----------
    result (dict or str): a dict for a single compound, otherwise the results path — with a warning naming any structure CPSign could not featurize.
    '''

    spec = CPSIGN_MODELS[endpoint]
    try:
        data_path = smiles_csv(smiles_input)
        requested = len(pd.read_csv(data_path).index)
        output_path = _prepare_output_file(spec.output_file)
        _run_cpsign(spec.model_file, data_path, output_path, spec.confidence)
        df = pd.read_csv(output_path)
        if len(df.columns) != len(spec.columns):
            raise PredictionError(
                f"CPSign returned {len(df.columns)} columns for {endpoint}, expected "
                f"{len(spec.columns)}: {list(df.columns)}"
            )
        df.columns = spec.columns
        if spec.label_column:
            df = format_clf_df(df, spec.label_column)
        df.to_csv(output_path, index=False)
    except ExecutionCancelled:
        raise
    except PredictionError as exc:
        logger.warning("%s prediction failed: %s", endpoint, exc)
        return f"Error: {exc}"
    except Exception as exc:  # noqa: BLE001 - reported to the agent, not swallowed
        logger.exception("%s prediction failed", endpoint)
        return f"Error: {type(exc).__name__}: {exc}"

    # CPSign silently omits molecules whose signature descriptors it cannot
    # compute — very small fragments especially. The result is then a shorter
    # table than the input, which a ranking step would happily average over
    # without noticing, so the shortfall is reported rather than left implicit.
    returned = len(df.index)
    missing = _missing_smiles(data_path, df)
    if returned == 0:
        return (
            f"Error: {endpoint} produced no predictions. CPSign could not featurize "
            f"any of the {requested} input structure(s)"
            + (f" (e.g. {', '.join(missing[:5])})" if missing else "")
            + ". These models need drug-like molecules; very small fragments fail."
        )

    if returned == 1 and requested == 1:
        return _single_row_to_dict(df)

    if missing:
        listed = ", ".join(missing[:10]) + ("…" if len(missing) > 10 else "")
        return (
            f"{output_path}\n\n[warning] {returned} of {requested} structures were "
            f"predicted. CPSign could not featurize {len(missing)}: {listed}. "
            "They are absent from the CSV, so join on `smiles` rather than by row order."
        )
    return str(output_path)


def _missing_smiles(data_path: str, results: pd.DataFrame) -> List[str]:
    '''Input structures absent from a results table, in input order.

    Parameters:
    ---------
    data_path (str): the input CSV that was predicted on.
    results (Pandas DataFrame): the table CPSign produced.

    Returns:
    ----------
    missing (list): input structures absent from the results, in input order.
    '''

    try:
        wanted = pd.read_csv(data_path)["smiles"].astype(str).tolist()
    except Exception:  # noqa: BLE001 - diagnostics must never break a prediction
        return []
    if "smiles" not in results.columns:
        return []
    produced = set(results["smiles"].astype(str))
    return [item for item in wanted if item not in produced]


_SMILES_ARG_DOC = """
    Parameters:
    ---------
    smiles_input (str or list): A SMILES string, a comma-separated string of SMILES, a list of SMILES, or a path to a CSV/TSV file with a 'smiles' column.

    Returns:
    ----------
    results (dict or str): A dict for a single compound, otherwise the path to the results CSV in this conversation's output folder.
"""


@tool
def CYP3A4_classifier(smiles_input: Union[str, List[str]]):
    '''Predict CYP3A4 inhibition (conformal classifier; 0.5 = both labels plausible).
    {args}
    '''

    return _predict("CYP3A4", smiles_input)


@tool
def CYP2C19_classifier(smiles_input: Union[str, List[str]]):
    '''Predict CYP2C19 inhibition (conformal classifier; 0.5 = both labels plausible).
    {args}
    '''

    return _predict("CYP2C19", smiles_input)


@tool
def CYP2D6_classifier(smiles_input: Union[str, List[str]]):
    '''Predict CYP2D6 inhibition (conformal classifier; 0.5 = both labels plausible).
    {args}
    '''

    return _predict("CYP2D6", smiles_input)


@tool
def CYP1A2_classifier(smiles_input: Union[str, List[str]]):
    '''Predict CYP1A2 inhibition (conformal classifier; 0.5 = both labels plausible).
    {args}
    '''

    return _predict("CYP1A2", smiles_input)


@tool
def CYP2C9_classifier(smiles_input: Union[str, List[str]]):
    '''Predict CYP2C9 inhibition (conformal classifier; 0.5 = both labels plausible).
    {args}
    '''

    return _predict("CYP2C9", smiles_input)


@tool
def hERG_classifier(smiles_input: Union[str, List[str]]):
    '''Predict hERG inhibition — cardiotoxicity risk (conformal classifier).
    {args}
    '''

    return _predict("hERG", smiles_input)


@tool
def AMES_classifier(smiles_input: Union[str, List[str]]):
    '''Predict Ames mutagenicity (conformal classifier; 0.5 = both labels plausible).
    {args}
    '''

    return _predict("AMES", smiles_input)


@tool
def PGP_classifier(smiles_input: Union[str, List[str]]):
    '''Predict P-glycoprotein inhibition (conformal classifier).
    {args}
    '''

    return _predict("PGP", smiles_input)


@tool
def PAMPA_classifier(smiles_input: Union[str, List[str]]):
    '''Predict PAMPA passive permeability (conformal classifier).
    {args}
    '''

    return _predict("PAMPA", smiles_input)


@tool
def BBB_classifier(smiles_input: Union[str, List[str]]):
    '''Predict blood-brain-barrier penetration (conformal classifier).
    {args}
    '''

    return _predict("BBB", smiles_input)


@tool
def Solubility_regressor(smiles_input: Union[str, List[str]]):
    '''Predict aqueous solubility (logS) with a conformal prediction interval.

    Returns logS plus lower/upper bounds; the interval width is part of the
    answer, so do not report the point estimate alone.
    {args}
    '''

    return _predict("Solubility", smiles_input)


@tool
def Lipophilicity_regressor(smiles_input: Union[str, List[str]]):
    '''Compute lipophilicity as RDKit Crippen logP.

    Deterministic cheminformatics rather than a trained model, so there is no
    prediction interval.
    {args}
    '''

    from rdkit import Chem
    from rdkit.Chem import Crippen

    try:
        data_path = smiles_csv(smiles_input)
        output_path = _prepare_output_file("Lipophilicity_results.csv")
        df = pd.read_csv(data_path)
        molecules = df["smiles"].apply(Chem.MolFromSmiles)
        unparsed = int(molecules.isna().sum())
        df["logP"] = [None if mol is None else Crippen.MolLogP(mol) for mol in molecules]
        df.to_csv(output_path, index=False)
        if unparsed:
            # Silently emitting nulls for unparseable structures would let the
            # caller average over gaps it never noticed.
            logger.warning("logP: %s of %s SMILES could not be parsed", unparsed, len(df))
    except PredictionError as exc:
        return f"Error: {exc}"
    except Exception as exc:  # noqa: BLE001
        logger.exception("Lipophilicity calculation failed")
        return f"Error: {type(exc).__name__}: {exc}"

    if len(df.index) == 1:
        return _single_row_to_dict(df)
    return str(output_path)


# The `{args}` placeholder above keeps the shared argument contract in one place;
# expand it into the real docstrings the model sees.
for _tool in (
    CYP3A4_classifier,
    CYP2C19_classifier,
    CYP2D6_classifier,
    CYP1A2_classifier,
    CYP2C9_classifier,
    hERG_classifier,
    AMES_classifier,
    PGP_classifier,
    PAMPA_classifier,
    BBB_classifier,
    Solubility_regressor,
    Lipophilicity_regressor,
):
    _tool.description = _tool.description.replace("{args}", _SMILES_ARG_DOC.strip())
del _tool


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
    if any(ch.isdigit() for ch in s):  # ring closures
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

    '''Predict new indicator for given drugs.

    Parameters:
    ---------
    query (Union[str, List[str]]): str | list[str] A SMILES string, a compound name resolvable by PubChem, a list of SMILES or names, or a path to a CSV/TSV file containing a column with "smiles" in its name.
    name (str): str | None, optional Optional label for the drug. If not provided, the compound name (when `query` is a name) or the string ``"custom"`` (when `query` is SMILES) is used.
    base_url (str, optional): Base URL of the RepurposeDrugs service. Defaults to ``BASE_URL``.
    timeout_pubchem_s (int): int, optional Timeout in seconds for the PubChem name-to-SMILES request.
    timeout_runqc_s (int): int, optional Timeout in seconds for the RepurposeDrugs RunQC request.

    Returns:
    ----------
    dict | str
    If the input is a single SMILES string, returns a dictionary with the
    SMILES, ChEMBL ID, and prediction list. Otherwise writes a CSV under the
    active user's task folder and returns the file path.
    '''

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
