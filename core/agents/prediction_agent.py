import json
import os
import re
import shlex
import subprocess
from typing import List, Union

import pandas as pd
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from rapidfuzz import process

from app.config import logger
from core.prompts.prompts import PREDICTION_SYSTEM_PROMPT_ver3
from backend.utils.fuzzy_path import prompt_with_file_path
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
        if "smiles" not in df.columns:
            return "Error: No 'smiles' column found in the file"
        smiles_list = df["smiles"].dropna().astype(str).tolist()

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
    df['tmp'] = df[column].apply(format_clf_label)
    df[column] = df['tmp']
    df = df.drop(['tmp'],axis=1)
    return df


def _prepare_output_file(filename: str):
    """Return the path for a classifier artifact inside the active task directory."""
    path = task_file_path(filename)
    if path.exists():
        path.unlink()
    return path

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
    cmd = f'java -jar models/CPSign/cpsign-2.0.0-fatjar.jar predict \
    --model models/CYP3A4_clf_trained.jar \
    --predict-file CSV {data_path} \
    --confidences 0.79 \
    --output-format CSV \
    --output {output_str}'
    _ = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    # Modify output columns 
    df = pd.read_csv(output_path)
    df.columns = ['smiles','p_value_0','p_value_1','CYP3A4_inhibition']
    df = format_clf_df(df, 'CYP3A4_inhibition')
    df.to_csv(output_path, index=False)

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
    cmd = f'java -jar models/CPSign/cpsign-2.0.0-fatjar.jar predict \
    --model models/hERG_clf_trained.jar \
    --predict-file CSV {data_path} \
    --confidences 0.78 \
    --output-format CSV \
    --output {output_str}'
    _ = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    # Modify output columns 
    df = pd.read_csv(output_path)
    df.columns = ['smiles','p_value_0','p_value_1','hERG_inhibition']
    df = format_clf_df(df, 'hERG_inhibition')
    df.to_csv(output_path, index=False)
    
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
    cmd = f'java -jar models/CPSign/cpsign-2.0.0-fatjar.jar predict \
    --model models/AMES_clf_trained.jar \
    --predict-file CSV {data_path} \
    --confidences 0.81 \
    --output-format CSV \
    --output {output_str}'
    _ = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    # Modify output columns 
    df = pd.read_csv(output_path)
    df.columns = ['smiles','p_value_0','p_value_1','AMES_mutagenic']
    df = format_clf_df(df, 'AMES_mutagenic')
    df.to_csv(output_path, index=False)
    
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
    cmd = f'java -jar models/CPSign/cpsign-2.0.0-fatjar.jar predict \
    --model models/PGP_clf_trained.jar \
    --predict-file CSV {data_path} \
    --confidences 0.83 \
    --output-format CSV \
    --output {output_str}'
    _ = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    # Modify output columns
    df = pd.read_csv(output_path)
    df.columns = ['smiles','p_value_0','p_value_1','PGP_inhibition']
    df = format_clf_df(df, 'PGP_inhibition')
    df.to_csv(output_path, index=False)

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
    cmd = f'java -jar models/CPSign/cpsign-2.0.0-fatjar.jar predict \
    --model models/Solubility_rgs_trained.jar \
    --predict-file CSV {data_path} \
    --confidences 0.71 \
    --output-format CSV \
    --output {output_str}'
    _ = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    # Modify output columns 
    df = pd.read_csv(output_path)
    df.columns = ['smiles','logS','logS_lower_bound','logS_upper_bound', 'Capped_logS_lower_bound', 'Capped_logS_upper_bound']
    df.to_csv(output_path, index=False)

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
    df['mol'] = df['smiles'].apply(lambda smi: Chem.MolFromSmiles(smi))
    df['logP'] = df['mol'].apply(lambda mol: Crippen.MolLogP(mol))
    df = df.drop(['mol'], axis=1)

    # Modify output columns
    df.to_csv(output_path, index=False)

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
    cmd = f'java -jar models/CPSign/cpsign-2.0.0-fatjar.jar predict \
    --model models/PAMPA_clf_trained.jar \
    --predict-file CSV {data_path} \
    --confidences 0.75 \
    --output-format CSV \
    --output {output_str}'
    _ = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    # Modify output columns 
    df = pd.read_csv(output_path)
    df.columns = ['smiles','p_value_0','p_value_1','PAMPA_permeability']
    df = format_clf_df(df, 'PAMPA_permeability')
    df.to_csv(output_path, index=False)

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
    cmd = f'java -jar models/CPSign/cpsign-2.0.0-fatjar.jar predict \
    --model models/BBB_clf_trained.jar \
    --predict-file CSV {data_path} \
    --confidences 0.80 \
    --output-format CSV \
    --output {output_str}'
    _ = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    # Modify output columns 
    df = pd.read_csv(output_path)
    df.columns = ['smiles','p_value_0','p_value_1','BBB_penetration']
    df = format_clf_df(df, 'BBB_penetration')
    df.to_csv(output_path, index=False)

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
    cmd = f'java -jar models/CPSign/cpsign-2.0.0-fatjar.jar predict \
    --model models/CYP2C19_clf_trained.jar \
    --predict-file CSV {data_path} \
    --confidences 0.80 \
    --output-format CSV \
    --output {output_str}'
    _ = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    # Modify output columns
    df = pd.read_csv(output_path)
    df.columns = ['smiles','p_value_0','p_value_1','CYP2C19_inhibition']
    df = format_clf_df(df, 'CYP2C19_inhibition')
    df.to_csv(output_path, index=False)

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
    cmd = f'java -jar models/CPSign/cpsign-2.0.0-fatjar.jar predict \
    --model models/CYP2D6_clf_trained.jar \
    --predict-file CSV {data_path} \
    --confidences 0.80 \
    --output-format CSV \
    --output {output_str}'
    _ = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    # Modify output columns 
    df = pd.read_csv(output_path)
    df.columns = ['smiles','p_value_0','p_value_1','CYP2D6_inhibition']
    df = format_clf_df(df, 'CYP2D6_inhibition')
    df.to_csv(output_path, index=False)

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
    cmd = f'java -jar models/CPSign/cpsign-2.0.0-fatjar.jar predict \
    --model models/CYP1A2_clf_trained.jar \
    --predict-file CSV {data_path} \
    --confidences 0.84 \
    --output-format CSV \
    --output {output_str}'
    _ = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    # Modify output columns 
    df = pd.read_csv(output_path)
    df.columns = ['smiles','p_value_0','p_value_1','CYP1A2_inhibition']
    df = format_clf_df(df, 'CYP1A2_inhibition')
    df.to_csv(output_path, index=False)

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
    cmd = f'java -jar models/CPSign/cpsign-2.0.0-fatjar.jar predict \
    --model models/CYP2C9_clf_trained.jar \
    --predict-file CSV {data_path} \
    --confidences 0.80 \
    --output-format CSV \
    --output {output_str}'
    _ = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    # Modify output columns 
    df = pd.read_csv(output_path)
    df.columns = ['smiles','p_value_0','p_value_1','CYP2C9_inhibition']
    df = format_clf_df(df, 'CYP2C9_inhibition')
    df.to_csv(output_path, index=False)

    return str(output_path)

def build_prediction_agent(llm):
    prediction_agent = create_react_agent(
        model = llm, 
        tools = [CYP3A4_classifier, 
                CYP2C19_classifier,
                CYP2D6_classifier,
                CYP1A2_classifier,
                CYP2C9_classifier,
                hERG_classifier, 
                AMES_classifier, 
                PGP_classifier, 
                Solubility_regressor, 
                Lipophilicity_regressor,
                PAMPA_classifier, 
                BBB_classifier,
                prompt_with_file_path], 
        name='prediction_agent',
        prompt=PREDICTION_SYSTEM_PROMPT_ver3,
        version='v2'
    )

    return prediction_agent
