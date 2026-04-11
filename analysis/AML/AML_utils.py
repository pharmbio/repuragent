import os
import json
import time
import random
from collections import defaultdict
from pathlib import Path
import re
import unicodedata
import csv
import requests
import numpy as np
import pandas as pd
from typing import List, Optional, Dict
from pydantic import BaseModel, Field
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from scipy.stats import kruskal, mannwhitneyu
from statsmodels.stats.multitest import multipletests

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')


# APIs
OPENTARGETS_GQL = "https://api.platform.opentargets.org/api/v4/graphql"
REACTOME_BASE = "https://reactome.org/ContentService"

# Networking
TIMEOUT = 30
MAX_RETRIES = 6

class DrugInfo(BaseModel):
    """Information about a single drug candidate"""
    drug_name: str = Field(description="Primary drug name")
    category: str = Field(description="Drug category (e.g., Kinase inhibitor, HDAC inhibitor)")
    mechanism: str = Field(description="Mechanism of action")
    test_range_start: float = Field(description="Lowest concentration value")
    test_range_end: float = Field(description="Highest concentration value")
    unit: str = Field(description="Unit of measurement (nM, µM, mM)")
    additional_notes: Optional[str] = Field(default="", description="Additional relevant information")

class DrugList(BaseModel):
    """List of drug candidates for AML repurposing"""
    drugs: List[DrugInfo] = Field(description="List of 20 drug candidates with their information")

SALT_TOKENS = {"hydrochloride", "hydrobromide", "chloride", "bromide", "iodide",
    "sulfate", "sulphate", "phosphate", "nitrate", "carbonate",
    "acetate", "citrate", "tartrate", "maleate", "fumarate", "succinate",
    "oxalate", "lactate", "mesylate", "besylate", "tosylate",
    "formate", "benzoate", "gluconate", "valerate", "stearate",
    "palmitate", "pamoate", "embonate", "hcl",
    "sodium", "potassium", "calcium", "magnesium", "lithium", "zinc",
    "ammonium",
    "hydrate", "monohydrate", "dihydrate", "trihydrate", "tetrahydrate",
    "hemihydrate",
    "freebase", "free base",
}
TRAILING_BRACKETED_RE = re.compile(r"\s*[\(\[\{].*?[\)\]\}]\s*$")
PUNCT_TO_SPACE_RE = re.compile(r"[^0-9a-zA-Z]+")
MULTISPACE_RE = re.compile(r"\s+")

def vanila_repurposing(model, prompt, output_file='vanila_LLM_output/results.txt'):
    """Run the model with the given prompt"""
    results = model.invoke(prompt)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=== AML Drug Repurposing Suggestions ===\n\n")
        f.write(f"Prompt:\n{prompt}\n\n")
        f.write("=" * 50 + "\n\n")
        f.write("Results:\n\n")
        f.write(results.content)
    
    return results

def vanila_output_parse(model, result_files='vanila_LLM_output/results.txt', 
                        output_file_csv='vanila_LLM_output/results_parsed.csv'):
    """Parse the results using structured output"""
    structured_model = model.with_structured_output(DrugList)

    # Load results
    results_content = Path(result_files).read_text(encoding="utf-8", errors="replace")

    parsing_prompt = f"""Extract information about drug candidates for AML repurposing from the text below.

    For each drug mentioned, extract:
    - drug_name: The primary drug name (without parentheses)
    - category: The drug category or class
    - mechanism: Brief mechanism of action
    - test_range_start: Numeric value of the lowest test concentration
    - test_range_end: Numeric value of the highest test concentration
    - unit: The unit of measurement (nM, µM, or mM)
    - additional_notes: Any other relevant information like alternative names, specific targets, etc.

    Extract all 20 drugs mentioned in the text.

    TEXT:
    {results_content}
    """

    parsed_data = structured_model.invoke(parsing_prompt)

    # Write to CSV using the structured data
    with open(output_file_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'drug_name', 'category', 'mechanism', 
            'test_range_start', 'test_range_end', 'unit', 'additional_notes'
        ])
        writer.writeheader()
        
        for drug in parsed_data.drugs:
            writer.writerow({
                'drug_name': drug.drug_name,
                'category': drug.category,
                'mechanism': drug.mechanism,
                'test_range_start': drug.test_range_start,
                'test_range_end': drug.test_range_end,
                'unit': drug.unit,
                'additional_notes': drug.additional_notes
            })

    return output_file_csv

def normalize_and_desalt(name: str) -> str:
    """Normalize drug names for comparison"""
    if name is None:
        return ""

    s = str(name).strip()
    if not s:
        return ""

    s = unicodedata.normalize("NFKC", s).casefold()

    while True:
        new_s = TRAILING_BRACKETED_RE.sub("", s).strip()
        if new_s == s:
            break
        s = new_s

    s = PUNCT_TO_SPACE_RE.sub(" ", s)
    s = MULTISPACE_RE.sub(" ", s).strip()

    tokens = s.split()
    changed = True
    while tokens and changed:
        changed = False
        if tokens[-1] in SALT_TOKENS:
            tokens.pop()
            changed = True
        elif len(tokens) >= 2 and f"{tokens[-2]} {tokens[-1]}" in SALT_TOKENS:
            tokens = tokens[:-2]
            changed = True

    return " ".join(tokens).strip()

def compare_drug_lists(
    results_csv: str,
    known_csv: str,
    results_col: str = None,
    known_col: str = None,
    return_tables: bool = True
) -> Dict[str, object]:
    """Compare drugs in results_csv against known_csv"""
    results_df = pd.read_csv(results_csv)
    known_df = pd.read_csv(known_csv)

    results_df["_name_norm"] = results_df[results_col].map(normalize_and_desalt)
    known_df["_name_norm"] = known_df[known_col].map(normalize_and_desalt)

    known_set = set(x for x in known_df["_name_norm"].dropna() if x)

    results_df["_in_known"] = results_df["_name_norm"].apply(
        lambda x: bool(x) and x in known_set
    )

    nonblank_rows = int((results_df["_name_norm"] != "").sum())
    unique_results = results_df.loc[
        results_df["_name_norm"] != "", "_name_norm"
    ].nunique()
    matched_rows = int(results_df["_in_known"].sum())
    matched_unique = results_df.loc[
        results_df["_in_known"], "_name_norm"
    ].nunique()

    return results_df, {
        "matched_rows": matched_rows,
        "matched_unique": matched_unique,
        "row_match_rate": matched_rows / nonblank_rows if nonblank_rows else 0.0,
        "unique_match_rate": matched_unique / unique_results if unique_results else 0.0,
    }

def cache_path(name: str) -> str:
    return os.path.join(CACHE_DIR, name)

def load_json(path: str):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None

def save_json(path: str, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def request_with_retry(method, url, *, headers=None, json_body=None, params=None):
    """
    Conservative retry with exponential backoff + jitter.
    Retries on 429/5xx and transient network errors.
    For 400, prints server payload (often GraphQL field error) then raises.
    """
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.request(
                method,
                url,
                headers=headers,
                json=json_body,
                params=params,
                timeout=TIMEOUT
            )
            if resp.status_code == 200:
                return resp

            if resp.status_code == 400:
                print(f"[HTTP 400] {url}")
                print(resp.text[:2000])
                resp.raise_for_status()

            if resp.status_code in (429, 500, 502, 503, 504):
                sleep_s = min(2 ** attempt, 30) + random.random()
                time.sleep(sleep_s)
                continue

            print(f"[HTTP {resp.status_code}] {url}")
            print(resp.text[:2000])
            resp.raise_for_status()

        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            last_exc = e
            sleep_s = min(2 ** attempt, 30) + random.random()
            time.sleep(sleep_s)
            continue

    raise RuntimeError(f"Failed after {MAX_RETRIES} retries: {method} {url} (last={last_exc})")

def jaccard(a: set, b: set) -> float:
    u = a | b
    return (len(a & b) / len(u)) if u else float("nan")

def cosine_binary(a: set, b: set) -> float:
    u = sorted(list(a | b))
    if not u:
        return float("nan")
    va = np.array([1 if x in a else 0 for x in u]).reshape(1, -1)
    vb = np.array([1 if x in b else 0 for x in u]).reshape(1, -1)
    return float(cosine_similarity(va, vb)[0, 0])

def opentargets_drug_name_from_chembl(chembl_id: str) -> str | None:
    """
    Fetch drug name from Open Targets by ChEMBL ID. Cached.
    """
    cache_file = cache_path(f"opentargets_drug_name_{chembl_id}.json")
    cached = load_json(cache_file)
    if cached is not None:
        return cached.get("name")

    query = """
    query DrugName($chemblId: String!) {
      drug(chemblId: $chemblId) {
        id
        name
      }
    }
    """
    resp = request_with_retry(
        "POST",
        OPENTARGETS_GQL,
        headers={"Content-Type": "application/json"},
        json_body={"query": query, "variables": {"chemblId": chembl_id}}
    )
    data = resp.json()
    drug = (data.get("data") or {}).get("drug")
    name = None
    if drug:
        name = drug.get("name")
    save_json(cache_file, {"name": name})
    return name

def normalize_efo_id(efo: str) -> str:
    """
    User provided "EFO:0000222". Open Targets GraphQL typically uses "EFO_0000222".
    """
    efo = (efo or "").strip()
    return efo.replace(":", "_")

def opentargets_disease_target_score_map(efo_id: str, page_size: int = 500) -> dict:
    """
    Build a map {ENSG -> association score} for disease associatedTargets.

    GraphQL:
      disease(efoId){
        associatedTargets(page:{index,size}){
          rows{ score target{ id approvedSymbol } }
        }
      }

    Cached.
    """
    efo_norm = normalize_efo_id(efo_id)
    cache_file = cache_path(f"opentargets_disease_target_scores_{efo_norm}_ps{page_size}.json")
    cached = load_json(cache_file)
    if cached is not None:
        return {k: float(v) for k, v in cached.items()}

    query = """
    query DiseaseAssociatedTargets($efoId: String!, $index: Int!, $size: Int!) {
      disease(efoId: $efoId) {
        id
        name
        associatedTargets(page: { index: $index, size: $size }) {
          count
          rows {
            score
            target {
              id
              approvedSymbol
            }
          }
        }
      }
    }
    """

    score_map: dict[str, float] = {}
    index = 0

    while True:
        resp = request_with_retry(
            "POST",
            OPENTARGETS_GQL,
            headers={"Content-Type": "application/json"},
            json_body={"query": query, "variables": {"efoId": efo_norm, "index": index, "size": page_size}},
        )
        data = resp.json()

        # GraphQL errors can come back with 200
        if isinstance(data, dict) and data.get("errors"):
            print(f"[OpenTargets GraphQL errors] disease={efo_norm}")
            print(json.dumps(data["errors"], indent=2)[:2000])

        disease = ((data.get("data") or {}).get("disease") or None)
        if not disease:
            break

        assoc = disease.get("associatedTargets") or {}
        rows = assoc.get("rows") or []
        if not rows:
            break

        for r in rows:
            score = r.get("score")
            tgt = r.get("target") or {}
            ensg = tgt.get("id")
            if ensg and score is not None:
                # if repeated, keep max
                score_map[ensg] = max(score_map.get(ensg, 0.0), float(score))

        index += 1

        # if fewer than page_size rows, that was the last page
        if len(rows) < page_size:
            break

    save_json(cache_file, score_map)
    return score_map

def load_repuragent_topN_as_drug_df(top_path: str, n: int = 20) -> pd.DataFrame:
    """
    Load Repuragent top-N from master_ranked.csv and return DataFrame with:
      columns: drug_name, chembl_id

    Requires: chembl_id column.
    If drug_name missing or blank, resolves via Open Targets.
    """
    top_df = pd.read_csv(top_path)

    if "chembl_id" not in top_df.columns:
        raise ValueError(f"{top_path} must include a 'chembl_id' column")

    sub = top_df.dropna(subset=["chembl_id"]).head(n).copy()
    sub["chembl_id"] = sub["chembl_id"].astype(str)

    if "drug_name" not in sub.columns:
        sub["drug_name"] = None

    missing = sub["drug_name"].isna() | (sub["drug_name"].astype(str).str.strip() == "")
    if missing.any():
        for idx in sub[missing].index:
            cid = sub.loc[idx, "chembl_id"]
            nm = opentargets_drug_name_from_chembl(cid)
            sub.loc[idx, "drug_name"] = nm if nm else cid

    out = sub[["drug_name", "chembl_id"]].drop_duplicates().reset_index(drop=True)
    return out

def opentargets_targets_for_drug_chembl(chembl_id: str):
    """
    Robust Open Targets targets retrieval.
    Uses mechanismsOfAction, supports both targets[] and target shapes.

    Returns list of dicts:
      { "ensembl_id": "ENSG....", "approvedSymbol": "..." }
    """
    cache_file = cache_path(f"opentargets_targets_{chembl_id}.json")
    cached = load_json(cache_file)
    if cached is not None:
        return cached

    query_a = """
    query DrugMoA($chemblId: String!) {
      drug(chemblId: $chemblId) {
        id
        name
        mechanismsOfAction {
          rows {
            targets {
              id
              approvedSymbol
            }
          }
        }
      }
    }
    """

    query_b = """
    query DrugMoA($chemblId: String!) {
      drug(chemblId: $chemblId) {
        id
        name
        mechanismsOfAction {
          rows {
            target {
              id
              approvedSymbol
            }
          }
        }
      }
    }
    """

    def run_query(q):
        resp = request_with_retry(
            "POST",
            OPENTARGETS_GQL,
            headers={"Content-Type": "application/json"},
            json_body={"query": q, "variables": {"chemblId": chembl_id}}
        )
        data = resp.json()
        if isinstance(data, dict) and data.get("errors"):
            print(f"[OpenTargets GraphQL errors] {chembl_id}")
            print(json.dumps(data["errors"], indent=2)[:2000])
        return data

    def parse_targets(data):
        drug = (data.get("data") or {}).get("drug")
        if not drug:
            return []
        moa = drug.get("mechanismsOfAction") or {}
        rows = moa.get("rows") or []
        out = []
        for r in rows:
            tlist = r.get("targets")
            if tlist is None and r.get("target") is not None:
                tlist = [r["target"]]
            if not tlist:
                continue
            for t in tlist:
                tid = t.get("id")
                sym = t.get("approvedSymbol")
                if tid and sym:
                    out.append({"ensembl_id": tid, "approvedSymbol": sym})

        # dedupe
        seen = set()
        dedup = []
        for x in out:
            key = (x["ensembl_id"], x["approvedSymbol"])
            if key not in seen:
                seen.add(key)
                dedup.append(x)
        return dedup

    for q in (query_a, query_b):
        try:
            data = run_query(q)
            targets = parse_targets(data)
            if targets:
                save_json(cache_file, targets)
                return targets
        except Exception:
            continue

    save_json(cache_file, [])
    return []

def reactome_pathways_for_ensembl(ensembl_id: str):
    """
    Map Ensembl gene ID -> Reactome pathways.
    404 means no pathways (return empty list).
    """
    cache_file = cache_path(f"reactome_pathways_{ensembl_id}.json")
    cached = load_json(cache_file)
    if cached is not None:
        return cached

    url = f"{REACTOME_BASE}/data/mapping/Ensembl/{ensembl_id}/pathways"

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, timeout=TIMEOUT)

            if resp.status_code == 200:
                try:
                    data = resp.json()
                    if not isinstance(data, list):
                        data = []
                except Exception:
                    data = []
                save_json(cache_file, data)
                return data

            if resp.status_code == 404:
                save_json(cache_file, [])
                return []

            if resp.status_code in (429, 500, 502, 503, 504):
                sleep_s = min(2 ** attempt, 30) + random.random()
                time.sleep(sleep_s)
                continue

            print(f"[Reactome HTTP {resp.status_code}] {url}")
            print(resp.text[:2000])
            resp.raise_for_status()

        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
            sleep_s = min(2 ** attempt, 30) + random.random()
            time.sleep(sleep_s)
            continue

    save_json(cache_file, [])
    return []

def pathways_via_targets(
    drugs_df: pd.DataFrame,
    disease_score_map: dict | None = None,
    protein_score_threshold: float | None = None,
):
    """
    For each drug:
      ChEMBL -> OpenTargets targets (ENSG) -> optional disease-score filter -> Reactome pathways

    Returns:
      pathways_by_drug: chembl_id -> set(pathway IDs)
      targets_by_drug:  chembl_id -> set(ENSG) AFTER filtering (if enabled)
      pw_id_to_name:    pathway_id -> pathway name
    """
    pathways_by_drug = {}
    targets_by_drug = {}
    pw_id_to_name = {}

    use_filter = (disease_score_map is not None) and (protein_score_threshold is not None)

    for _, row in drugs_df.iterrows():
        cid = row["chembl_id"]
        name = row.get("drug_name", cid)

        targets = opentargets_targets_for_drug_chembl(cid)
        raw_targets = [t["ensembl_id"] for t in targets if t.get("ensembl_id")]

        if use_filter:
            filtered = []
            for ensg in raw_targets:
                s = float(disease_score_map.get(ensg, 0.0))
                if s >= float(protein_score_threshold):
                    filtered.append(ensg)
            ensembl_targets = set(filtered)
        else:
            ensembl_targets = set(raw_targets)

        targets_by_drug[cid] = ensembl_targets


        pw_set = set()
        for ens in sorted(ensembl_targets):
            pathways = reactome_pathways_for_ensembl(ens)
            for p in pathways:
                pid = p.get("stId") or p.get("stableIdentifier") or p.get("id") or p.get("dbId")
                pname = p.get("displayName") or p.get("name")
                if pid is None:
                    continue
                pid = str(pid)
                pw_set.add(pid)
                if pname and pid not in pw_id_to_name:
                    pw_id_to_name[pid] = pname

        pathways_by_drug[cid] = pw_set

    return pathways_by_drug, targets_by_drug, pw_id_to_name

def rank_shared_pathways_by_frequency(pw_by_drug_A: dict, pw_by_drug_B: dict):
    """
    Rank shared pathways by how often they appear across drugs in each set.
    Score = freq_in_A + freq_in_B
    """
    freqA = defaultdict(int)
    for s in pw_by_drug_A.values():
        for p in s:
            freqA[p] += 1

    freqB = defaultdict(int)
    for s in pw_by_drug_B.values():
        for p in s:
            freqB[p] += 1

    shared = set(freqA.keys()) & set(freqB.keys())
    ranked = sorted(shared, key=lambda p: (freqA[p] + freqB[p], freqA[p], freqB[p]), reverse=True)
    return ranked, freqA, freqB

def sankey_aggregated(P_rep, P_co, title="Pathway overlap (Aggregated)"):
    shared = P_rep & P_co
    rep_only = P_rep - P_co
    co_only = P_co - P_rep

    nodes = ["Repuragent pathways", "Co-scientist pathways", "Shared", "Rep-only", "Co-only"]
    idx = {n: i for i, n in enumerate(nodes)}

    sources = [
        idx["Repuragent pathways"],
        idx["Repuragent pathways"],
        idx["Co-scientist pathways"],
        idx["Co-scientist pathways"],
    ]
    targets = [idx["Shared"], idx["Rep-only"], idx["Shared"], idx["Co-only"]]
    values = [len(shared), len(rep_only), len(shared), len(co_only)]

    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(label=nodes, pad=15, thickness=20),
                link=dict(source=sources, target=targets, value=values),
            )
        ]
    )
    fig.update_layout(title_text=title, font_size=12)
    return fig

def sankey_top_shared_by_frequency(
    rep_pw_by_drug, co_pw_by_drug, pw_name_map, top_n=15,
    title="Top shared pathways (ranked by frequency across drugs)"
):
    ranked, freqA, freqB = rank_shared_pathways_by_frequency(rep_pw_by_drug, co_pw_by_drug)
    if not ranked:
        return None

    top = ranked[:top_n]
    node_labels = ["Repuragent"] + [pw_name_map.get(pid, pid) for pid in top] + ["Co-scientist"]
    rep_idx = 0
    co_idx = len(node_labels) - 1

    sources, targets, values = [], [], []

    for i, pid in enumerate(top, start=1):
        w = min(freqA[pid], freqB[pid])
        sources.append(rep_idx); targets.append(i); values.append(w)
        sources.append(i); targets.append(co_idx); values.append(w)

    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(label=node_labels, pad=15, thickness=18),
                link=dict(source=sources, target=targets, value=values),
            )
        ]
    )
    fig.update_layout(title_text=title, font_size=11)
    return fig

def get_chembl_id(drug_name):
    """
    Query ChEMBL database to get ChEMBL ID for a given drug name.
    
    Parameters:
    drug_name (str): Name of the drug to search for
    
    Returns:
    dict: Dictionary containing ChEMBL ID and related information
    """
    # ChEMBL REST API endpoint for molecule search
    base_url = "https://www.ebi.ac.uk/chembl/api/data/molecule/search"
    
    # Parameters for the search
    params = {
        'q': drug_name,
        'format': 'json'
    }
    
    try:
        # Make the API request
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        
        # Parse the JSON response
        data = response.json()
        
        # Check if any results were found
        if 'molecules' in data and len(data['molecules']) > 0:
            results = []
            for molecule in data['molecules']:
                result = {
                    'chembl_id': molecule.get('molecule_chembl_id'),
                    'pref_name': molecule.get('pref_name'),
                    'molecule_type': molecule.get('molecule_type'),
                    'max_phase': molecule.get('max_phase'),
                    'synonyms': molecule.get('molecule_synonyms', [])[:5]  # First 5 synonyms
                }
                results.append(result)
            return results[0]['chembl_id']
        else:
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"Error making request: {e}")
        return None


import time
from dataclasses import dataclass
from typing import Dict, Optional, Set, Tuple, Any, List

import pandas as pd
import requests
import gseapy as gp
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


@dataclass(frozen=True)
class DrugTargets:
    drug_name: str
    chembl_id: str
    targets: Set[str]


class DrugPathwayEnrichment:
    """
    Performs pathway enrichment analysis for drugs based on their target genes.

    Input df must contain columns:
      - drug_name
      - chembl_id
    """

    CHEMBL_BASE = "https://www.ebi.ac.uk/chembl/api/data"

    def __init__(
        self,
        df: pd.DataFrame,
        pvalue_threshold: float = 0.05,
        background_type: str = "pathway_genes",
        max_phase: int = 4,
        min_pchembl: float = 6.0,
        human_only_activities: bool = True,
        request_timeout: int = 20,
    ):
        """
        background_type options:
          - 'pathway_genes': background=None (Enrichr uses library genes)
          - 'all': HGNC approved protein-coding gene symbols
          - 'druggable': DGIdb-derived druggable genes (best-effort; API can change)
        """
        required = {"drug_name", "chembl_id"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"df is missing required columns: {sorted(missing)}")

        self.df = df.reset_index(drop=True)
        self.pvalue_threshold = float(pvalue_threshold)
        self.background_type = background_type
        self.max_phase = int(max_phase)
        self.min_pchembl = float(min_pchembl)
        self.human_only_activities = bool(human_only_activities)
        self.request_timeout = int(request_timeout)

        self.drug_targets: Dict[str, DrugTargets] = {}  # chembl_id -> DrugTargets
        self.all_targets: Set[str] = set()

        self.session = self._make_session()

    # -------------------------
    # HTTP session with retries
    # -------------------------
    def _make_session(self) -> requests.Session:
        s = requests.Session()
        retry = Retry(
            total=6,
            backoff_factor=0.8,  # exponential-ish
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET",),
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry)
        s.mount("https://", adapter)
        s.mount("http://", adapter)
        s.headers.update({"User-Agent": "DrugPathwayEnrichment/1.0"})
        return s

    def _get_json(self, url: str, params: Optional[dict] = None, headers: Optional[dict] = None) -> Optional[dict]:
        try:
            r = self.session.get(url, params=params, headers=headers, timeout=self.request_timeout)
            if r.status_code != 200:
                print(f"Warning: GET {url} -> {r.status_code}")
                return None
            return r.json()
        except Exception as e:
            print(f"Warning: GET {url} failed: {e}")
            return None

    # -------------------------
    # ChEMBL target fetching
    # -------------------------
    def get_targets_from_chembl(self, chembl_id: str) -> Set[str]:
        """
        Get gene symbols for a compound via:
          1) mechanism endpoint (preferred)
          2) activity endpoint (fallback)
        """
        genes = self._targets_from_mechanisms(chembl_id, max_phase=self.max_phase)
        if not genes:
            genes = self._targets_from_activities(chembl_id)
        return genes

    def _targets_from_mechanisms(self, chembl_id: str, max_phase: int) -> Set[str]:
        url = f"{self.CHEMBL_BASE}/mechanism.json"
        limit = 1000
        offset = 0
        genes: Set[str] = set()

        while True:
            params = {"molecule_chembl_id": chembl_id, "limit": limit, "offset": offset}
            data = self._get_json(url, params=params)
            if not data:
                break

            mechs = data.get("mechanisms", []) or []
            for mech in mechs:
                # Keep your original interpretation:             >=
                if mech.get("max_phase", 0) <= max_phase:
                    target_id = mech.get("target_chembl_id")
                    if target_id:
                        sym = self._get_gene_symbol_from_target(target_id)
                        if sym:
                            genes.add(sym)

            page_meta = data.get("page_meta") or {}
            total_count = page_meta.get("total_count")
            if total_count is None:
                # Can't safely paginate; stop
                break

            offset += limit
            if offset >= total_count:
                break

        return genes

    def _targets_from_activities(self, chembl_id: str) -> Set[str]:
        url = f"{self.CHEMBL_BASE}/activity.json"
        limit = 200
        offset = 0
        genes: Set[str] = set()

        while True:
            params = {
                "molecule_chembl_id": chembl_id,
                "pchembl_value__gte": self.min_pchembl,
                "limit": limit,
                "offset": offset,
            }
            data = self._get_json(url, params=params)
            if not data:
                break

            acts = data.get("activities", []) or []
            for act in acts:
                # Optional: try to keep human-only when the field exists
                if self.human_only_activities:
                    org = (act.get("target_organism") or "").lower()
                    if org and "homo sapiens" not in org:
                        continue

                target_id = act.get("target_chembl_id")
                if target_id:
                    sym = self._get_gene_symbol_from_target(target_id)
                    if sym:
                        genes.add(sym)

            page_meta = data.get("page_meta") or {}
            total_count = page_meta.get("total_count")
            if total_count is None:
                break

            offset += limit
            if offset >= total_count:
                break

        return genes

    def _get_gene_symbol_from_target(self, target_chembl_id: str) -> Optional[str]:
        """
        Returns a single gene symbol if found via target_component_synonyms.
        If not found, returns None (no guessing).
        """
        url = f"{self.CHEMBL_BASE}/target/{target_chembl_id}.json"
        data = self._get_json(url)
        if not data:
            return None

        for comp in data.get("target_components", []) or []:
            for syn in comp.get("target_component_synonyms", []) or []:
                if syn.get("syn_type") == "GENE_SYMBOL":
                    sym = syn.get("component_synonym")
                    if sym:
                        return sym

        # No reliable gene symbol
        return None

    # -------------------------
    # Annotate all drugs
    # -------------------------
    def annotate_all_drugs(self, delay: float = 0.3) -> Dict[str, DrugTargets]:
        print(f"Annotating {len(self.df)} drugs with target genes...")

        for i, row in self.df.iterrows():
            drug_name = str(row["drug_name"])
            chembl_id = str(row["chembl_id"])

            print(f"  {i+1}/{len(self.df)}: {drug_name} ({chembl_id})")
            targets = self.get_targets_from_chembl(chembl_id)

            self.drug_targets[chembl_id] = DrugTargets(
                drug_name=drug_name,
                chembl_id=chembl_id,
                targets=targets,
            )
            self.all_targets.update(targets)

            time.sleep(delay)

        print(f"\nTotal unique targets found: {len(self.all_targets)}")
        print(f"Drugs with ≥1 target: {sum(1 for x in self.drug_targets.values() if x.targets)}")
        return self.drug_targets

    # -------------------------
    # Backgrounds
    # -------------------------
    def get_background_genes(self) -> Optional[Set[str]]:
        if self.background_type == "pathway_genes":
            # Enrichr library genes used as implicit background
            return None
        if self.background_type == "all":
            return self._get_all_human_protein_coding_symbols()
        if self.background_type == "druggable":
            return self._get_dgidb_druggable_symbols()
        raise ValueError(f"Unknown background_type: {self.background_type}")

    def _get_all_human_protein_coding_symbols(self) -> Optional[Set[str]]:
        """
        HGNC approved symbols, protein-coding only.
        """
        print("Fetching HGNC approved protein-coding gene symbols...")
        url = "https://rest.genenames.org/fetch/status/Approved"
        headers = {"Accept": "application/json"}

        data = self._get_json(url, headers=headers)
        if not data:
            print("Warning: HGNC fetch failed; using pathway_genes background (None).")
            return None

        docs = data.get("response", {}).get("docs", []) or []
        genes: Set[str] = set()
        for doc in docs:
            if doc.get("locus_type") == "gene with protein product":
                sym = doc.get("symbol")
                if sym:
                    genes.add(sym)

        if len(genes) < 15000:
            print(f"Warning: HGNC returned unexpectedly few genes ({len(genes)}). Using None.")
            return None

        print(f"Retrieved {len(genes)} HGNC protein-coding symbols.")
        return genes

    def _get_dgidb_druggable_symbols(self) -> Optional[Set[str]]:
        """
        DGIdb API can change; this is best-effort with validation.
        """
        print("Fetching druggable genes from DGIdb (best-effort)...")
        url = "https://dgidb.org/api/v2/druggable_gene_categories.json"  # more common than /all.json in some versions
        data = self._get_json(url)

        if not data:
            print("Warning: DGIdb fetch failed; using pathway_genes background (None).")
            return None

        genes: Set[str] = set()

        # Try a couple plausible response shapes
        if isinstance(data, dict):
            # shape A: {"categories":[{"genes":[{"name":"..."}]}]}
            cats = data.get("categories")
            if isinstance(cats, list):
                for cat in cats:
                    for g in (cat.get("genes") or []):
                        name = g.get("name") if isinstance(g, dict) else None
                        if name:
                            genes.add(name)

            # shape B: {"genes":[{"name":"..."}]}
            if not genes and isinstance(data.get("genes"), list):
                for g in data["genes"]:
                    name = g.get("name") if isinstance(g, dict) else None
                    if name:
                        genes.add(name)

        if len(genes) < 500:
            print(f"Warning: DGIdb response parsed to only {len(genes)} genes; using None.")
            return None

        print(f"Retrieved {len(genes)} druggable genes.")
        return genes

    # -------------------------
    # Enrichment
    # -------------------------
    def perform_enrichment(
        self,
        gene_sets: str = "KEGG_2021_Human",
        organism: str = "Human",
    ) -> pd.DataFrame:
        if not self.all_targets:
            raise ValueError("No targets found. Run annotate_all_drugs() first.")

        gene_list = sorted(self.all_targets)
        background = self.get_background_genes()

        print(f"\nEnrichment using {gene_sets}")
        print(f"Query genes: {len(gene_list)}")
        print(f"Background: {'library genes (None)' if background is None else len(background)}")

        enr = gp.enrichr(
            gene_list=gene_list,
            gene_sets=gene_sets,
            organism=organism,
            background=background,
            cutoff=1.0,
            no_plot=True,
        )

        results = enr.results.copy()
        sig = results[results["Adjusted P-value"] < self.pvalue_threshold].copy()

        print(f"Pathways tested: {len(results)}")
        print(f"Significant (FDR < {self.pvalue_threshold}): {len(sig)}")
        return sig

    def get_significant_pathways(self, gene_sets: str = "KEGG_2021_Human") -> Set[str]:
        sig = self.perform_enrichment(gene_sets=gene_sets)
        return set(sig["Term"].tolist()) if not sig.empty else set()

    def get_enrichment_summary(self, gene_sets: str = "KEGG_2021_Human") -> Dict[str, Any]:
        sig = self.perform_enrichment(gene_sets=gene_sets)
        return {
            "total_drugs": len(self.df),
            "drugs_with_targets": sum(1 for x in self.drug_targets.values() if x.targets),
            "total_unique_targets": len(self.all_targets),
            "significant_pathways": set(sig["Term"].tolist()) if not sig.empty else set(),
            "enrichment_table": sig,
            "drug_target_mapping": self.drug_targets,
        }


def _find_adj_p_col(df: pd.DataFrame) -> str:
    # exact matches first
    for c in ["Adjusted P-value", "Adjusted P-value ", "Adjusted P-value (FDR)", "Adjusted P-value (BH)"]:
        if c in df.columns:
            return c

    # fuzzy contains match
    lowered = {c.lower(): c for c in df.columns}
    for key in lowered:
        if "adjusted" in key and "p" in key:
            return lowered[key]
        if "fdr" in key:
            return lowered[key]
        if "q-value" in key or "q value" in key:
            return lowered[key]

    raise KeyError(f"No adjusted-p/FDR column found. Columns={df.columns.tolist()}")

def pathways_enrichment_from_chembl_id(
    df: pd.DataFrame,
    min_drugs_with_pathway: int = 1,
) -> set:
    """Per-drug pathway enrichment, then aggregate pathways across drugs.

    This avoids a single promiscuous drug dominating the union-of-targets list.

    Returns a set of pathway terms, like the original implementation.
    """

    # Initialize analyzer
    analyzer = DrugPathwayEnrichment(
        df=df,
        pvalue_threshold=0.05,
        background_type="pathway_genes",
    )

    # Step 1: Annotate drugs with target genes
    analyzer.annotate_all_drugs(delay=0.5)

    # Step 2: Enrichment per drug
    background = analyzer.get_background_genes()
    rows = []

    for dt in analyzer.drug_targets.values():
        genes = sorted(dt.targets)
        if len(genes) < 2:
            continue

        enr = gp.enrichr(
            gene_list=genes,
            gene_sets="Reactome_2022",  # or 'GO_Biological_Process_2023', etc.
            organism="Human",
            background=background,
            cutoff=1.0,
            no_plot=True,
        )

        res = enr.results
        if res is None or res.empty:
            continue
        res = res.copy()

        adj_col = _find_adj_p_col(res)
        res[adj_col] = pd.to_numeric(res[adj_col], errors="coerce")
        sig = res[res[adj_col] < analyzer.pvalue_threshold].copy()
        if sig.empty:
            continue

        sig.insert(0, "chembl_id", dt.chembl_id)
        sig.insert(1, "drug_name", dt.drug_name)
        rows.append(sig)

    if not rows:
        return set()

    per_drug_enr = pd.concat(rows, ignore_index=True)

    # How many distinct drugs show each pathway as significant?
    counts = (
        per_drug_enr.groupby("Term")["chembl_id"]
        .nunique()
        .rename("n_drugs")
        .reset_index()
    )

    keep_terms = counts.loc[counts["n_drugs"] >= int(min_drugs_with_pathway), "Term"]
    return set(keep_terms.tolist())



def nonparametric_repuragent_tests(
    df: pd.DataFrame,
    metrics=("recall", "precision", "jaccard"),
    ref_model="Repuragent",
    p_adjust="holm",          # "holm", "bonferroni", "fdr_bh", ...
):
    """
    Non-parametric testing per metric:
      1) Kruskal–Wallis across all models
      2) Pairwise Mann–Whitney U: ref_model vs each other model
      3) Multiple testing correction across those pairwise p-values

    Returns:
      stats: dict keyed by metric with kruskal p, pairwise raw/adjusted p, reject flags
      summary_df: tidy table of pairwise results (one row per metric x comparison)
    """
    
    if "model" not in df.columns:
        raise ValueError("df must include a 'model' column")

    models = df["model"].dropna().unique().tolist()
    if ref_model not in models:
        raise ValueError(f"Reference model '{ref_model}' not found. Available: {models}")

    stats = {}
    rows = []

    for metric in metrics:
        if metric not in df.columns:
            raise ValueError(f"Missing metric column: '{metric}'")

        # Global Kruskal–Wallis
        groups = [df.loc[df["model"] == m, metric].dropna().values for m in models]
        _, kw_p = kruskal(*groups)

        # Pairwise vs reference
        ref_vals = df.loc[df["model"] == ref_model, metric].dropna().values
        comps, pvals = [], []

        for m in models:
            if m == ref_model:
                continue
            vals = df.loc[df["model"] == m, metric].dropna().values
            _, p = mannwhitneyu(ref_vals, vals, alternative="two-sided")
            comps.append(m)
            pvals.append(p)

        pvals = np.asarray(pvals, dtype=float)
        reject, p_adj, _, _ = multipletests(pvals, method=p_adjust)

        stats[metric] = {
            "kruskal_p": float(kw_p),
            "comparisons": comps,
            "pvals_raw": pvals,
            "pvals_adj": p_adj,
            "reject": reject,
        }

        for other, pr, pa, rj in zip(comps, pvals, p_adj, reject):
            rows.append(
                {
                    "metric": metric,
                    "ref": ref_model,
                    "other": other,
                    "kruskal_p": float(kw_p),
                    "p_raw": float(pr),
                    "p_adj": float(pa),
                    "significant": bool(rj),
                    "p_adjust": p_adjust,
                }
            )

    summary_df = pd.DataFrame(rows).sort_values(["metric", "p_adj"])
    return stats, summary_df


def plot_grouped_bars_with_repuragent_significance(
    df: pd.DataFrame,
    stats: dict,
    metrics=("recall", "precision", "jaccard"),
    ref_model="Repuragent",
    err="std",        # "std", "sem", or "ci95"
    figsize=(12, 6),
    title=None,
):
    """
    Grouped bar plot with metrics as groups on x-axis and models as bars within each group.
    Shows mean ± std/sem/95%CI and significance brackets from ref_model to any model where 
    """

    def _significance_label(p_value: float) -> str:
        if p_value <= 0.001:
            return "***"
        if p_value <= 0.01:
            return "**"
        if p_value <= 0.05:
            return "*"
        return "ns"

    # Means and errors
    grp = df.groupby("model")[list(metrics)]
    means = grp.mean()

    if err == "std":
        errs = grp.std()
    elif err == "sem":
        errs = grp.sem()
    elif err == "ci95":
        errs = grp.sem() * 1.96
    else:
        raise ValueError(f"err must be 'std', 'sem', or 'ci95', got {err}")

    models_order = means.index.tolist()
    if ref_model not in models_order:
        raise ValueError(f"Reference model '{ref_model}' not found in data.")

    n_models = len(models_order)
    n_metrics = len(metrics)

    # x-axis represents metrics, bars represent models
    x = np.arange(n_metrics)
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=figsize)

    # Bars: iterate over models, plot across metrics
    for i, model in enumerate(models_order):
        values = [means.loc[model, metric] for metric in metrics]
        errors = [errs.loc[model, metric] for metric in metrics]

        ax.bar(
            x + i * width,
            values,
            width,
            yerr=errors,
            capsize=3,
            label=model,
        )

    # Significance brackets: for each metric, compare ref_model to others
    ref_idx = models_order.index(ref_model)
    max_annotation_top = None
    all_bar_tops = (means + errs).to_numpy(dtype=float)
    global_bar_top = np.nanmax(all_bar_tops)
    group_offset = max(0.035, global_bar_top * 0.045)  # vertical gap between the tallest bar and the brackets
    group_step = max(0.04, global_bar_top * 0.075)  # vertical gap between the brackets
    bracket_height = group_step * 0.35
    label_offset = 0

    for metric_idx, metric in enumerate(metrics):
        info = stats[metric]

        # Max height in this metric group
        metric_values = [
            means.loc[model, metric] + errs.loc[model, metric]
            for model in models_order
        ]
        y_top = np.nanmax(metric_values)
        y = y_top + group_offset

        for other, p_adj in zip(info["comparisons"], info["pvals_adj"]):
            label = _significance_label(p_adj)

            other_idx = models_order.index(other)
            x1 = metric_idx + ref_idx * width
            x2 = metric_idx + other_idx * width

            # Draw bracket
            bracket_top = y + bracket_height
            label_y = bracket_top + label_offset

            ax.plot(
                [x1, x1, x2, x2],
                [y, bracket_top, bracket_top, y],
                color="black",
                linewidth=1,
            )
            ax.text(
                (x1 + x2) / 2,
                label_y,
                label,
                ha="center",
                va="bottom",
                fontsize=17,
            )

            max_annotation_top = (
                label_y if max_annotation_top is None else max(max_annotation_top, label_y)
            )
            y += group_step

    ax.set_xticks(x + (n_models - 1) * width / 2)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Score", fontsize=15)
    ax.set_xlabel("Metric", fontsize=15)
    ax.tick_params(axis='y', labelsize=14)
    ax.tick_params(axis='x', labelsize=14)

    if title is not None:
        ax.set_title(title, fontsize=16)

    if max_annotation_top is not None:
        current_bottom, current_top = ax.get_ylim()
        ax.set_ylim(current_bottom, max(current_top, max_annotation_top + group_step * 0.7))

    # Legend inside figure, top-right
    ax.legend(title="Model", loc="upper right", fontsize=13, title_fontsize=13)

    plt.tight_layout()
    return fig, ax