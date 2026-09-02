'''chembl_utils.py

Utility functions for retrieving and processing data from the KEGG database.

Author: Flavio Ballante

Contact: flavio.ballante@ki.se, flavioballante@gmail.com

Institution: CBCS-SciLifeLab-Karolinska Institutet

Year: 2025
'''

import re
import time

import pandas as pd
import requests

_KEGG_PATHWAY_CACHE = {}
_KEGG_ENTRY_TEXT_CACHE = {}


def _kegg_get_with_retry(url, *, max_attempts=4, base_sleep_seconds=1, timeout=30):
    last_exception = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url, timeout=timeout)
            response.raise_for_status()
            return response
        except (
            requests.exceptions.SSLError,
            requests.exceptions.Timeout,
            requests.exceptions.ConnectionError,
        ) as exc:
            last_exception = exc
            if attempt == max_attempts:
                raise
            time.sleep(base_sleep_seconds * (2 ** (attempt - 1)))
        except requests.RequestException:
            raise
    raise last_exception

def get_pathways_from_ec(ec_number):
    '''Retrieve KEGG pathway information for a given EC (Enzyme Commission) number.

    Parameters:
    ---------
    ec_number (str): The EC number for which to retrieve associated KEGG pathways.

    Returns:
    ----------
    pathways (Pandas DataFrame): A DataFrame with columns ['EC Numbers', 'KEGG_ID', 'Pathway'] containing pathway information. Returns an empty DataFrame if no pathways are found or if an error occurs.
    '''

    # Define column names
    COLUMNS = ['EC Numbers', 'KEGG_ID', 'Pathway']
    # Check if ec_number is empty, None, or NaN
    if pd.isna(ec_number) or not ec_number: 
        #print(f"Warning: EC Number is empty, None, or NaN")
        return pd.DataFrame(columns=COLUMNS)

    if ec_number in _KEGG_PATHWAY_CACHE:
        return _KEGG_PATHWAY_CACHE[ec_number].copy()

    link_url = f"https://rest.kegg.jp/link/pathway/enzyme:{ec_number}"
    try:
        response = _kegg_get_with_retry(link_url)

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
            pathway_text = _KEGG_ENTRY_TEXT_CACHE.get(pathway)
            if pathway_text is None:
                pathway_url = f"https://rest.kegg.jp/get/{pathway}"
                pathway_text = _kegg_get_with_retry(pathway_url).text
                _KEGG_ENTRY_TEXT_CACHE[pathway] = pathway_text

            lines = pathway_text.split('\n')
            pathway_maps = [line.replace('PATHWAY_MAP', '').strip() for line in lines if line.startswith('PATHWAY_MAP') and re.match(r'^PATHWAY_MAP\s+map\d+', line)]

            for pathway_map in pathway_maps:
                results.append({'EC Numbers': ec_number, 'KEGG_ID': pathway, 'Pathway': pathway_map})

        if not results:
            print(f"No pathway information found for EC number {ec_number}")
            return pd.DataFrame(columns=COLUMNS)

        # Create DataFrame
        df = pd.DataFrame(results)
        _KEGG_PATHWAY_CACHE[ec_number] = df.copy()
        return df

    except requests.RequestException as e:
        print(f"Network error occurred: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    empty_df = pd.DataFrame(columns=COLUMNS)
    _KEGG_PATHWAY_CACHE[ec_number] = empty_df.copy()
    return empty_df
