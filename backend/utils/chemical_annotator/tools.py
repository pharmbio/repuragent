"""
=== Chemical Annotator - Notebook Version ===
Fetches drug annotations from public repositories based on exact match with query pattern.
"""
VERSION = "1.0"
AUTHOR = "Flavio Ballante"
INSTITUTION = "2025, CBCS-SciLifeLab-Karolinska Institutet"
CONTACT = "flavio.ballante@ki.se, flavioballante@gmail.com"

import logging
from pathlib import Path
import pandas as pd
from backend.utils.chemical_annotator.misc_utils import process_compounds
from backend.utils.chemical_annotator.chembl_utils import process_targets, get_protein_classifications, trace_hierarchy, chembl_status
from backend.utils.chemical_annotator.kegg_utils import get_pathways_from_ec
from langchain_core.tools import tool
from backend.utils.output_paths import resolve_output_folder


@tool
def annotate_chemicals(
    input_file,
    output_prefix,
    format_type='AUTO',
    confidence_threshold=8,
    assay_type_in='B,F',
    pchembl_value_gte=6.0,
    log_file='chemical_annotator.log'
):
    """
    Annotate chemicals with data from ChEMBL database.
    
    Parameters
    ----------
    input_file : str
        Path to input CSV file containing compounds
    output_prefix : str
        Prefix for output CSV files
    format_type : str
        Type of chemical notation to use as query ('SMILES', 'InChI', 'InChIKey', or 'ANY'/'AUTO'); column matching is case-insensitive.
        'ANY'/'AUTO' will attempt to resolve arbitrary identifiers to SMILES before querying.
    confidence_threshold : int, optional
        Minimum confidence score value (default: 8)
    assay_type_in : str, optional
        Comma-separated list of assay types (default: 'B,F')
    pchembl_value_gte : float, optional
        Minimum pChEMBL value (default: 6.0)
    log_file : str, optional
        Path to log file (default: 'chemical_annotator.log')
    
    Returns
    -------
    dict
        Mapping of output file paths (written under the active conversation's `results/<thread_id>/` folder) to brief descriptions.
    """

    # Store artifacts under the active conversation's results directory.
    output_dir = resolve_output_folder()
    safe_prefix = "annotations"
    log_path = output_dir / Path(str(log_file)).name

    # Remove previous log file if it exists
    if log_path.exists():
        log_path.unlink()
    
    # Configure logging
    logging.basicConfig(
        filename=str(log_path),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger()
    
    try:
        # Add title/header to the log file
        logger.info("===========================================")
        logger.info("            Chemical Annotator             ")
        logger.info(f"Version:  {VERSION}")
        logger.info("Format: txt")
        logger.info(f"Author: {AUTHOR}")
        logger.info(f"Contact: {CONTACT}")
        logger.info(f"Institution: {INSTITUTION}")
        logger.info("===========================================")
        
        if chembl_status:
            logger.info(f"ChEMBL Database Version: {chembl_status['chembl_db_version']}")
            logger.info(f"ChEMBL Release Date: {chembl_status['chembl_release_date']}")
            logger.info(f"ChEMBL Status: {chembl_status['status']}")
            logger.info(f"Number of Activities: {chembl_status['activities']}")
            logger.info(f"Number of Distinct Compounds: {chembl_status['disinct_compounds']}")
            logger.info(f"Number of Targets: {chembl_status['targets']}")
        else:
            logger.warning("Unable to fetch ChEMBL status information")
        
        logger.info("===========================================")
        logger.info("")

        # Read input file
        compounds_list = pd.read_csv(input_file, delimiter=r",")

        # Fetch data from ChEMBL
        print("Processing compounds...")
        Drugs_data = process_compounds(
            compounds_list,
            format_type,
            confidence_threshold=confidence_threshold,
            assay_type_in=assay_type_in.split(','),
            pchembl_value_gte=pchembl_value_gte
        )

        # Shift index by 1
        Drugs_info = Drugs_data[0]
        Drugs_assay = Drugs_data[1]
        Drugs_MoA = Drugs_data[2]
        Drugs_info.index = Drugs_info.index + 1
        Drugs_assay.index = Drugs_assay.index + 1
        Drugs_MoA.index = Drugs_MoA.index + 1
     
        # Write data to separate .csv files
        Drugs_info_output = output_dir / f"{safe_prefix}_drugs_info.csv"
        Drugs_assay_output = output_dir / f"{safe_prefix}_drugs_assay.csv"
        Drugs_MoA_output = output_dir / f"{safe_prefix}_drugs_moa.csv"

        Drugs_info.to_csv(Drugs_info_output, index=False)
        Drugs_assay.to_csv(Drugs_assay_output, index=False)
        Drugs_MoA.to_csv(Drugs_MoA_output, index=False)

        print("All compounds have been processed. Now processing targets data...")
        logger.info("All compounds have been processed. Now processing targets data...")

        # Fetch target data from ChEMBL
        Targets_data = process_targets(Drugs_assay)
        Targets_data = Targets_data.reset_index(drop=True)
        Targets_data.index = Targets_data.index + 1
        
        # Process EC numbers and get pathway information
        print("Processing EC numbers and retrieving pathway information...")
        logger.info("Processing EC numbers and retrieving pathway information...")

        pathway_data = []
        unique_targets = Targets_data.drop_duplicates(subset=['target_chembl_id'])
        unique_targets = unique_targets.dropna(subset=['EC Numbers'])
        
        for _, row in unique_targets.iterrows():
            chembl_id = row['target_chembl_id']
            ec_list = row['EC Numbers']

            if pd.isna(ec_list):
                continue

            ec_numbers = ec_list.split(';')
            kegg_ids = []
            pathways = []

            for ec in ec_numbers:
                ec = ec.strip()
                ec_pathways = get_pathways_from_ec(ec)
    
                if not ec_pathways.empty:
                    kegg_ids.extend(ec_pathways['KEGG_ID'].unique())
                    pathways.extend(ec_pathways['Pathway'].unique())

            kegg_ids = list(dict.fromkeys(kegg_ids))
            pathways = list(dict.fromkeys(pathways))
    
            pathway_data.append({
                'target_chembl_id': chembl_id,
                'EC Numbers': ec_list,
                'KEGG_ID': ';'.join(kegg_ids),
                'Pathway': ';'.join(pathways)
            })
        
        pathway_data = pd.DataFrame(pathway_data)

        # Write pathway data to CSV
        pathway_data_output = output_dir / f"{safe_prefix}_pathway_info.csv"
        pathway_data.to_csv(pathway_data_output, index=False)
    
        # Merge pathway_data with Targets_data
        Targets_data = Targets_data.drop(columns=['EC Numbers'], errors='ignore')
        Targets_data_with_pathways = pd.merge(Targets_data, pathway_data, on='target_chembl_id', how='left')
        
        # Process protein hierarchy data
        print("Retrieving protein hierarchy information...")
        logger.info("Retrieving protein hierarchy information...")
        
        unique_targets = Targets_data.drop_duplicates(subset=['target_chembl_id']).copy()
        unique_targets['protein_classifications'] = unique_targets['target_chembl_id'].apply(get_protein_classifications)
        unique_targets['protein_hierarchy'] = unique_targets['protein_classifications'].apply(trace_hierarchy)
        
        Targets_data_with_pathways_p_class = Targets_data_with_pathways.merge(
            unique_targets[['target_chembl_id', 'protein_classifications', 'protein_hierarchy']],
            on='target_chembl_id',
            how='left'
        )

        Targets_data_with_pathways_p_class = Targets_data_with_pathways_p_class.reset_index(drop=True)
        Targets_data_with_pathways_p_class.index += 1

        # Concatenate dataframes horizontally
        Drugs_assay_Targets_data = pd.concat(
            [Drugs_assay, Targets_data_with_pathways_p_class.drop('target_chembl_id', axis=1)],
            axis=1
        )

        # Reorder columns
        targets_columns = [col for col in Targets_data_with_pathways_p_class.columns if col != 'target_chembl_id']
        insert_position = Drugs_assay_Targets_data.columns.get_loc('target_chembl_id') + 1

        new_column_order = (
            list(Drugs_assay_Targets_data.columns[:insert_position]) +
            targets_columns +
            [col for col in Drugs_assay_Targets_data.columns[insert_position:] if col not in targets_columns]
        )

        Drugs_assay_Targets_data = Drugs_assay_Targets_data[new_column_order]
        Drugs_assay_Targets_data = Drugs_assay_Targets_data.reset_index(drop=True)
        Drugs_assay_Targets_data.index = Drugs_assay_Targets_data.index + 1
        
        # Write final output files
        Targets_data_output = output_dir / f"{safe_prefix}_targets_info.csv"
        Drugs_assay_Targets_data_output = output_dir / f"{safe_prefix}_drugs_assay_targets_info.csv"

        Targets_data.to_csv(Targets_data_output, index=False)
        Drugs_assay_Targets_data.to_csv(Drugs_assay_Targets_data_output, index=False)

        output_files = {
            "Output description": "Belows are output files that can be used for subsequent analysis.",
            "Files":
                {
                    str(Drugs_info_output): "Compound-level ChEMBL molecule annotations + drug indications, merged onto the original input rows (often one row per matched indication).",
                    str(Drugs_assay_output): "ChEMBL bioactivity/activity records for matched molecules, merged onto the original input rows (often one row per activity/assay measurement).",
                    str(Drugs_MoA_output): "ChEMBL mechanism-of-action annotations for matched molecules (one row per MoA record when available).",
                    str(Targets_data_output): "Target metadata derived from assay targets (aligned to activity rows), including target description and UniProt ID when available.",
                    str(pathway_data_output): "KEGG pathway mappings derived from target EC numbers (one row per target–pathway pair when available).",
                    str(Drugs_assay_Targets_data_output): "Activity rows enriched with target metadata, KEGG pathway mappings, and protein classifications/hierarchy (when available).",
                    str(log_path): "Run log with tool/version metadata and progress messages.",
                },
        }
        
        print("Script execution completed successfully.")
        logger.info("Script execution completed successfully.")
        
        # Return only a lightweight description map for agent context
        return output_files
        
    except Exception as e:
        logger.exception("An error occurred: %s", str(e))
        raise
