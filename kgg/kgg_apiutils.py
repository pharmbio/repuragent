# This script contains some functions utils_v2.py, kg_gen_v5.py modified explicitly for the KGG RestAPI.
import logging
import random
import time
from collections import defaultdict
import networkx as nx
import pandas as pd
import requests
from chembl_webresource_client.new_client import new_client
from chembl_webresource_client.http_errors import BaseHttpException, HttpApplicationError
from pybel import BELGraph
from pybel.dsl import Protein, Abundance, Pathology, BiologicalProcess, Gene
from tqdm.auto import tqdm
import pybel
import json
import requests
import pandas as pd
from tqdm.auto import tqdm
from pandasgwas.get_variants import get_variants_by_efo_id
import re

from backend.utils.storage_paths import get_data_root

DATA_ROOT = get_data_root()

logger = logging.getLogger(__name__)
OPENTARGETS_GRAPHQL_URL = "https://api.platform.opentargets.org/api/v4/graphql"


def _opentargets_query(query_string, variables):
    response = requests.post(
        OPENTARGETS_GRAPHQL_URL,
        json={"query": query_string, "variables": variables},
        timeout=60,
    )
    response.raise_for_status()
    payload = response.json()
    errors = payload.get("errors") or []
    if errors:
        raise ValueError(errors[0].get("message") or "Open Targets GraphQL query failed")
    data = payload.get("data")
    if data is None:
        raise ValueError("Open Targets GraphQL response did not include data")
    return data


def _clinical_stage_to_phase(value):
    if value is None:
        return None
    stage = str(value).strip().upper()
    if not stage:
        return None
    if "APPROVED" in stage:
        return 4
    matches = [int(match) for match in re.findall(r"PHASE[_ ]?(\d)", stage)]
    if not matches:
        return None
    return max(matches)


def _normalize_disease_candidate_rows(disease_id, disease_name, rows):
    normalized_rows = []
    for row in rows or []:
        drug = row.get("drug") or {}
        drug_id = drug.get("id")
        if not drug_id:
            continue
        normalized_rows.append(
            {
                "approvedSymbol": None,
                "approvedName": None,
                "prefName": drug.get("name"),
                "drugType": drug.get("drugType"),
                "drugId": drug_id,
                "phase": _clinical_stage_to_phase(row.get("maxClinicalStage")),
                "ctIds": [
                    report.get("id")
                    for report in (row.get("clinicalReports") or [])
                    if report.get("id")
                ],
                "status": row.get("maxClinicalStage"),
                "id": disease_id,
                "disease": disease_name,
            }
        )
    return normalized_rows


def _normalize_target_candidate_rows(rows):
    normalized_rows = []
    for row in rows or []:
        drug = row.get("drug") or {}
        drug_id = drug.get("id")
        if not drug_id:
            continue
        disease_entries = []
        for disease_entry in row.get("diseases") or []:
            disease = (disease_entry or {}).get("disease") or {}
            disease_entries.append(
                {
                    "disease.id": disease.get("id"),
                    "disease.name": disease.get("name"),
                }
            )
        if not disease_entries:
            disease_entries = [{"disease.id": None, "disease.name": None}]
        for disease_entry in disease_entries:
            normalized_rows.append(
                {
                    "phase": _clinical_stage_to_phase(row.get("maxClinicalStage")),
                    "status": row.get("maxClinicalStage"),
                    "drug.id": drug_id,
                    "drug.name": drug.get("name"),
                    **disease_entry,
                }
            )
    return normalized_rows


def _retry_chembl_request(func, *, max_attempts=3, base_delay=1.0, jitter=0.25, context=""):
    for attempt in range(1, max_attempts + 1):
        try:
            return func()
        except (HttpApplicationError, BaseHttpException, requests.exceptions.RequestException) as exc:
            if attempt == max_attempts:
                raise
            delay = base_delay * (2 ** (attempt - 1)) + random.uniform(0, jitter)
            suffix = f" [{context}]" if context else ""
            logger.warning("ChEMBL request failed%s (attempt %s/%s): %s", suffix, attempt, max_attempts, exc)
            time.sleep(delay)


def getDrugCount(disease_id):
    """Returns the count of drugs associated with a disease.
    Parameters:
    disease_id (str): The disease ID.
    
    Returns:
    int: The count of drugs associated with the disease.
    """
    efo_id = disease_id
    query_string = """
        query ClinicalCandidatesFromDisease($my_efo_id: String!){
          disease(efoId: $my_efo_id){
            id
            name
            drugAndClinicalCandidates{
                count
            }
          }
        }

    """
    variables = {"my_efo_id": efo_id}
    data = _opentargets_query(query_string, variables)
    return (((data.get("disease") or {}).get("drugAndClinicalCandidates")) or {}).get("count", 0)


def GetDiseaseAssociatedDrugs(disease_id,CT_phase):
    """
    Returns a DataFrame of drugs associated with a disease.
    
    Parameters:
    disease_id (str): The disease ID.
    CT_phase (int): The clinical trial phase to filter drugs.
    
    Returns:
    pd.DataFrame: DataFrame containing drug information.
    """
    efo_id = disease_id
    size = getDrugCount(efo_id)
    query_string = """
        query ClinicalCandidatesFromDisease($my_efo_id: String!){
          disease(efoId: $my_efo_id){
            id
            name
            drugAndClinicalCandidates{
                count
                rows{
                    id
                    maxClinicalStage
                    drug{
                        id
                        name
                        drugType
                    }
                    clinicalReports{
                        id
                    }
                }
            }
          }
        }

    """
    variables = {"my_efo_id": efo_id}
    if size > 10000:
        return None
    data = _opentargets_query(query_string, variables)
    disease = data.get("disease") or {}
    candidate_rows = _normalize_disease_candidate_rows(
        disease_id=efo_id,
        disease_name=disease.get("name"),
        rows=((disease.get("drugAndClinicalCandidates") or {}).get("rows") or []),
    )
    df = pd.DataFrame(candidate_rows)
    if not df.empty:
        df["phase"] = pd.to_numeric(df["phase"], errors="coerce")
        df = df.loc[df["phase"].fillna(-1) >= int(CT_phase), :]
        return(df)    
    else:
        return None


def GetDiseaseSNPs(disease_id):
    """
    Returns a DataFrame of SNPs associated with a disease.
    Parameters:
    disease_id (str): The disease ID.
    
    Returns:
    pd.DataFrame: DataFrame containing SNP information.
    """
    try:
        snps = get_variants_by_efo_id(disease_id)
        snps_df = snps.genomic_contexts
        snps_df['disease_id'] = disease_id
        #snps_functional_class_df = snps.variants
        #snps_functional_class_df['disease_id'] = disease_id

        return(snps_df)
    except:
        return None


def GetDiseaseAssociatedProteins(disease_id,index_counter=0,merged_list=None):
    """
    Fetches disease-associated proteins from the Open Targets API.
    
    Parameters:
    disease_id (str): The disease ID.
    index_counter (int): Index for pagination.
    merged_list (list): List to store results.
    
    Returns:
    pd.DataFrame: DataFrame containing protein information.
    """
    efo_id = str(disease_id)
    query_string = """
        query associatedTargets($efoId: String!,$index:Int!){
          disease(efoId: $efoId){
            id
            name
            associatedTargets(page:{size:3000,index:$index}){
              count
              rows {
                target {
                  id
                  approvedSymbol
                  proteinIds {
                    id
                    source
                  }
                }
                score
              }
            }
          }
        }

    """
    if merged_list is None:
        merged_list = []
    variables = {"efoId":efo_id,"index":index_counter}
    data = _opentargets_query(query_string, variables)
    result = (((data.get("disease") or {}).get("associatedTargets")) or {}).get("rows") or []
    merged_list.extend(result)    
    if result:
        counter = index_counter+1
        GetDiseaseAssociatedProteins(disease_id,counter,merged_list)
    temp_list = []
    for item in merged_list:
        for obj in item['target']['proteinIds']:
            if obj['source'] == 'uniprot_swissprot':
                uprot = obj['id']
                source = obj['source']
                score = item['score']
                ensg = item['target']['id']
                name = item['target']['approvedSymbol']
                temp = {'Protein':name,'ENSG':ensg,'UniProt':uprot,'Source':source,'Score':score}
                temp_list.append(temp)
    df = pd.DataFrame(temp_list)
    df['disease_id'] = efo_id
    return(df)

def ExtractFromUniProt(uniprot_id) -> dict:
    """Uniprot parser to retrieve information about OMIM disease, reactome pathway, biological process,
     and molecular functions.

    :param uniprot_id:
    :return:
    """
    Uniprot_Dict = []
    mapped_uprot = []
    for id in tqdm(uniprot_id, desc='Fetching Protein-related info'):
        ret_uprot = requests.get(
            'https://www.uniprot.org/uniprot/' + id + '.txt'
        ).text.split('\n')

        if ret_uprot == ['']:
            continue

        id_copy = id
        mapped_uprot.append(id_copy)
        k = 0
        id = {}
        id['Disease'] = {}
        id['Reactome'] = {}
        id['Function'] = {}
        id['BioProcess'] = {}
        id['Gene'] = {}
        for line in ret_uprot:
            if '-!- DISEASE:' in line:
                if ('[MIM:' in line):
                    dis = line.split(':')
                    id['Disease'].update({dis[1][1:-5]: dis[2][:-1]})
            if 'Reactome;' in line:
                ract = line.split(';')
                id['Reactome'].update({ract[2][1:-2]: ract[1][1:]})
            if ' F:' in line:
                fn = line.split(';')
                id['Function'].update({fn[2][3:]: fn[1][1:]})
            if ' P:' in line and 'GO;' in line:
                bp = line.split(';')
                id['BioProcess'].update({bp[2][3:]: bp[1][1:]})
            if 'GN   Name' in line:
                if k == 0:
                    gene = line.split('=')
                    gene = gene[1].split(' ')
                    if ';' in gene[0]:
                        gene = gene[0].split(';')
                        gene = {'Gene': gene[0]}
                    else:
                        gene = {'Gene': gene[0]}
                    id.update(gene)
                    k += 1
        Uniprot_Dict.append(id)
    Uniprot_Dict = dict(zip(mapped_uprot, Uniprot_Dict))
    return Uniprot_Dict

def RetMech(chemblIds) -> dict:
    """Function to retrieve mechanism of actions and target proteins from ChEMBL

    :param chemblIds:
    :return:
    """
    getMech = new_client.mechanism
    mechList = []
    for chemblid in tqdm(chemblIds, desc='Retrieving mechanisms from ChEMBL'):
        def _fetch_mechs():
            return list(getMech.filter(
                molecule_chembl_id=chemblid
            ).only(['mechanism_of_action', 'target_chembl_id', 'action_type']))

        mechList.append(_retry_chembl_request(_fetch_mechs, context=f"mechanism {chemblid}"))

    named_mechList = dict(zip(chemblIds, mechList))
    named_mechList = {
        k: v
        for k, v in named_mechList.items()
        if v
    }
    return named_mechList

def RetAct(chemblIds) -> dict:
    """Function to retrieve associated assays from ChEMBL

    :param chemblIds:
    :return:
    """
    GetAct = new_client.activity
    getTar = new_client.target
    ActList = []
    filtered_list=['assay_chembl_id','assay_type','pchembl_value','target_chembl_id',
                   'target_organism','bao_label','target_type']
    for chembl in tqdm(chemblIds, desc='Retrieving bioassays from ChEMBL'):
        def _fetch_acts():
            return list(GetAct.filter(
                molecule_chembl_id=chembl,
                pchembl_value__isnull=False,
                assay_type_iregex='(B|F)',
                target_organism='Homo sapiens'
            ).only(filtered_list))

        acts = _retry_chembl_request(_fetch_acts, context=f"activity {chembl}")
        data = []
        for d in acts:
            if float(d.get('pchembl_value')) < 6:
                continue      
            if (d.get('bao_label') != 'single protein format'):
                continue
            tar = d.get('target_chembl_id')   
            tar_dict = _retry_chembl_request(lambda: getTar.get(tar), context=f"target {tar}")
            try:
                if tar_dict['target_type'] in ('CELL-LINE', 'UNCHECKED'):
                    continue
            except KeyError:
                continue
            data.append(d)
        ActList.append(list(data))
    named_ActList = dict(zip(chemblIds, ActList))
    named_ActList = {
        k: v
        for k, v in named_ActList.items()
        if v
    }
    return named_ActList

def Ret_chembl_protein(sourceList) -> list:
    """Method to retrieve ChEMBL ids which are proteins/targets

    :param sourceList:
    :return:
    """
    protein_List = []
    for item in sourceList:
        for j in range(len(sourceList[item])):
            protein_List.append(sourceList[item][j]['target_chembl_id'])

    protein_List = set(protein_List)
    protein_List = list(filter(None, protein_List))
    return protein_List

def chembl2uniprot(chemblIDs) -> dict:
    """Method to convert ChEMBL id to UNIPROT and get associated REACTOME pathways

    :param chemblIDs:
    :return:
    """
    getTarget = new_client.target
    chem2Gene2path = []
    chemHasNoPath = set()
    chemNotprotein = set()
    chem2path = defaultdict(list)

    for chemblid in tqdm(chemblIDs, desc='Filtering UniProt proteins from ChEMBL'):
        chem = getTarget.filter(
            chembl_id=chemblid
        ).only('target_components')
        try:
            uprot_id = chem[0]['target_components'][0]['accession']
            if not uprot_id:
                chemHasNoPath.add(chemblid)
        except IndexError:
            chemHasNoPath.add(chemblid)
    chemblIDs_filtered = [
        item
        for item in chemblIDs
        if item not in chemHasNoPath
    ]
    for chemblid in tqdm(chemblIDs_filtered, desc='Filtering human proteins from ChEMBL'):
        chem = getTarget.filter(chembl_id=chemblid).only('target_components')
        getGene = chem[0]['target_components'][0]['target_component_synonyms']
        try:
            getGene = [item for item in getGene if item["syn_type"] == "GENE_SYMBOL"][0]
            if not getGene:
                chemNotprotein.add(chemblid)
        except IndexError:
            chemNotprotein.add(chemblid)
    chemblIDs_filtered = [
        item
        for item in chemblIDs_filtered
        if item not in chemNotprotein
    ]
    for chemblid in tqdm(chemblIDs_filtered, desc='Populating ChEMBL data for human proteins'):
        chem = getTarget.filter(
            chembl_id=chemblid
        ).only('target_components')
        uprot_id = chem[0]['target_components'][0]['accession']
        getGene = chem[0]['target_components'][0]['target_component_synonyms']
        getGene = [item for item in getGene if item["syn_type"] == "GENE_SYMBOL"][0]
        chem2path = [item for item in chem[0]['target_components'][0]['target_component_xrefs'] if
                     item["xref_src_db"] == "Reactome"]
        uprot = {'accession': uprot_id}
        chem2path.append(uprot)
        chem2path.append(getGene)
        chem2Gene2path.append(chem2path)
    named_chem2Gene2path = dict(zip(chemblIDs_filtered, chem2Gene2path))
    named_chem2Gene2path = {
        k: v
        for k, v in named_chem2Gene2path.items()
        if v
    }
    return named_chem2Gene2path

def chembl2gene2path(
    chem2geneList,
    ActList
):
    """Method for updating chembl protein nodes with gene symbol.

    :param chem2geneList:
    :param ActList:
    :return:
    """
    for item in chem2geneList:
        sizeOfitem = len(chem2geneList[item])
        gene = chem2geneList[item][sizeOfitem - 1]['component_synonym']
        uprot = chem2geneList[item][-2]['accession']
        for jtem in ActList:
            for i in range(len(ActList[jtem])):
                if item == ActList.get(jtem)[i]['target_chembl_id']:
                    newkey = {'Protein': gene,'Accession':uprot}
                    ActList[jtem][i].update(newkey)
    return ActList


def chem2moa_rel(
    named_mechList,
    org,otp_prots,
    graph: BELGraph
) -> BELGraph:
    """Method to create the monkeypox graph

    :param named_mechList:
    :param org:
    :param graph: BEL graph of Monkeypox
    :return:
    """
    pos = ['POSITIVE ALLOSTERIC MODULATOR','AGONIST','ACTIVATOR','PARTIAL AGONIST']
    neg = ['INHIBITOR','NEGATIVE ALLOSTERIC MODULATOR','ANTAGONIST','BLOCKER']
    misc = ['MODULATOR','DISRUPTING AGENT','SUBSTRATE','OPENER','SEQUESTERING AGENT']
    for chembl_name, chembl_entries in tqdm(named_mechList.items(), desc='Populating Chemical-MoA edges'):
        for info in chembl_entries:
            graph.add_association(
                Abundance(namespace='ChEMBL', name=chembl_name),
                BiologicalProcess(namespace='MOA', name=info['mechanism_of_action']),
                citation='ChEMBL database',
                evidence='ChEMBL query'
            )
            if not info['target_chembl_id']:
                continue
            if 'Protein' in info and info['Accession'] in otp_prots:
                if info['action_type'] in pos:
                    graph.add_increases(
                        Abundance(namespace='ChEMBL', name=chembl_name),
                        Protein(namespace=org, name=info['Protein']),
                        citation='ChEMBL database',
                        evidence='ChEMBL query'
                    )
                if info['action_type'] in neg and info['Accession'] in otp_prots:
                    graph.add_decreases(
                        Abundance(namespace='ChEMBL', name=chembl_name),
                        Protein(namespace=org, name=info['Protein']),
                        citation='ChEMBL database',
                        evidence='ChEMBL query'
                    )

                if info['action_type'] in misc and info['Accession'] in otp_prots:
                    graph.add_association(
                        Abundance(namespace='ChEMBL', name=chembl_name),
                        Protein(namespace=org, name=info['Protein']),
                        citation='ChEMBL database',
                        evidence='ChEMBL query'
                    )
    return graph

def chem2act_rel(
    named_ActList,
    org,otp_prots,
    graph: BELGraph
) -> BELGraph:
    """Method to add bioassay edges to the KG.

    :param named_ActList:
    :param org:
    :param graph:
    :return:
    """
    for chemical, chem_entries in tqdm(named_ActList.items(), desc='Adding bioassay edges to BEL'):
        for chem_data in chem_entries:
            if chem_data['target_chembl_id']:
                if 'Protein' in chem_data and chem_data['Accession'] in otp_prots:
                    graph.add_association(
                        Abundance(namespace='ChEMBLAssay', name=chem_data['assay_chembl_id']),
                        Protein(namespace=org, name=chem_data['Protein']),
                        citation='ChEMBL database',
                        evidence='ChEMBL query'
                    )
            graph.add_association(
                Abundance(namespace='ChEMBL', name=chemical),
                Abundance(namespace='ChEMBLAssay', name=chem_data['assay_chembl_id']),
                citation='ChEMBL database',
                evidence='ChEMBL query',
                annotation={
                    'assayType': chem_data['assay_type'],
                    'pChEMBL': chem_data['pchembl_value']
                }
            )
    return graph

def gene2path_rel(
    named_chem2geneList,
    org,otp_prots,
    graph
) -> BELGraph:
    """Method to add protein and reactome data to KG

    :param named_chem2geneList:
    :param org:
    :param graph:
    :return:
    """
    for item in named_chem2geneList:
        itemLen = len(named_chem2geneList[item]) - 1
        for j in range(itemLen - 1):
            if named_chem2geneList[item][itemLen - 1]['accession'] in otp_prots:
            
                graph.add_association(
                    Protein(namespace=org, name=named_chem2geneList[item][itemLen]['component_synonym']),
                    BiologicalProcess(namespace='Reactome', name=named_chem2geneList[item][j]['xref_name']),
                    citation='ChEMBL database',
                    evidence='ChEMBL query',
                    annotation={
                        'Reactome': 'https://reactome.org/content/detail/'+named_chem2geneList[item][j]['xref_id']
                    }
                )
    return graph

def getAdverseEffectCount(chembl_id):
    get_id = chembl_id
    query_string = """
        query AdverseEventsQuery(
          $chemblId: String!
          $index: Int = 0
          $size: Int = 10
        ) {
          drug(chemblId: $chemblId) {
            id
            maxLlr: adverseEvents(page: { index: 0, size: 1 }) {
              rows {
                logLR
              }
            }
            adverseEvents(page: { index: $index, size: $size }) {
              criticalValue
              count
              rows {
                name
                count
                logLR
                meddraCode
              }
            }
          }
        }

    """
    variables = {"chemblId": get_id}
    data = _opentargets_query(query_string, variables)
    return (((data.get("drug") or {}).get("adverseEvents")) or {}).get("count", 0)

def GetAdverseEvents(chem_list):
    api_response = pd.DataFrame()
    for chem in tqdm(chem_list, desc = 'Retrieving Adverse Effects for each drug'):
        chembl_id = chem
        try:
            count = getAdverseEffectCount(chembl_id)
            query_string = """
                query AdverseEventsQuery(
                  $chemblId: String!
                  $index: Int = 0
                  $size: Int!
                ) {
                  drug(chemblId: $chemblId) {
                    id
                    maxLlr: adverseEvents(page: { index: 0, size: 1 }) {
                      rows {
                        logLR
                      }
                    }
                    adverseEvents(page: { index: $index, size: $size }) {
                      criticalValue
                      count
                      rows {
                        name
                        count
                        logLR
                        meddraCode
                      }
                    }
                  }
                }

        """
            variables = {"chemblId": chembl_id, "size": count}
            data = _opentargets_query(query_string, variables)
            api_response_temp = (((data.get("drug") or {}).get("adverseEvents")) or {}).get("rows") or []
            api_response_temp = pd.DataFrame(api_response_temp)
            api_response_temp ['chembl_id'] = chembl_id
            api_response = pd.concat([api_response,api_response_temp])
        except:
            continue
    api_response.reset_index(drop=True, inplace=True)
    return(api_response)

def chembl2adverseEffect_rel(
    chembl_adveff_df,
    graph: BELGraph
) -> BELGraph:
    """
    :param chembl_adveff_df:
    :param graph:
    :return:
    """
    for i in range(len(chembl_adveff_df)):
        graph.add_qualified_edge(
            Abundance(namespace='ChEMBL', name= str(chembl_adveff_df['chembl_id'][i])),
            Pathology(namespace='SideEffect', name= str(chembl_adveff_df['name'][i])),  
            relation='hasSideEffect',
            citation="OpenTargets Platform",
            evidence='DrugReactions'
        )
    return graph

def getNodeList(nodeName,graph):
    node_list = []
    for node in graph.nodes():
        if isinstance(node,pybel.dsl.Abundance):
            if node.namespace == nodeName:
                node_list.append(node.name)
    return(node_list)

def chembl_annotation(graph):
    chemblids = getNodeList('ChEMBL',graph)
    for item in tqdm(chemblids,desc='Adding ChEMBL URLs'):
        nx.set_node_attributes(graph,{Abundance(namespace='ChEMBL',
                                               name=item):'https://www.ebi.ac.uk/chembl/compound_report_card/'+item},'ChEMBL')
    return graph

def chembl_name_annotation(graph, drugs_df):
    drugName_chembl_dict = {}
    for v, k in drugs_df[['prefName', 'drugId']].values:
        drugName_chembl_dict.update({k: v})
    molecule = new_client.molecule
    chemblids = getNodeList('ChEMBL', graph)
    for chem in tqdm(chemblids, desc='Adding preferred names and trade names'):
        trade_names = []
        getNames = molecule.filter(molecule_chembl_id=chem).only(['pref_name', 'molecule_synonyms'])
        nx.set_node_attributes(graph, {Abundance(namespace='ChEMBL', name=chem): drugName_chembl_dict[chem]},
                               'PreferredName')
        try:
            for item in getNames[0]['molecule_synonyms']:
                if item['syn_type'] == 'TRADE_NAME':
                    trade_names.append(item['molecule_synonym'])
            nx.set_node_attributes(graph, {Abundance(namespace='ChEMBL', name=chem): trade_names}, 'TradeName')
        except:
            continue
    return graph

def uniprot_rel(
    named_uprotList,
    org,
    graph
) -> BELGraph:
    """Method to add UniProt related edges
    :param named_uprotList:
    :param org:
    :param graph:
    :return:
    """
    for item in tqdm(named_uprotList,desc='Populating Uniprot edges'):
        fun = list(named_uprotList[item]['Function'].keys())
        bp = list(named_uprotList[item]['BioProcess'].keys())
        pathway = list(named_uprotList[item]['Reactome'].keys())
        for f in fun:
            if str(named_uprotList[item]['Gene']) != 'nan' and not isinstance(named_uprotList[item]['Gene'], dict):
                graph.add_qualified_edge(
                    Protein(namespace=org, name=named_uprotList[item]['Gene']),
                    BiologicalProcess(namespace='GOMF', name=f),
                    relation='hasMolecularFunction',
                    citation='UniProt database',
                    evidence='UniProt query'
                )
            else:
                graph.add_qualified_edge(
                    Protein(namespace=org, name=item),
                    BiologicalProcess(namespace='GOMF', name=f),
                    relation='hasMolecularFunction',
                    citation='UniProt database',
                    evidence='UniProt query'
                )

        for b in bp:
            if str(named_uprotList[item]['Gene']) != 'nan' and not isinstance(named_uprotList[item]['Gene'], dict):
                graph.add_qualified_edge(
                    Protein(namespace=org, name=named_uprotList[item]['Gene']),
                    BiologicalProcess(namespace='GOBP', name=b),
                    relation='hasBiologicalProcess',
                    citation='UniProt database',
                    evidence='UniProt query'
                )
            else:
                graph.add_qualified_edge(
                    Protein(namespace=org, name=item),
                    BiologicalProcess(namespace='GOBP', name=b),
                    relation='hasBiologicalProcess',
                    citation='UniProt database',
                    evidence='UniProt query'
                )
        for path in pathway:
            if str(named_uprotList[item]['Gene']) != 'nan' and not isinstance(named_uprotList[item]['Gene'], dict):
                graph.add_qualified_edge(
                    Protein(namespace=org, name=named_uprotList[item]['Gene']),
                    BiologicalProcess(namespace='Reactome', name=path),
                    relation='hasPathway',
                    citation='UniProt database',
                    evidence='UniProt query',
                    annotation = {
                        'Reactome':f"https://reactome.org/content/detail/{named_uprotList[item]['Reactome'][path]}"
                    }
                )
            else:
                graph.add_qualified_edge(
                    Protein(namespace=org, name=item),
                    BiologicalProcess(namespace='Reactome', name=path),
                    relation='hasPathway',
                    citation='UniProt database',
                    evidence='UniProt query',
                    annotation = {'Reactome':f"https://reactome.org/content/detail/{named_uprotList[item]['Reactome'][path]}"}
                )
        
        if str(named_uprotList[item]['Gene']) != 'nan' and not isinstance(named_uprotList[item]['Gene'], dict):
            nx.set_node_attributes(graph,{Protein(namespace=org, name=named_uprotList[item]['Gene']):'https://3dbionotes.cnb.csic.es/?queryId='+item},'3Dbio')            
            nx.set_node_attributes(graph,{Protein(namespace=org, name=named_uprotList[item]['Gene']):'https://www.uniprot.org/uniprotkb/'+item},'UniProt')
        else:
            nx.set_node_attributes(graph,{Protein(namespace=org, name=item):'https://3dbionotes.cnb.csic.es/?queryId='+item},'3Dbio')
            nx.set_node_attributes(graph,{Protein(namespace=org, name=item):'https://www.uniprot.org/uniprotkb/'+item},'UniProt')
    return graph

def getGeneOntolgyNodes(nodeName,graph):
    node_list = []
    for node in graph.nodes():
        if isinstance(node,pybel.dsl.BiologicalProcess):
            if node.namespace == nodeName:
                node_list.append(node.name)
    return(node_list)

def gene_ontology_annotation(graph,uprotDict):
    gobp_dict = {}
    gomf_dict = {}
    for prot in uprotDict:
        gobp_dict.update(uprotDict[prot]['BioProcess'])        
    for prot in uprotDict:
        gomf_dict.update(uprotDict[prot]['Function'])
    bp_kg = getGeneOntolgyNodes('GOBP',graph)
    mf_kg = getGeneOntolgyNodes('GOMF',graph)    
    for item in tqdm(bp_kg,desc='adding biological process annotations'):
        gobp_id = gobp_dict[item]
        nx.set_node_attributes(graph,{BiologicalProcess(namespace='GOBP', name=item):'https://www.ebi.ac.uk/QuickGO/term/'+gobp_id},'QuickGO')
        nx.set_node_attributes(graph,{BiologicalProcess(namespace='GOBP', name=item):gobp_id},'Gene Ontology identifier')    
    for item in tqdm(mf_kg,desc='adding molecular function annotations'):
        gomf_id = gomf_dict[item]
        nx.set_node_attributes(graph,{BiologicalProcess(namespace='GOMF', name=item):'https://www.ebi.ac.uk/QuickGO/term/'+gomf_id},'QuickGO')
        nx.set_node_attributes(graph,{BiologicalProcess(namespace='GOMF', name=item):gomf_id},'Gene Ontology identifier')        
    return(graph)
    
def protein_annotation_druggability(graph):
    df = pd.read_csv(DATA_ROOT / "api_related_data" / "DruggableProtein_annotation_OT.csv")
    protein_list = []
    for node in graph.nodes():
        if isinstance(node,pybel.dsl.Protein):
            if node.namespace == 'HGNC':
                protein_list.append(node.name)                
    mapping_dict_id_symbol = {}
    for k,v in df[['approvedSymbol','ENSG']].values:
        mapping_dict_id_symbol.update({k:v})        
    mapping_dict_prot_drugability = {}
    for k,v in df[['approvedSymbol','Druggable Family']].values:
        mapping_dict_prot_drugability.update({k:v})
    for protein in tqdm(protein_list,desc='Adding druggability annotation'):
        if protein in mapping_dict_id_symbol.keys():                     
            nx.set_node_attributes(graph,{Protein(namespace="HGNC",name=protein):f"https://platform.opentargets.org/target/{mapping_dict_id_symbol[protein]}"},'OpenTargets')
            nx.set_node_attributes(graph,{Protein(namespace="HGNC",name=protein):mapping_dict_prot_drugability[protein]},'Druggability')
        else: 
            nx.set_node_attributes(graph,{Protein(namespace="HGNC",name=protein):'No'},'Druggability')          
    return(graph)

def protein_annotation_disease_score(graph, dis2prot_df):
    """Add disease association scores from Open Targets to HGNC protein nodes."""
    if dis2prot_df is None or dis2prot_df.empty:
        return graph
    score_map = (
        dis2prot_df[['Protein', 'Score']]
        .dropna(subset=['Protein'])
        .groupby('Protein', as_index=True)['Score']
        .max()
        .to_dict()
    )
    if not score_map:
        return graph
    for node in graph.nodes():
        if isinstance(node, pybel.dsl.Protein) and node.namespace == 'HGNC':
            score = score_map.get(node.name)
            if score is not None:
                nx.set_node_attributes(graph, {node: score}, 'DiseaseAssociationScore')
    return graph

def getProtfromKG(mainGraph):
    prot_list = []
    for u, v, data in tqdm(mainGraph.edges(data=True),desc='Filtering Proteins/Genes'):        
        if 'HGNC' in u.namespace:
            if u.name not in prot_list:
                prot_list.append(u.name)
        if 'HGNC' in v.namespace:
            if v.name not in prot_list:
                prot_list.append(v.name)
    return(prot_list)

def snp2gene_rel(snp_df,graph):
    kg_prots = getProtfromKG(graph)
    unique_prots_df = pd.DataFrame(kg_prots,columns=['Proteins'])    
    snp_df = unique_prots_df.merge(snp_df, how='inner', left_on='Proteins', right_on='gene.geneName')
    snp_df = snp_df.loc[snp_df['distance'] == 0]
    snp_df = snp_df.reset_index(drop=True)
    for i in tqdm(range(len(snp_df)),desc='Adding disease associated SNPs'):    
        graph.add_qualified_edge(
        Gene(namespace="dbSNP",name=snp_df['rsId'][i]),
        Protein(namespace = "HGNC", name = snp_df['Proteins'][i]),
        relation='hasGeneticVariant',
        citation = "GWAS Central",
        evidence = "SNPs for queried disease"
    )
    return(graph)


def searchDisease(keyword,logger=None):
    """Finding disease identifiers using OpenTargets API"""
    disease_name = str(keyword)
    if disease_name is None:
        return None
    query_string = """
        query searchAnything ($disname:String!){
            search(queryString:$disname,entityNames:"disease",page:{size:20,index:0}){
                total
                hits {
                    id
                    entity
                    name
                    description
                }
            }
        }
        """
    variables = {"disname": disease_name}
    data = _opentargets_query(query_string, variables)
    df = pd.DataFrame(((data.get("search") or {}).get("hits")) or [])
    if not df.empty:
        df = df[df["entity"] == "disease"]
        df.drop(columns=["entity"], inplace=True)
        return df
    else:
        return None

def getDrugsforProteins_count(ensg):
    
    #get_id = protein

    query_string = """
            
        query DrugAndClinicalCandidatesQuery($ensgId: String!) {
          target(ensemblId: $ensgId) {
            id
            drugAndClinicalCandidates {
              count
              }
            }
        }

    """

    # Set variables object of arguments to be passed to endpoint
    variables = {"ensgId": ensg}
    data = _opentargets_query(query_string, variables)
    return (((data.get("target") or {}).get("drugAndClinicalCandidates")) or {}).get("count", 0)


def getDrugsforProteins(prot_list):

    api_response = pd.DataFrame()
        
    df = pd.read_csv(DATA_ROOT / "api_related_data" / "DruggableProtein_annotation_OT.csv")
    
    mapping_dict_id_symbol = {}
    for k,v in df[['approvedSymbol','ENSG']].values:
        mapping_dict_id_symbol.update({k:v})
    
    #print(prot)
    #print(mapping_dict_id_symbol[prot])
    
    for prot in tqdm(prot_list):
        
        #print(prot)
        #print(mapping_dict_id_symbol[prot])
        
        try:
            count = getDrugsforProteins_count(mapping_dict_id_symbol[prot])

            query_string = """

                query DrugAndClinicalCandidatesQuery($ensgId: String!) {
                  target(ensemblId: $ensgId) {
                    id
                    drugAndClinicalCandidates {
                      count
                      rows {
                        id
                        maxClinicalStage
                        drug {
                          id
                          name
                        }
                        diseases {
                          disease {
                            id
                            name
                          }
                        }
                      }
                    }
                  }
                }


            """

            # Set variables object of arguments to be passed to endpoint
            variables = {"ensgId": mapping_dict_id_symbol[prot]}
            data = _opentargets_query(query_string, variables)
            rows = (((data.get("target") or {}).get("drugAndClinicalCandidates")) or {}).get("rows") or []
            api_response_temp = pd.json_normalize(_normalize_target_candidate_rows(rows))
            #return(df)
            api_response_temp['Protein'] = prot
            
            #api_response_temp = pd.DataFrame(api_response_temp)
            
            
            api_response = pd.concat([api_response,api_response_temp])


        except:
            continue

    api_response.reset_index(drop=True, inplace=True)
    return(api_response)
    

def createKG(disease_id: str, clinical_trial_phase: int, protein_threshold: float, logger=None):
    """Creates a Knowledge Graph for a given disease ID.
    
    Parameters:
    efo_id (str): The disease ID.
    clinical_trial_phase (int): The clinical trial phase to filter drugs.
    protein_threshold (float): The threshold for protein scores.
    logger: Optional logger instance for logging operations.
    
    Returns:
    dict: A dictionary containing the Knowledge Graph data.
    """
    if logger is None:
        import logging
        logger = logging.getLogger(__name__)
    if logger is None:
        import logging
        logger = logging.getLogger(__name__)
    
    logger.info(f"Starting KG creation for disease_id: {disease_id}")
    logger.info(f"Parameters - CT phase: {clinical_trial_phase}, protein threshold: {protein_threshold}")
    
    drugs_df = GetDiseaseAssociatedDrugs(disease_id=disease_id, CT_phase=clinical_trial_phase)
    if drugs_df is not None:
        logger.info(f"Shape of drugs DataFrame: {drugs_df.shape}")
    
    dis2prot_df = GetDiseaseAssociatedProteins(disease_id=disease_id)
    if dis2prot_df is not None:
        logger.info(f"Shape of disease-associated proteins DataFrame: {dis2prot_df.shape}")
    
    dis2snp_df = GetDiseaseSNPs(disease_id=disease_id)
    if dis2snp_df is not None:
        logger.info(f"Shape of disease-associated SNPs DataFrame: {dis2snp_df.shape}")
    
    # Thresholding on dis2prot_df
    if dis2prot_df is not None and not dis2prot_df.empty:
        uprot_df = dis2prot_df.loc[dis2prot_df['Score'] >= float(protein_threshold),:]
        logger.info(f"Filtered disease-associated proteins DataFrame shape: {uprot_df.shape}")
    else:
        uprot_df = pd.DataFrame(columns=['UniProt', 'Score'])

    adv_effect = pd.DataFrame()
          
    kg_name = disease_id + '_KG'
    kg = pybel.BELGraph(name=kg_name, version="0.0.1")
    
    uprot_list = list(set(uprot_df['UniProt']))
    logger.info(f"Processing {len(uprot_list)} unique UniProt IDs")

    uprot_ext = ExtractFromUniProt(uprot_list)
    logger.info(f"Extracted UniProt data for {len(uprot_ext)} proteins")

    if drugs_df is not None:
        logger.info("Processing drug data and mechanisms...")
        chembl2mech = RetMech(list(set(drugs_df['drugId'])))
        logger.info(f"Retrieved mechanisms for {len(chembl2mech)} drugs")
        
        chembl2act = RetAct(list(set(drugs_df['drugId'])))
        logger.info(f"Retrieved activities for {len(chembl2act)} drugs")
        
        prtn_as_chembl = Ret_chembl_protein(chembl2act) + Ret_chembl_protein(chembl2mech)
        prtn_as_chembl = set(prtn_as_chembl)
        prtn_as_chembl = list(prtn_as_chembl)
        logger.info(f"Retrieved {len(prtn_as_chembl)} ChEMBL proteins")
        
        chembl2uprot = chembl2uniprot(prtn_as_chembl)
        
        chembl2act = chembl2gene2path(chembl2uprot, chembl2act)
        logger.info(f"Processed ChEMBL activities for {len(chembl2act)} drugs")
        chembl2mech = chembl2gene2path(chembl2uprot, chembl2mech)
        logger.info(f"Processed ChEMBL mechanisms for {len(chembl2mech)} drugs")
        kg = chem2moa_rel(chembl2mech,'HGNC',uprot_list, kg)
        logger.info(f"Added chemical-mechanism relationships to KG with {len(kg.nodes())} nodes and {len(kg.edges())} edges")
        kg = chem2act_rel(chembl2act,'HGNC',uprot_list, kg)
        logger.info(f"Added chemical-activity relationships to KG with {len(kg.nodes())} nodes and {len(kg.edges())} edges") 
        kg = gene2path_rel(chembl2uprot, 'HGNC',uprot_list, kg)
        logger.info(f"Added gene-pathway relationships to KG with {len(kg.nodes())} nodes and {len(kg.edges())} edges")
        adv_effect = GetAdverseEvents(list(set(drugs_df['drugId'])))
        logger.info(f"Retrieved adverse effects for {len(adv_effect)} drugs")
        kg = chembl2adverseEffect_rel(adv_effect,kg)
        logger.info(f"Added adverse effects to KG with {len(kg.nodes())} nodes and {len(kg.edges())} edges")
        kg = chembl_annotation(kg)
        logger.info("Added ChEMBL annotations to KG")
        kg = chembl_name_annotation(kg,drugs_df)
        logger.info("Added drug names to KG")
    else:
        logger.warning("No drug data found for the given parameters")
    kg = uniprot_rel(uprot_ext, 'HGNC', kg)
    logger.info(f"Added UniProt relationships to KG with {len(kg.nodes())} nodes and {len(kg.edges())} edges")    
    if dis2prot_df is not None:
        kg = protein_annotation_disease_score(kg, uprot_df)
        logger.info("Added disease association scores to protein nodes")
    kg = gene_ontology_annotation(kg,uprot_ext)
    logger.info("Added Gene Ontology annotations to KG")
    kg = protein_annotation_druggability(kg)
    logger.info("Added druggability annotations to KG")
    if dis2snp_df is not None:
        kg = snp2gene_rel(dis2snp_df,kg)
        logger.info(f"Added SNP-gene relationships to KG with {len(kg.nodes())} nodes and {len(kg.edges())} edges")
    else:
        logger.info("No SNP data available for integration")        
    logger.info(f"Knowledge graph creation completed successfully. Graph contains {len(kg.nodes())} nodes and {len(kg.edges())} edges")
    return(kg)
    
