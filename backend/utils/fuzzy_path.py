import os
import re

from fuzzywuzzy import fuzz
from langchain_core.tools import tool
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent

from app.config import logger
from core.prompts.prompts import FILE_SEARCH_SYSTEM
llm = init_chat_model('gpt-4o')

@tool
def fuzzy_file_search(fuzzy_str: str, file_types: str = "csv"):
    """
    Efficiently search for files matching a fuzzy string, or return all files in a folder/directory if requested.
    Args:
        fuzzy_str (str): The search string, e.g. "all files in data folder" or "all in data directory".
        file_types (str): Space-separated list of file extensions.
    Returns:
        str or list: Best match (absolute path) or comma-separated list of all matching files.
    """
    root_dir = "."
    file_types_tuple = tuple(file_types.split())
    fuzzy_str_lower = fuzzy_str.lower().strip()

    # Improved: check for "all files in <folder|directory>" or "all in <folder|directory>"
    match = re.search(
        r'all(?: files)? in ([\w\-/\\\.]+)\s*(?:folder|directory)?', fuzzy_str_lower
    )
    if match:
        folder = match.group(1).strip("/\\ .")
        # Recursively search for all folders matching the name at any depth under root_dir and models/
        found_files = []
        for dirpath, *_ in os.walk(root_dir):
            if os.path.basename(dirpath).lower() == folder.lower():
                    files = [
                        os.path.join(dirpath, f)
                        for f in os.listdir(dirpath)
                        if os.path.isfile(os.path.join(dirpath, f)) and (f.endswith(file_types_tuple) or not file_types_tuple)
                    ]
                    found_files.extend(files)
        if found_files:
            return ", ".join(found_files)
        return None

    # Normal fuzzy search for best match
    best_score = -1
    best_path = None
    for root, _, files in os.walk(root_dir):
        for name in files:
            if name.endswith(file_types_tuple) or not file_types_tuple:
                path = os.path.join(root, name)
                score = fuzz.token_sort_ratio(fuzzy_str_lower, name.lower())
                if fuzzy_str_lower.replace(" ", "") in name.lower().replace("_", ""):
                    score += 20
                if score > best_score:
                    best_score = score
                    best_path = path

    return best_path if best_path else None

filesearch_agent = create_react_agent(model = llm,
                                        tools =[fuzzy_file_search],
                                        name='File_Search_Engine',
                                        prompt = FILE_SEARCH_SYSTEM)

@tool
def prompt_with_file_path(init_prompt: str):
    "Tool to refine prompt, replace file path description by actual file path"
    
    
    response = filesearch_agent.invoke({
        "messages": [
        {
            "role": "user",
            "content": init_prompt
            }
        ]
    })

    refine_prompt = response['messages'][-1].content
    return refine_prompt
