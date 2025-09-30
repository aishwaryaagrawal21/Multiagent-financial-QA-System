# Imports
import pandas as pd
from database_proxy import Neo4jGraphStore, VectorDBClient, graph_client, vector_client
from knowledge_graph import KnowledgeGraph
from typing_extensions import Annotated
from QA import QAEngine
import os

def build_graph(rebuild_kg = False, triplet_data_path = None):
    '''
    Build the Knowledge Graph from the Triplets
    Params:
    - rebuild_kg (Bool): Indicate whether to re-build the KG or not. Default is True.
    - triplet_data_path (str): The path of the Processed Triplet Data CSVs. Default is None.
    '''
    # Create KG Class
    kg = KnowledgeGraph(graph_client, vector_client)
    # Re-build KG if required
    if rebuild_kg and triplet_data_path is not None:
        kg.reset_kg()
        for file in os.listdir(triplet_data_path):
            file_path = os.path.join(triplet_data_path, file)
            print(f'Processing: {file_path}')
            kg.process_csv(file_path)

def start_finatars(question, agent_type, search_type = None, web_search = False, llm_config_path = 'finatars_pipeline/llm_config.json'):
    '''
    Function to perform the QA task.
    Params:
    - question (str): The question to ask the QA System.
    - agent_type (str): Either 'single' or 'multi' to indicate type of QA system.
    - search_type (str): Either 'beam_search', 'beam_search_advanced' or 'vector_search' to indicate type of retrieval.
    - web_search (Bool): True if web_search needs to be incorporated into the answer, False otherwise.
    - llm_config_path (str): The path of the LLM config file.

    Returns the answer, along with additional data like chunk_text, chunk_source, and chunk_id
    '''

    # Start the QA Engine
    qa_engine = QAEngine(graph_client, vector_client)

    # Lower-case the question
    question = question.lower()

    # Single Agent QA
    if agent_type == 'single':
        # Check if retrieval type is specified
        if not search_type:
            print('Please specify retrieval type for single-agent QA. Either "beam_search", "beam_search_advanced" or "vector_search".')
            return
        else:
            answer, chunk_text, chunk_source, chunk_id = qa_engine.single_agent_qa(question, search_type)
            return answer, chunk_text, chunk_source, chunk_id
    
    # Multi-Agent QA
    if agent_type == 'multi':
        # Check if llm_config is specified
        if not llm_config_path:
            print('Please specify path for the LLM config for multi-agent QA.')
            return
        else:
            answer, chunk_text, chunk_source, chunk_id, web_data = qa_engine.multi_agent_qa(question, web_search, llm_config_path)
            return answer, chunk_text, chunk_source, chunk_id, web_data


# ------------------------For running directly--------------------------
# Uncomment below part to run directly without the UI. # For running through UI, use streamlit instructions

# # Initialize Parameters - Modify AS required
# rebuild_kg = False # Set to True if you wanna re-build the KG
# triplet_path = 'path' # Set triplet extraction final data path if you wanna re-build the KG
# agent_type = 'multi' # 'single' or 'multi'
# search_type = None # if using 'single', select 'beam_search', 'beam_search_advanced' or 'vector_search'
# web_search = True # True if you want internet search, False otherwise.
#
# # Build KG
# build_graph(rebuild_kg = rebuild_kg, triplet_data_path = triplet_path)
#
# # Start QA system
# print('Enter your question:')
# question = input()
# answer, chunk_text, chunk_source, chunk_id, web_data = start_finatars(question=question, agent_type=agent_type, search_type=search_type, web_search = web_search)