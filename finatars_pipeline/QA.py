from datasets import load_dataset
from database_proxy import Neo4jGraphStore, VectorDBClient, graph_client, vector_client
from knowledge_graph import KnowledgeGraph
from retrieval_functions import retrieval
from prompt import SINGLE_AGENT_QA_PROMPT, SINGLE_AGENT_QA_SYSTEM_PROMPT
from util import run_query
from agent import MultiLLMSystem
from typing_extensions import Annotated, List
# Internet Agent Imports
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

web_data = None
final_answer = None
graph_data = None  # list() text of the chunk
chunk_id   = None  # list() id of the chunk
chunk_source = None # list() source article name of the chunk (multiple chunks may have the same source)


#### SEARCH GRAPH FUNCTION (Without Web Search)
def search_graph(query:Annotated[str, "the question to ask"],
                 beam_width = 3,
                 depth = 7, 
                 limit = 5, 
                 retrieval_method = 'beam_search_advanced')->str:
    
    print("**PERFORMING GRAPH SEARCH**")
    text, source, id = retrieval(graph_client, 
                                 vector_client, 
                                 query, 
                                 depth, 
                                 beam_width, 
                                 limit=limit, 
                                 retrieval_method=retrieval_method,
                                 use_summary=False)
    # record the source 
    global chunk_id, graph_data, chunk_source
    # reset data
    chunk_id = None
    graph_data = None
    chunk_source = None
    # re-initialize data
    chunk_id = id
    graph_data = text
    chunk_source = source

    if not graph_data:
        return "GRAPH DATA:\n" + "" + "\n\nGRAPH SEARCH COMPLETE"
    else:
        return "GRAPH DATA:\n" + "\n".join(graph_data) + "\n\nGRAPH SEARCH COMPLETE"


#### SEARCH GRAPH AND WEB FUNCTION (With Web Search)
def search_graph_and_web(query:Annotated[str, "the question to ask"],
                 beam_width = 3,
                 depth = 7, 
                 limit = 5, 
                 retrieval_method = 'beam_search_advanced')->str:

    print("**PERFORMING GRAPH SEARCH**")
    text, source, id = retrieval(graph_client, 
                                 vector_client, 
                                 query, 
                                 depth, 
                                 beam_width, 
                                 limit=limit, 
                                 retrieval_method=retrieval_method,
                                 use_summary=False)
    
    # record the source 
    global chunk_id, graph_data, chunk_source, web_data, final_answer

    # reset data
    chunk_id = None
    graph_data = None
    web_data = None
    chunk_source = None
    final_answer = None

    # re-initialize data
    chunk_id = id
    graph_data = text
    chunk_source = source
    final_answer = "GRAPH DATA:\n"
    if not graph_data:
        final_answer += ""
    else:
        final_answer += "\n".join(graph_data)

    print("**PERFORMING WEB SEARCH**")
    final_answer += "\n\nWEB DATA:\n"
    wrapper = DuckDuckGoSearchAPIWrapper(region="us-en", time="w", max_results=2)
    if graph_data:
        search_agent = DuckDuckGoSearchResults(api_wrapper=wrapper, source="news")
        search_result = search_agent.run(query)
        web_data = []
        for segment in search_result.strip().split("], ["):
            segment = segment.strip("[] ")
            parts = segment.split(", title: ")
            snippet = parts[0].split(": ", 1)[1]
            title, link = parts[1].split(", link: ")
            web_data.append([snippet, link])
        final_answer += str(web_data)
    else:
        final_answer += str([["no web search as entities are not present in the graph"]])
    
    return final_answer + "\n\nGRAPH AND WEB SEARCH COMPLETE"


#### DEFINING QA ENGINE
class QAEngine:
    def __init__(self, graph_client:Neo4jGraphStore, vector_client:VectorDBClient, search = 'beam_search'):
        self.graph_client = graph_client
        self.vector_client = vector_client
        self.extraction_prompt_template = SINGLE_AGENT_QA_PROMPT
        self.single_agent_model = "gpt-3.5-turbo-0125"

    #### Single-Agent QA
    def single_agent_qa(self, 
                        question, 
                        retrieval_method = 'beam_search_advanced', 
                        depth = 7, 
                        beam_width=3, 
                        limit = 5,
                        use_summary=False):
        '''
        retrieval_method: choose between "beam_search" or "vector_search"

        return Tuple(answer:str, graph_data:list[str], chunk_source:list[str], chunk_id:list[str]), 
                answer to the question and a list of context chunks, the source and id of those chunks. 
        '''
        c_text, c_source, c_id = retrieval(self.graph_client, 
                                     self.vector_client, 
                                     question, 
                                     depth, 
                                     beam_width=beam_width, 
                                     limit=limit, 
                                     retrieval_method=retrieval_method,
                                     use_summary=use_summary)

        if not c_text:
            print("No document found, try to decrease threshold")
            return None, None, None, None 
        
        fin_prompt = self.extraction_prompt_template.format("\n".join(c_text), question)

        answer = run_query(fin_prompt, SINGLE_AGENT_QA_SYSTEM_PROMPT, self.single_agent_model)
        return answer, c_text, c_source, c_id
    
    #### Multi-Agent QA
    def multi_agent_qa(self, question, web_search, llm_config_path):
        '''
        for vector search, change the input for retrieval inside search_graph function,
        choose between "beam_search" or "vector_search"
        - web_search (Bool): True if web_search needs to be incorporated into the answer, False otherwise.
        '''
        # Check if internet agent is required
        if web_search:
            retrieval_function = search_graph_and_web
        else:
            retrieval_function = search_graph

        # Initialize the Multi-LLM system 
        multi_agent = MultiLLMSystem(llm_config_path, retrieval_function)

        # Check if internet agent is required
        if web_search:
            multi_agent.initialize_agents_with_web()
        else:
            multi_agent.initialize_agents_no_web()
        
        # Start the QA process
        multi_agent.start_conversation(question)

        # Get the formatted results
        answer = multi_agent.get_result()

        # Return document_source as well 
        return answer, graph_data, chunk_source, chunk_id, web_data

    def __del__(self):
        self.graph_client._driver.close()