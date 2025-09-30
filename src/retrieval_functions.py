from pyvis.network import Network
import numpy as np
from util import get_embedding,get_retrieval_embedding, get_point_uuid, lemmatize_entity, entity_extraction
from llama_index.core.query_engine import KnowledgeGraphQueryEngine
from llama_index.core import StorageContext
from datasets import load_dataset
from database_proxy import Neo4jGraphStore, VectorDBClient
from knowledge_graph import KnowledgeGraph
from llama_index.llms.openai import OpenAI
from heapq import heappush, heappop 
import re 
import networkx as nx
import random 

class Beam_Search_Retrieval:
    def __init__(self, graph_client:Neo4jGraphStore, vector_client:VectorDBClient, no_of_hops = 5):
        self.depth = no_of_hops
        self.threshold_search_docs = set()
        self.document_score = {} # document_id: similarity_score
        self.graph_client = graph_client
        self.vector_client = vector_client
        self.kg = KnowledgeGraph(self.graph_client, self.vector_client)
        self.G = nx.Graph() # networkx graph for visualization 
        self.node_place_holder = ""
        self.colors = ["royalblue", "steelblue", "hotpink", 
                       "darkorchid", "yellowgreen", "olive",
                       "goldenrod", "orange", "indianred"]

    def calculate_embedding_similarity(self, emb1, emb2):
        if isinstance(emb1, list):
            emb1 = np.array(emb1)
        if isinstance(emb2, list):
            emb2 = np.array(emb2)
        res = emb1 @ emb2.T
        try:
            return res[0]
        except:
            return res 

    def calculate_cosine_similarity(self, emb1, emb2):
        if isinstance(emb1, list):
            emb1 = np.array(emb1)
        if isinstance(emb2, list):
            emb2 = np.array(emb2)
        similarity = np.dot(emb1, emb2)/(np.linalg.norm(emb1)*np.linalg.norm(emb2))
        return similarity

    ''' def threshold_search(self, node1, emb1, threshold = 0.7, depth = 0):
            if depth == self.depth:
                return

            connections = self.graph_client.get_incoming_outgoing(node1)
            #emb1 = node1['properties']['embedding']
            path = {}
            for i, nodes in enumerate(connections):
                n1, e, n2 = nodes
                node1_connections = e    #connections[i][1]

                #print(node1_connections)
                rel_id = node1_connections.element_id
                emb2 = node1_connections['embedding']
                score = self.calculate_embedding_similarity(emb1, emb2)
                path[]
                #print(score)
                if score >= threshold:
                    if connections[i][2]['id'] not in self.threshold_search_nodes:
                        doc_id = node1_connections['document_id']
                        self.threshold_search_nodes.add(connections[i][2]['id'])
                        self.threshold_search_reln.add(rel_id)
                        self.threshold_search_docs.add(doc_id)
                        print(connections[i][2]['id'], doc_id)
                        self.threshold_search(connections[i][2]['id'], emb1, threshold, depth = depth+1)

    def threshold_search(self, node1, emb1, advanced, beam_width):
            
            # assume there exist a graph, with at least 1 triplet
            # ignore path with the same score for now
            path = {} # stores (edge_1, edge_2...):[sum_embedding, [document_1,document_2...], searched_path((node1, tag1),connection_type, (node2, tag2))]
            connections = self.graph_client.get_incoming_outgoing(node1)
            if connections:
                print(node1)
            #print(connections)
            scores = []
            # initialization with all neighbor path 
            for n1, e, n2 in connections:
                #print(e.keys())
                n1_id = n1['id']
                n2_id = n2['id']
                n1_tag = n1['tag']
                n2_tag = n2['tag']

                e_id = e.element_id
                doc_id = e['document_id']

                if advanced:
                    embedding = np.array(e['retrieval_embedding'])
                else:
                    embedding = np.array(e['embedding'])
                self.document_score[doc_id] = self.calculate_embedding_similarity(embedding, emb1)


    '''

    def threshold_search(self, node1, emb1, advanced, beam_width):
        # assume there exist a graph, with at least 1 triplet
        # ignore path with the same score for now
        path = {} # stores (edge_1, edge_2...):[sum_embedding, [document_1,document_2...], searched_path((node1, tag1),connection_type, (node2, tag2))]
        connections = self.graph_client.get_incoming_outgoing(node1)
    
        #print(connections)
        scores = []
        # initialization with all neighbor path 
        for n1, e, n2 in connections:
            #print(e.keys())
            n1_id = n1['id']
            n2_id = n2['id']
            n1_tag = n1['tag']
            n2_tag = n2['tag']

            e_id = e.element_id
            doc_id = e['document_id']

            if advanced:
                embedding = np.array(e['retrieval_embedding'])
            else:
                embedding = np.array(e['embedding'])
  

            self.document_score[doc_id] = self.calculate_embedding_similarity(embedding, emb1)
            #print(self.document_score[doc_id])
            #print(scores)
            path[tuple([e_id])] = [embedding, [doc_id],[((n1_id, n1_tag), e.type,(n2_id, n2_tag))]]
            heappush(scores, [self.calculate_embedding_similarity(embedding, emb1), tuple([e_id])])
        # limit the beam width
        #print('----------',scores, self.depth)
        while scores:
            #print('yo')
            _, p = heappop(scores)
            if len(scores) + 1 > beam_width:
                del path[p]
        # beam search
        #print(path)
        #print('----------',len(path))
        for depth in range(self.depth - 1):
            new_path = {}
            for prev_path, (prev_embedding, prev_document, prev_graph) in path.items():
                #print(prev_path)
                connections = self.graph_client.get_incoming_outgoing(prev_graph[-1][-1][0])
                prev_path = list(prev_path)
                for n1, e, n2 in connections:
                    n1_id = n1['id']
                    n2_id = n2['id']
                    n1_tag = n1['tag']
                    n2_tag = n2['tag']

                    e_id = e.element_id
                    if advanced:
                        embedding = np.array(e['retrieval_embedding'])
                    else:
                        embedding = np.array(e['embedding'])
                    doc_id = e['document_id']
                    '''if e_id in prev_path:
                        new_path[tuple(prev_path)] = path[tuple(prev_path)] #[new_embedding, prev_document + [doc_id], prev_graph + new_graph]
                        continue '''
                    self.document_score[doc_id] = self.calculate_embedding_similarity(embedding, emb1)
                    #print(self.document_score[doc_id])

                    new_embedding = prev_embedding + embedding
                    new_graph = [((n1_id, n1_tag), e.type,(n2_id, n2_tag))]
                    new_path[tuple(prev_path + [e_id])] = [new_embedding, prev_document + [doc_id], prev_graph + new_graph]
                    #print('--------------######', new_path.keys())
                    avg_embedding = self.calculate_embedding_similarity(emb1, new_embedding) #new_embedding / (1 + len(prev_path))
                    #print(avg_embedding)
                    heappush(scores, [avg_embedding, tuple(prev_path + [e_id])])

            if new_path:
                path = new_path 
            # restrict beam width by cosine sim 
            
            #print(scores)

            while scores:
                _, p = heappop(scores)
                if len(scores) + 1 > beam_width:
                    del path[p]
            
            
        #print(path)
        # find only the path with highest score 
        # while len(scores) > 1:
        #     _, p = heappop(scores)
        #     del path[p]
        # now path should 
        replace = False
        if beam_width > len(self.colors):
            replace = True
        colors = list(np.random.choice(self.colors, beam_width, replace=replace))
        # record the search graph, and searched documents 
        for prev_path, (prev_embedding, prev_document, prev_graph) in path.items():
            color = "steelblue"
            for doc_id in set(prev_document):
                self.threshold_search_docs.add((doc_id, self.document_score[doc_id]))
            for n1, e, n2 in prev_graph:
                self.G.add_node(n1[0], tag=n1[1], color=color)
                self.G.add_edge(n1[0], n2[0], label=e, weight=5)
                self.G.add_node(n2[0], tag=n2[1], color=color)
                # randomly add some node and edge to the existing graph 
                if random.uniform(0,1) < 0.5:
                    self.G.add_node(self.node_place_holder, color="lightgray")
                    self.G.add_edge(self.node_place_holder, n2[0], color="lightgray")
                    self.node_place_holder += " "
            if random.uniform(0,1) < 2: # add 2 nodes to the end 
                self.G.add_node(self.node_place_holder, color="lightgray")
                self.G.add_edge(self.node_place_holder, n2[0], color="lightgray")
                self.node_place_holder += " "
                self.G.add_node(self.node_place_holder, color="lightgray")
                self.G.add_edge(self.node_place_holder, n2[0], color="lightgray")
                self.node_place_holder += " "
            if random.uniform(0,1) < 0.5 and len(self.node_place_holder) >= 2: # add one more hop to the end 
                self.G.add_node(self.node_place_holder, color="lightgray")
                self.G.add_edge(self.node_place_holder, self.node_place_holder[:-1], color="lightgray")
                self.node_place_holder += " "
            

    def __search__(self, question, limit = 1, beam_width = 3, advanced = False):
        #print(advanced)
        if advanced:
            qs_emb = get_retrieval_embedding(question, type_ = "query")
        else:
            qs_emb = get_embedding(question)
            print('corr')

        entity_list = entity_extraction(question)
        fin_entities = lemmatize_entity(entity_list)
        print('Entities extracted from question', fin_entities)
        for e in fin_entities:
            self.threshold_search(e, qs_emb, advanced, beam_width)
            # try:
            #     self.threshold_search(e, qs_emb, beam_width = 3)
            # except:
            #     print(f'Entity {e} not found')

        sorted_docs = sorted(list(self.threshold_search_docs), key = lambda x:x[1], reverse = True)[:limit]
        return set([x[0] for x in sorted_docs]) 
    
    
def vector_retrieval(vector_client, question, limit = 4):
    qs_emb = get_embedding(question)
    docs = vector_client.search('Documents', qs_emb, None, limit = limit)
    res = []
    for i in range(len(docs)):
        res.append((docs[i].payload["document"], docs[i].payload["source"], docs[i].id))
    return res

def retrieval(graph_client, vector_client, question, depth, beam_width = 3, limit = 5, retrieval_method="beam_search", use_summary=False):
    '''
    question:str,           the question as query the graph or vectorDB
    depth:int,              the maximum number of hops to search on the graph 
    beam_width:int ,        the width of beamsearch
    limit:int,              the final number of chunks to return, 
                            if the retrieved chunks is larger, then sort by similarity score, and return topK
    retrieval_method:str,   choose between "beam_search" and "vector_search"
    use_summary: bool       False if using original text, True if use the summary of the text
    '''
    if retrieval_method == 'beam_search':
        bs = Beam_Search_Retrieval(graph_client, vector_client, depth)
        chunk_id = bs.__search__(question, beam_width=beam_width, limit=limit)
        docs = vector_client.get_chunk_from_chunk_id(chunk_id)
        if not docs:
            print("No chunk found")
            return None, None, None  
        # save the graph 
        net = Network(notebook=True, cdn_resources="local", directed=False)
        net.from_nx(bs.G)
        question = question.replace(" ", "_")
        net.show(f'./path.html')
        chunk_text, chunk_source, chunk_id = zip(*docs)
    elif retrieval_method == 'beam_search_advanced':
        bs = Beam_Search_Retrieval(graph_client, vector_client, depth)
        chunk_id = bs.__search__(question, beam_width=beam_width, limit=limit, advanced=True)
        docs = vector_client.get_chunk_from_chunk_id(chunk_id)
        if not docs:
            print("No chunk found")
            return None, None, None  
        # save the graph 
        net = Network(notebook=True, cdn_resources="local", directed=False)
        net.from_nx(bs.G)
        question = question.replace(" ", "_")
        net.show(f'./path_advanced.html')
        chunk_text, chunk_source, chunk_id = zip(*docs)
    elif retrieval_method == 'vector_search':
        docs = vector_retrieval(vector_client, question, limit)
        chunk_text, chunk_source, chunk_id = zip(*docs)
    
    return chunk_text, chunk_source, chunk_id