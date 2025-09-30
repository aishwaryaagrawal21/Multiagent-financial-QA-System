
from llama_index.core.node_parser import SentenceSplitter
from triplet_extraction import extract_triplet
from util import get_embedding, get_retrieval_embedding, lemmatize_entity, get_point_uuid
from database_proxy import VectorDBClient, Neo4jGraphStore
import os, json 
import pandas as pd 
import math 
from tqdm import tqdm

class KnowledgeGraph:
    def __init__(self, graph_client:Neo4jGraphStore, vector_client:VectorDBClient, chunk_size = 1024):
        self.graph_client = graph_client
        self.vector_client = vector_client
        self.chunk_size = chunk_size
        try:
            self.vector_client.create_collection("Documents")
        except:
            #print('Could not create vector db')
            pass 

    def process_news(self, csv_folder_path, limit = None):
        result = [] # list of (text, source)
        files = os.listdir(csv_folder_path)
        for f in files:
            path = os.path.join(csv_folder_path, f)
            data = pd.read_csv(path)[["text", "title"]]
            result += data.values.tolist()
        if limit: # for testing purpose 
            result = result[:limit]
        # start processing data 
        processed_count = 0
        for text, source in result:
            processed_count += self.process_text(text, source)
        print(f"processed {processed_count} documents")

    def process_sec(self, sec_file_path, limit = None):
        # path to a json file 
        with open(sec_file_path, "r") as f:
            data = json.load(f)
        result = [] # list of (text, source)
        for company, sec_files in data.items():
            for section, text in sec_files.items():
                result.append((text, f"SEC_2023_{company}_{section}"))
        # start processing data 
        processed_count = 0
        for text, source in result:
            processed_count += self.process_text(text, source)
        print(f"processed {processed_count} documents")

    def process_text(self, text, source):
        
        init_document_id = f"{source}::0" 
        if self.vector_client.get_payload(init_document_id)[0] is not None:
            print(f"{source} exists in database")
            return 0

        # chunk text 
        parser = SentenceSplitter()
        chunks = parser.split_text(text)
        embeddings = [get_embedding(chunk) for chunk in chunks]

        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            document_id = f"{source}::{i}"
            # get id by source::index
            payload = {
                "source":source,
                "document":chunk
            }
            payload_document_id = get_point_uuid(document_id)
            triplets, entities = extract_triplet(chunk)
            processed_count = 0
            for e1, r, e2 in triplets:
                if e1 not in entities or e2 not in entities:
                    continue 
                processed_count += 1
                e1_tag = entities[e1]
                e2_tag = entities[e2]
                # convert e1,e2 to lemma form before inserting into graph 
                e1 = lemmatize_entity([e1])[0]
                e2 = lemmatize_entity([e2])[0]
                self.graph_client.upsert_triplet(e1, r, e2, (e1_tag, embedding, payload_document_id, e2_tag))
            
            if processed_count > 0: # only insert the valid documents
                self.vector_client.insert_payload("Documents", document_id, payload, embedding)
        return 1 if processed_count > 0 else 0 

    def process_csv(self, csv_path):
        '''
        csv contains columns: "Input Texts", "Relationships", "Labels, "Metadatas"
        Input Text: str
        Relationships: (e1,r,e2)\n...
        SectorData: {category: [e1, e2...]}
        Metadata: {key: [author...]}
        '''

        data = pd.read_csv(csv_path)
        for i in tqdm(range(len(data))):
            row = data.iloc[i]
            text = row["InputText"] if "InputText" in data.columns else row["Text"]
            triplets = row["Relationships"]
            # some field contains NaN 
            labels = eval(row["SectorData"])
            if "MetaData" in data.columns:
                metadata = eval(row["MetaData"].replace("NaN", "None"))
                source = metadata.get("title", "")
                # TODO: define chunk_id in metadata
                chunk = 0
                document_id = f"{source}::{chunk}"
            else: 
                company = row["Company"]
                section = row["ItemKey"]
                source = f"{company}_{section}"
                # TODO: put chunk_id inside
                chunk = i
                metadata = {"company": company, "section":section, "chunk":chunk}
                document_id = f"{source}::{chunk}"

            payload_document_id = get_point_uuid(document_id)

            if self.vector_client.get_payload(payload_document_id)[0] is not None:
                print(f"{source} exists in database")
                continue

            if isinstance(triplets, float): # if no valid triplet extracted, return
                continue 
            triplets = [t[1:-1].split(", ")[:3] for t in triplets.split("\n")]
            print(triplets, labels)
            # create entity: category labels
            entities = {}

            for label, entity in labels.items():
                for e in entity:
                    entities[e.lower()] = label 
            # get id by source::index
            payload = metadata | {
                "source":source,
                "document":text
            }
            # process the triples 
            embedding = get_embedding(text)
            retrieval_embedding = get_retrieval_embedding(text, type_="keys")
            #print(retrieval_embedding)
            for e1, r, e2 in triplets:
                if e1 not in entities or e2 not in entities:
                    continue 
                e1_tag = entities[e1]
                e2_tag = entities[e2]
                # convert e1,e2 to lemma form before inserting into graph 
                e1 = lemmatize_entity([e1])[0]
                e2 = lemmatize_entity([e2])[0]
                # print('yea---------')
                self.graph_client.upsert_triplet(e1, r, e2, (e1_tag, embedding, retrieval_embedding[0], payload_document_id, e2_tag))
            self.vector_client.insert_payload("Documents", document_id, payload, embedding)





    def reset_kg(self):
        try:
            self.vector_client.delete_collection("Documents")
            self.vector_client.create_collection("Documents")
        except:
            print("Vector database delete failed")

        delete_query = '''MATCH (n)
        DETACH DELETE n;'''
        with self.graph_client._driver.session(database=self.graph_client._database) as session:
            session.run(
                delete_query
            )


    




