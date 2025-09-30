"""Neo4j graph store index."""
import logging
from typing import Any, Dict, List, Optional, Tuple


logger = logging.getLogger(__name__)

node_properties_query = """
CALL apoc.meta.data()
YIELD label, other, elementType, type, property
WHERE NOT type = "RELATIONSHIP" AND elementType = "node"
WITH label AS nodeLabels, collect({property:property, type:type}) AS properties
RETURN {labels: nodeLabels, properties: properties} AS output

"""

rel_properties_query = """
CALL apoc.meta.data()
YIELD label, other, elementType, type, property
WHERE NOT type = "RELATIONSHIP" AND elementType = "relationship"
WITH label AS nodeLabels, collect({property:property, type:type}) AS properties
RETURN {type: nodeLabels, properties: properties} AS output
"""

rel_query = """
CALL apoc.meta.data()
YIELD label, other, elementType, type, property
WHERE type = "RELATIONSHIP" AND elementType = "node"
UNWIND other AS other_node
RETURN {start: label, type: property, end: toString(other_node)} AS output
"""


class Neo4jGraphStore:
    def __init__(
        self,
        username: str,
        password: str,
        url: str,
        database: str = "neo4j",
        node_label: str = "Entity",
        **kwargs: Any,
    ) -> None:
        try:
            import neo4j
        except ImportError:
            raise ImportError("Please install neo4j: pip install neo4j")
        self.node_label = node_label
        self._driver = neo4j.GraphDatabase.driver(url, auth=(username, password))
        self._database = database
        self.schema = ""
        self.structured_schema: Dict[str, Any] = {}
        # Verify connection
        try:
            self._driver.verify_connectivity()
        except neo4j.exceptions.ServiceUnavailable:
            raise ValueError(
                "Could not connect to Neo4j database. "
                "Please ensure that the url is correct"
            )
        except neo4j.exceptions.AuthError:
            raise ValueError(
                "Could not connect to Neo4j database. "
                "Please ensure that the username and password are correct"
            )
        # Set schema
        try:
            self.refresh_schema()
        except neo4j.exceptions.ClientError:
            raise ValueError(
                "Could not use APOC procedures. "
                "Please ensure the APOC plugin is installed in Neo4j and that "
                "'apoc.meta.data()' is allowed in Neo4j configuration "
            )
        # Create constraint for faster insert and retrieval
        try:  # Using Neo4j 5
            self.query(
                """
                CREATE CONSTRAINT IF NOT EXISTS FOR (n:%s) REQUIRE n.id IS UNIQUE;
                """
                % (self.node_label)
            )
        except Exception:  # Using Neo4j <5
            self.query(
                """
                CREATE CONSTRAINT IF NOT EXISTS ON (n:%s) ASSERT n.id IS UNIQUE;
                """
                % (self.node_label)
            )

    @property
    def client(self) -> Any:
        return self._driver

    def get(self, subj: str) -> List[List[str]]:
        """Get triplets."""
        query = """
            MATCH (n1:%s)-[r]->(n2:%s)
            WHERE n1.id = $subj
            RETURN n1, r, n2;
        """

        prepared_statement = query % (self.node_label, self.node_label)

        with self._driver.session(database=self._database) as session:
            data = session.run(prepared_statement, {"subj": subj})
            return [record.values() for record in data]

    def get_incoming_outgoing(self, subj: str) -> List[List[str]]:
        """Get triplets."""
        query = """
            MATCH (n1:%s)-[r]-(n2:%s)
            WHERE n1.id = $subj
            RETURN n1, r, n2;
        """

        prepared_statement = query % (self.node_label, self.node_label)

        with self._driver.session(database=self._database) as session:
            data = session.run(prepared_statement, {"subj": subj})
            #print([record.values() for record in data])
            return [record.values() for record in data]


    def get_rel_map(
        self, subjs: Optional[List[str]] = None, depth: int = 2, limit: int = 30
    ) -> Dict[str, List[List[str]]]:
        """Get flat rel map."""
        # The flat means for multi-hop relation path, we could get
        # knowledge like: subj -> rel -> obj -> rel -> obj -> rel -> obj.
        # This type of knowledge is useful for some tasks.
        # +-------------+------------------------------------+
        # | subj        | flattened_rels                     |
        # +-------------+------------------------------------+
        # | "player101" | [95, "player125", 2002, "team204"] |
        # | "player100" | [1997, "team204"]                  |
        # ...
        # +-------------+------------------------------------+

        rel_map: Dict[Any, List[Any]] = {}
        if subjs is None or len(subjs) == 0:
            # unlike simple graph_store, we don't do get_all here
            return rel_map

        query = (
            f"""MATCH p=(n1:{self.node_label})-[*1..{depth}]->() """
            f"""{"WHERE n1.id IN $subjs" if subjs else ""} """
            "UNWIND relationships(p) AS rel "
            "WITH n1.id AS subj, p, apoc.coll.flatten(apoc.coll.toSet("
            "collect([type(rel), endNode(rel).id]))) AS flattened_rels "
            f"RETURN subj, collect(flattened_rels) AS flattened_rels LIMIT {limit}"
        )

        data = list(self.query(query, {"subjs": subjs}))
        if not data:
            return rel_map

        for record in data:
            rel_map[record["subj"]] = record["flattened_rels"]
        return rel_map

    def upsert_triplet(self, subj: str, rel: str, obj: str, meta_data=None) -> None:
        # metadata contains (sub_tag, embedding, obj_tag)
        """Add triplet."""
        query = """
            MERGE (n1:`%s` {id:$subj})
            MERGE (n2:`%s` {id:$obj})
            MERGE (n1)-[r:`%s`]->(n2)
        """
        prepared_statement = query % (
            self.node_label,
            self.node_label,
            rel.replace(" ", "_").upper(),
        )
        payload = {"subj": subj, "obj": obj}
        if meta_data:
            set_query = """
                SET n1.tag = $tag1
                SET r.embedding = $embedding
                SET r.retrieval_embedding = $retrieval_embedding
                SET r.document_id = $document_id
                SET n2.tag = $tag2
            """
            prepared_statement += set_query
            res = {"tag1": meta_data[0], "embedding":meta_data[1], "retrieval_embedding":meta_data[2], "document_id": meta_data[3], "tag2":meta_data[4]}
            payload.update(res) #dict(payload.items() | res.items())
        with self._driver.session(database=self._database) as session:
            session.run(prepared_statement, payload)

    def execute_query(self, query, parameters=None):
        with self._driver.session(database=self._database) as session:
            result = session.run(query, parameters)
            return [record.values() for record in result]


    def delete(self, subj: str, rel: str, obj: str) -> None:
        """Delete triplet."""

        def delete_rel(subj: str, obj: str, rel: str) -> None:
            with self._driver.session(database=self._database) as session:
                session.run(
                    (
                        "MATCH (n1:{})-[r:{}]->(n2:{}) WHERE n1.id = $subj AND n2.id"
                        " = $obj DELETE r"
                    ).format(self.node_label, rel, self.node_label),
                    {"subj": subj, "obj": obj},
                )

        def delete_entity(entity: str) -> None:
            with self._driver.session(database=self._database) as session:
                session.run(
                    "MATCH (n:%s) WHERE n.id = $entity DELETE n" % self.node_label,
                    {"entity": entity},
                )

        def check_edges(entity: str) -> bool:
            with self._driver.session(database=self._database) as session:
                is_exists_result = session.run(
                    "MATCH (n1:%s)--() WHERE n1.id = $entity RETURN count(*)"
                    % (self.node_label),
                    {"entity": entity},
                )
                return bool(list(is_exists_result))

        delete_rel(subj, obj, rel)
        if not check_edges(subj):
            delete_entity(subj)
        if not check_edges(obj):
            delete_entity(obj)

    def refresh_schema(self) -> None:
        """
        Refreshes the Neo4j graph schema information.
        """
        node_properties = [el["output"] for el in self.query(node_properties_query)]
        rel_properties = [el["output"] for el in self.query(rel_properties_query)]
        relationships = [el["output"] for el in self.query(rel_query)]

        self.structured_schema = {
            "node_props": {el["labels"]: el["properties"] for el in node_properties},
            "rel_props": {el["type"]: el["properties"] for el in rel_properties},
            "relationships": relationships,
        }

        # Format node properties
        formatted_node_props = []
        for el in node_properties:
            props_str = ", ".join(
                [f"{prop['property']}: {prop['type']}" for prop in el["properties"]]
            )
            formatted_node_props.append(f"{el['labels']} {{{props_str}}}")

        # Format relationship properties
        formatted_rel_props = []
        for el in rel_properties:
            props_str = ", ".join(
                [f"{prop['property']}: {prop['type']}" for prop in el["properties"]]
            )
            formatted_rel_props.append(f"{el['type']} {{{props_str}}}")

        # Format relationships
        formatted_rels = [
            f"(:{el['start']})-[:{el['type']}]->(:{el['end']})" for el in relationships
        ]

        self.schema = "\n".join(
            [
                "Node properties are the following:",
                ",".join(formatted_node_props),
                "Relationship properties are the following:",
                ",".join(formatted_rel_props),
                "The relationships are the following:",
                ",".join(formatted_rels),
            ]
        )

    def get_schema(self, refresh: bool = False) -> str:
        """Get the schema of the Neo4jGraph store."""
        if self.schema and not refresh:
            return self.schema
        self.refresh_schema()
        logger.debug(f"get_schema() schema:\n{self.schema}")
        return self.schema

    def query(self, query: str, param_map: Optional[Dict[str, Any]] = {}) -> Any:
        with self._driver.session(database=self._database) as session:
            result = session.run(query, param_map)
            return [d.data() for d in result]

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from qdrant_client.http.models import Filter, FieldCondition, MatchValue,HasIdCondition
from util import get_point_uuid

DOCUMENT_COLLECTION = "Documents"
class VectorDBClient:
    def __init__(self, url, port = 6333) -> None:
        self.url = url 
        self.port= port
        self.embedding_dim = 1536
        self.dummy_vector = [0] * self.embedding_dim
    
    def create_collection(self, name):
        client = QdrantClient(self.url, port=self.port)
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=self.embedding_dim, distance=Distance.DOT),
        )
        client.close()
        
    def delete_collection(self, name):
        client = QdrantClient(self.url, port=self.port)
        client.delete_collection(collection_name=name)
        client.close()

    def get_payload(self, document_id, collection=DOCUMENT_COLLECTION): #-> Tuple[dict,list[float]]:
        point_id = get_point_uuid(document_id)
        client = QdrantClient(self.url, port=self.port)
        search_result = client.search(
            collection_name=collection,
            query_vector=self.dummy_vector,
            query_filter=Filter(
                must=[HasIdCondition(has_id=[point_id])]
            ),
            with_payload=True,
            with_vectors=True,
            limit=1,
        )
        client.close()
        if search_result == []:
            return None, None
        else:
            return search_result[0].payload, search_result[0].vector
        
    def insert_payload(self, collection_name, document_id, payload, embedding):
        """
        insert a prompt to a collection
        """
        
        client = QdrantClient(self.url, port=self.port)
        document_id = get_point_uuid(document_id)
        operation_info = client.upsert(
            collection_name=collection_name,
            wait=True,
            points=[
                PointStruct(id=document_id, vector=embedding, payload=payload),
            ],
        )
        client.close()
        return document_id


    def search(
            self, 
            collection_name, 
            query_vector, 
            query_filter, 
            limit = 1):
        
        client = QdrantClient(self.url, port=self.port)
        result = client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            query_filter=query_filter,
            with_payload=True,
            limit=limit,
        )
        client.close()

        return result
    
    def get_document_from_embedding(self, embeddings):#->list[Tuple[str,str]]:
        '''
        input document embeddings
        return [(document, source)]
        '''
        result = []
        client = QdrantClient(self.url, port=self.port)
        for embedding in embeddings:
            doc = client.search(
                    collection_name=DOCUMENT_COLLECTION,
                    query_vector=embedding,
                    with_payload=True,
                    limit=1,
                ).payload
            result.append((doc["document"], doc["source"]))
        client.close()
        return result 
    

    def get_chunk_from_chunk_id(self, chunk_ids):#->list[Tuple[str,str, str]]:
        '''
        input a list of document ids
        return [(chunk_text, chunk_source, chunk_id)]
        '''
        result = []
        client = QdrantClient(self.url, port=self.port)
        for chunk_id in chunk_ids:
            res = client.search(
                    collection_name=DOCUMENT_COLLECTION,
                    query_vector=self.dummy_vector,
                    query_filter=Filter(
                        must=[HasIdCondition(has_id=[chunk_id])]
                    ),
                    with_payload=True,
                    limit=1,
                )
            if res: # if document id exists
                result.append((res[0].payload["document"], res[0].payload["source"], chunk_id))
        return result 

        

# connect to database 
username = "neo4j"
password = "testgraph"
url = "bolt://localhost:7687"
database = "neo4j"

graph_client = Neo4jGraphStore(username=username, password=password,url=url,database=database)
vector_client = VectorDBClient("localhost")

