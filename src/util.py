import uuid
import hashlib
import re, os
from openai import OpenAI
from dotenv import load_dotenv
import spacy
from sentence_transformers import SentenceTransformer
load_dotenv()

OPENAI_API_KEY =  os.getenv("OPENAI_API_KEY")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

openai_client = OpenAI(api_key = OPENAI_API_KEY)
together_client = OpenAI(api_key = TOGETHER_API_KEY,
                base_url= "https://api.together.xyz/v1")


OPENAI_MODEL = set(["gpt-3.5-turbo-0125","gpt-4" ])

INSTRUCTIONS = {
    "qa": {
        "query": "Represent this query for retrieving relevant documents: ",
        "key": "Represent this document for retrieval: ",
    }
}

spacy_model = spacy.load("en_core_web_md")
retrieval_model = SentenceTransformer('BAAI/llm-embedder', device="cpu")


def run_query(query, sys_msg, model):
    client = openai_client if model in OPENAI_MODEL else together_client
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": query}
            ],
        temperature=0.3
    )
    #print(response.choices[0].message.content)
    return response.choices[0].message.content

def get_embedding(text, model="text-embedding-3-small"):
    result = openai_client.embeddings.create(input = [text], model=model).data[0].embedding
    return result

def get_retrieval_embedding(text, type_ = "keys"):
    instruction = INSTRUCTIONS["qa"]
    if type_=="keys":
      keys = [instruction["key"] + text]
      embeddings = retrieval_model.encode(keys)
    else:
      keys = [instruction["query"] + text]
      embeddings = retrieval_model.encode(keys)
    return embeddings


def get_point_uuid(prompt_id: str):
  # first check if the prompt_id is a valid UUID
  pattern = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)
  if pattern.match(prompt_id):
    return prompt_id
  else:
    hashed = hashlib.sha256(prompt_id.encode()).digest()
    return str(uuid.UUID(bytes=hashed[:16]))


def entity_extraction(question):
    # doc = spacy_model(question)
    words = question.split()  # Split the question into words
    entity_list = []
    # Filter out stopwords
    for word in words:
        # Check if the word is a stopword in the SpaCy model
        if not spacy_model(word)[0].is_stop:
            entity_list.append(word)
    return entity_list
    # entity_list = question.split()
    # return entity_list
      
def lemmatize_entity(entities):
    lem_list = set()
    for e in entities:
      doc = spacy_model(e)
      lemm = ' '.join([token.lemma_ for token in doc])
      lemm = lemm.replace("'s", "").strip()
      lem_list.add(lemm.lower())
    return list(lem_list)


  

