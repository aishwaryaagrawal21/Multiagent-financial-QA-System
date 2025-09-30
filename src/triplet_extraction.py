from typing import List, Tuple
from util import run_query
from prompt import ENTITY_EXTRACTION_PROMPT,ENTITY_EXTRACTION_SYSTEM_PROMPT,\
    RELATION_EXTRACTION_PROMPT,RELATION_EXTRACTION_SYSTEM_PROMPT,\
    VERIFICATION_PROMPT,VERIFICATION_SYSTEM_PROMPT

ENTITY_SCHEMA = ["ORG", "PERSON", "GPE", "PRODUCT", "EVENT", "SECTOR", "ECON_INDICATOR", "COMMODITIES"]
model = "gpt-3.5-turbo-0125"
model2="gpt-4"
def parse_entity(text):
    lines = text.split('\n')
    entity_dict = {}
    for line in lines:
        # Skip empty lines
        splitted = line.split(':')
        if len(splitted) <= 1:
            continue
        entity_type, values, *_ = line.split(':')
        values = values.strip()
        if not values:
            continue 
        # Add the entity type to the dictionary if it doesn't exist
        if entity_type in ENTITY_SCHEMA:
            if entity_type not in entity_dict:
                entity_dict[entity_type] = []
            # Add the values to the list for the entity type
            entity_dict[entity_type].extend(values.split(', '))

    return entity_dict


def parse_triplet_response(
    response: str, max_length: int = 128
) -> List[Tuple[str, str, str]]:
    knowledge_strs = response.strip().split("\n")
    results = []
    for text in knowledge_strs:
        if "(" not in text or ")" not in text or text.index(")") < text.index("("):
            # skip empty lines and non-triplets
            continue
        triplet_part = text[text.index("(") + 1 : text.rindex(")")] 
        tokens = triplet_part.split(",")
        if len(tokens) != 3:
            continue

        if any(len(s.encode("utf-8")) > max_length for s in tokens):
            # We count byte-length instead of len() for UTF-8 chars,
            # will skip if any of the tokens are too long.
            # This is normally due to a poorly formatted triplet
            # extraction, in more serious KG building cases
            # we'll need NLP models to better extract triplets.
            continue

        subj, pred, obj = map(str.strip, tokens)
        if not subj or not pred or not obj:
            # skip partial triplets
            continue
        results.append((subj, pred, obj))
    return results


def extract_triplet(text) -> Tuple[Tuple[str, str, str], dict]:
    query = ENTITY_EXTRACTION_PROMPT.format(text=text)
    result = run_query(query, ENTITY_EXTRACTION_SYSTEM_PROMPT, model)
    entities = parse_entity(result) # extract {category: [entity1, entity2]}
    entity_str = "\n".join([f"{k}: {', '.join(v)}" for k,v in entities.items()])
    # extract relation
    query = RELATION_EXTRACTION_PROMPT.format(text=text, entity_str=entity_str)
    result = run_query(query, RELATION_EXTRACTION_SYSTEM_PROMPT, model)
    print('Initial relations', result)
    print('-------------------------------')
    # triplets = parse_triplet_response(result)

    query = VERIFICATION_PROMPT.format(entity_str=entity_str, triplets=result)
    result = run_query(query, VERIFICATION_SYSTEM_PROMPT, model2)
    triplets = parse_triplet_response(result)
    # use entities for labelling later
    entity_label = {}
    for label, entity in entities.items():
        for e in entity:
            entity_label[e] = label 
    return triplets, entity_label

# print(extract_triplet(text))



