SINGLE_AGENT_QA_SYSTEM_PROMPT = 'You are an analyst, that must think step-by-step to answer questions asked'

SINGLE_AGENT_QA_PROMPT = '''\
DOCUMENT:
{}

QUESTION:
{}

INSTRUCTIONS:
Answer the users QUESTION using the DOCUMENT text above.
Keep your answer ground in the facts of the DOCUMENT.
If the DOCUMENT doesn't contain the facts to answer the QUESTION return None'''

ENTITY_EXTRACTION_SYSTEM_PROMPT = """\
Your goal is to extract entities from given text. You have to list down the categories, as well as words extracted, that fall into those strict categories.

ORG: Organizations like government or regulatory bodies or companies (e.g., "Apple", "United States Government")
PERSON: Individuals (e.g., "Elon Musk")
GPE: Geopolitical entities such as countries, cities, etc. (e.g., "Germany")
PRODUCT: Products or services (e.g., "iPhone")
EVENT: Specific and Material Events (e.g., "Olympic Games", "Covid-19")
SECTOR: Company sectors or industries (e.g.,"Technology sector")
ECON_INDICATOR: Economic indicators (e.g.,"Inflation rate"), numerical value like "10%" is not a ECON INDICATOR;
COMMODITIES: Items used to trade in or build product (e.g., "Copper", "Silicon", "Gold")
"""

ENTITY_EXTRACTION_PROMPT = """\
Given below is INPUT_TEXT. Your task is to extract words only from INPUT_TEXT that fall into the categories below. \
List down the categories, as well as words extracted from INPUT_TEXT, that fall into those categories.

ORG: Organizations like government or regulatory bodies or companies (e.g., "Apple", "United States Government")
PERSON: Individuals (e.g., "Elon Musk")
GPE: Geopolitical entities such as countries, cities, etc. (e.g., "Germany")
PRODUCT: Products or services (e.g., "GPU")
EVENT: Specific and Material Events (e.g., "Olympic Games", "Covid-19")
SECTOR: Company sectors or industries (e.g.,"Technology sector")
ECON_INDICATOR: Economic indicators (e.g.,"Inflation rate", "earnings"), numerical value like "10%" is not a ECON INDICATOR;
COMMODITIES: Items used to trade in or build product (e.g., "Copper", "Silicon", "Gold")


Output the entities in the format of "Category: entity1, entity2, entity3..."

ORG: 
PERSON: 
GPE: 
PRODUCT: 
EVENT: 
SECTOR: 
ECON_INDICATOR:
COMMODITIES: 

INPUT_TEXT: 
{text}
"""



RELATION_EXTRACTION_PROMPT = """
Given below are INPUT_TEXT and a set of predefined ENTITIES. Extract high-level relationships between these ENTITIES from the INPUT_TEXT. The triplets should be in the form of (entity1, relation, entity2) where both entity1 and entity2 in each triplet STRICTLY belong to the provided set of ENTITIES. You are not allowed to make any additional entities or modify content inside these entities. Discard any irrelevant information. Do not include any external entities or modifcations, it is ok to discard the information that cannot be captured by just these entities.

INPUT_TEXT:
{text}
ENTITIES:
{entity_str}

###CONSTRAINT: Triplets only have 2 predefined entities and 1 captured high level relation - (entity1,relation,entity2). If you find any entities outside the given set, please discard that triplet.
"""

RELATION_EXTRACTION_SYSTEM_PROMPT = """
Given an input text and a set of entities, extract relations in the form of (entity, relation, entity). Both entities in a triplet STRICTLY belong to the provided set of ENTITIES. You are not allowed to make any additional entities or modify content inside these entities. You cannot deviate from this. If you are ###UNSURE about any triplet, discard it. Any 2 entities can only have zero or one relation, keep only the most effective one. Triplets can not be UNIQUE.

"""

VERIFICATION_PROMPT = """
Given below is a set of predefined ENTITIES and extracted TRIPLETS in the form of (entity1, relation, entity2). Verify that the TRIPLETS contain only the predefined entities, especially the ending entity which might have gotten rephrased. Remove any triplets that have irrelevant information or do not strictly contain the predefined entities format. Keep the quality of graph high by removing any redundant or irrelevant triplet. Also 2 entities can only have upto one relation, keep only the most effective one.
Return the final corrected triplets. Do not return anything else.

###STRICT_CONSTRAINT: Triplets (entity1, relation, entity2) can only contain provided entities. Remove any triplet that does not follow this constraint. Or rectify the entity to be only from the ENTITY_LIST. Keep most effective unique triplets.

ENTITIES_LIST:
{entity_str}

TRIPLETS:
{triplets}

Given below is an example keeping only the most effective unique triplets:
Text: Amazon, founded by Jeff Bezos, recently acquired Zoox, a self-driving car startup, to expand its delivery network. This acquisition aims to enhance Amazon's logistics capabilities and compete with other tech giants in autonomous vehicle technology.
Entities: Amazon, Jeff Bezos, Zoox, autonomous vehicle
Triplets:
(Amazon, founded by, Jeff Bezos)
(Amazon, acquired, Zoox)
(Zoox, associated with, autonomous vehicle)
"""

VERIFICATION_SYSTEM_PROMPT = """
Given below is a set of predefined ENTITIES and extracted TRIPLETS in the form of (entity1, relation, entity2). Verify that triplets strictly contain only given entities and high level relations. Both entities in a triplet STRICTLY belong to the provided set of ENTITIES. You are not allowed to deviate from this. Make sure you return only the triplets that follow this rule strictly.

"""
