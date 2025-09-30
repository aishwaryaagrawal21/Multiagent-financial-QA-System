# !pip install -q -r requirements.txt
# from llama_index.prompts.base import PromptTemplate
# from llama_index.prompts.prompt_type import PromptType
#
# TEXT_KG_TRIPLET_EXTRACT_TMPL = """\
# Some text is provided below. \
# Given the text section, extract up to \
# {max_knowledge_triplets} \
# knowledge triplets in the form of (subject, predicate, object). Avoid stopwords.\
# ---------------------
# Example 1:
# Text: Philz is a coffee shop founded in Berkeley in 1982.
# Entities: Philz, Berkeley, 1982
# Triplets:
# (Philz, founded in, Berkeley)
# (Philz, founded in, 1982)
#
# Example 2:
# Text: Amazon, founded by Jeff Bezos, recently acquired Zoox, a self-driving car startup, to expand its delivery network. This acquisition aims to enhance Amazon's logistics capabilities and compete with other tech giants in autonomous vehicle technology.
# Entities: Amazon, Jeff Bezos, Zoox, autonomous vehicle
# Triplets:
# (Amazon, founded by, Jeff Bezos)
# (Amazon, acquired, Zoox)
# (Zoox, associated with, autonomous vehicle)
# ---------------------\n
# Text: {text}\n
# Triplets:\n
# """
# TEXT_KG_TRIPLET_EXTRACT_PROMPT = PromptTemplate(
#     TEXT_KG_TRIPLET_EXTRACT_TMPL, prompt_type=PromptType.KNOWLEDGE_TRIPLET_EXTRACT
# )
#
# TABLE_KG_TRIPLET_EXTRACT_TMPL = """
# A text section that contains a table is provided below. \
# Given the text section, extract up to 100\
# knowledge triplets in the form of (subject, value, column).
# ---------------------
# Example:
# Text: \
# Name\tAge
# Alice\t45
# Bob\t24
# Leo\t23
# Triplets:
# (Alice, 45, Age)
# (Bob, 24, Age)
# (Leo, 23, Age)
# Text: \
# Menu Item\tFlavor\tPrice
# Hot coco\tsweet\t$4.45
# Caramel Tea\tBitter\t$5.99
# Triplets:
# (Hot coco, sweet, Flavor)
# (Hot coco, $4.55, Price)
# (Caramel Tea, Bitter, Flavor)
# (Caramel Tea, $5.99, Price)
# ---------------------\n
# Text: {text}\n
# Triplets:\n
# """
# TABLE_KG_TRIPLET_EXTRACT_PROMPT = PromptTemplate(
#     TABLE_KG_TRIPLET_EXTRACT_TMPL, prompt_type=PromptType.KNOWLEDGE_TRIPLET_EXTRACT
# )

VERIFICATION_PROMPT = """
Given below is a set of predefined ENTITIES and extracted TRIPLETS in the form of (entity1, relation, entity2). Verify that the TRIPLETS contain only the predefined entities, especially the ending entity which might have gotten rephrased. Verify these based on input text. Remove any triplets that have irrelevant information or do not strictly contain the predefined entities format. Keep the quality of graph high by removing any redundant or irrelevant triplet. Also 2 entities can only have upto one relation, keep only the most effective one.
Return the final corrected triplets. Do not return anything else.

###STRICT_CONSTRAINT: Triplets (entity1, relation, entity2) can only contain provided entities. Remove any triplet that does not follow this constraint. Or rectify the entity to be only from the ENTITY_LIST. Keep most effective unique triplets.

ENTITIES_LIST:
{entity_str}

TRIPLETS:
{triplets}

INPUT_TEXT:
{input_text}
"""

VERIFICATION_SYSTEM_PROMPT = """
Given below is a set of predefined ENTITIES and extracted TRIPLETS in the form of (entity1, relation, entity2) from a given input text. Verify that triplets strictly contain only given entities and high level relations. Both entities in a triplet STRICTLY belong to the provided set of ENTITIES. You are not allowed to deviate from this. Make sure you return only the triplets that follow this rule strictly.

"""

ENTITY_SYSTEM_PASS1 = """\
Your goal is to extract entities from given text. You have to list down the categories, as well as words extracted, that fall into those strict categories.

SECTOR: Company sectors or industries like "Finance", "Technology" 
COMPANY: Name of organizations or regulatory bodies or companies like "Apple"
PEOPLE: Individuals like "Elon Musk"
PRODUCTS: Products that are mentioned such as "iPhone"

"""

ENTITY_PASS1 = """\
Given below is INPUT_TEXT. Your task is to extract words only from INPUT_TEXT that fall into the categories below. \
List down the categories, as well as words extracted from INPUT_TEXT, that fall into those categories.

SECTOR: Company sectors or industries like "Finance", "Technology"  
COMPANY: Name of Organizations or regulatory bodies or companies like "Apple"
PEOPLE: Individuals like "Elon Musk"
PRODUCTS: Products that are mentioned such as "iPhone"

Note that SECTORS can have following values {sectors}

Output only entities in the format of "Category: entity1, entity2, entity3...". DO NOT return any explanations or notes.
SECTOR: 
COMPANY: 
PEOPLE: 
PRODUCTS: 

INPUT_TEXT: 
{text}
"""

RELATION_PROMPT = """
Given below are 2 predefined lists in the form of Category1: entities and Category2: entities. Also provided below is a set of POSSIBLE_RELATIONS. Check if any relationship from the list of POSSIBLE_RELATIONS exists between any entity in category1 to entity Category2, taking context from INPUT_TEXT. Return the triplets of (entity1, relation, entity2) based on below rules-
1. Relationship should connect category1 {cat1} to category 2 {cat2}, not within a category. Example: Category 1 - SECTOR: Technology, Category 2 - COMPANY: Nvidia, AMD
Triplets - (Technology, Primary Manufacturer, Nvidia), (Technology, Primary Manufacturer, AMD)
2. For every entity in category1 {cat1}, check if any predefined relation connects it to any entity in category2 {cat2}. 
3. Any 2 entities can only be connected by upto one unique relation strictly belonging to POSSIBLE_RELATIONS. There need not be any relation between 2 entities, skip them.
4. Triplets should be unique. If you have already found a relationship between any 2 entities, do not check for it again. Use the most suitable relation and move to next.
Example: 
Category 1 - COMPANY: Meta, Nvidia, PNC
Category 2 - PRODUCTS: smart glasses, AR glasses, AI chips
POSSIBLE_RELATIONS - ['Manufacturer', 'Seller/Retailer', 'Owner', 'Designer/Developer', 'Consumer', 'Distributor', 'Support Provider']
Answer - 
(Meta, Designer/Developer, smart glasses)
(Meta, Designer/Developer, AR glasses)
(Meta, Consumer, AI chips)
(Nvidia, Manufacturer, AI chips)
Meta is connected to all category2 entities, Nvidia is connected to 1, PNC is not connected to any. None of the triplets is repeating. Any 2 entities in a triplet are only joined by 1 unique relation. All entities between Category1 and Category2 need not be joined by any relation.

Return format: Return only triplets in below format, no explanation needed. Limit output to 50 triplets at max.
(Entity1, Relation, Entity2)
(Entity2, Relation2, Entity3)

CONSTRAINT: Don't keep on generating repeated data. If you find more than 50 triplets from a paragraph, break loop and move to next one

Inputs are as follows-
ENTITIES-
Category 1 - {cat1}: {entity1}
Category 2 - {cat2}: {entity2}

POSSIBLE_RELATIONS (Relations connect category1 to category2, not inter category)
{possible_relations}

INPUT_TEXT:
{text}
"""

RELATION_SYSTEM_PROMPT = """
Your goal is to extract relations from an inout text between predefined entities and strictly belonging to predefined relations. Return triplets of (entity from category1, relation, entity from category2). Use below rules to return triplets for an input text-
1. Check which predefined relationships exist between entity from category1 and entity from category2. 
2. These relations should STRICTLY be from the predefined relations. The enitities can also be STRICTLY from category1 or category 2 predefined.
3. You are not allowed to make any additional entities or include external relationships. 
4. Any 2 entities can only have no or one relation, keeping only the most effective one. Triplets are unique such that they cannot be repeated.

"""

VERIFICATION_PROMPT1 = """
Verify the following relationships based on the predefined entities and relations using below rules. Verify them based on context from input text.
1. Triplets should strictly contain both entities and relationships that are present in the predefined entities and relations. 
2. Relations are between entities from 2 categories, not within a category.
3. Any 2 entities must be connected by only 1 unique relation. If there are multiple relations, keep only the most relevant one. 
4. Triplets should make sense based on given input context or on a general level based on common knowledge.
5. If you are ###UNSURE about any triplet based on context text, make decision based on your knowledge as long as it contains predefined entities and relations. 
Return format: 
(Entity1, Relation, Entity2)
(Entity2, Relation2, Entity4)
Only for verified triplets in the given format. Do not return any explanation.

Predefined Entities: {predefined_entities}
Predefined Relations: {predefined_relations}

Triplets to verify:
{triplets}

Input text: {text}
"""


VERIFICATION_SYSTEM_PROMPT1 = """
You are a triplet verifier. You should verify triplets, given predefined entities and relations. Triplets are of the form (entity, relationship, entity). Use the following rules- 
1. Keep triplets where relationships make sense based on context paragraph. You can use some of your own knowledge as well but do not move too away from paragraph.
2. The triplets can only contain entities and relations, that have been predefined. If either enitity or relation in a triplet isn't predefined, discard it.
3. Any 2 entities can only have no or one relationship, keep only the most effective one. Triplets are unique such that they cannot be repeated.
4. If you are ###UNSURE about any triplet based on context text, use your own common knowledge. 
"""