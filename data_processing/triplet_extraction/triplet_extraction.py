import os, sys, json, traceback
from dotenv import load_dotenv
from openai import OpenAI
from typing import List, Tuple
from pathlib import Path
import pandas as pd
from triplet_prompts import ENTITY_PASS1, ENTITY_SYSTEM_PASS1
from triplet_prompts import VERIFICATION_PROMPT1, VERIFICATION_SYSTEM_PROMPT1
from triplet_prompts import RELATION_PROMPT, RELATION_SYSTEM_PROMPT

sys.path.append(str(Path(__file__).parent.parent))
load_dotenv()
# log_file_news = open('Data/console_output_news_tsm.log', 'w')
log_file_sec = open('Data/console_output_sec_NVDA.log', 'w')
original_stdout = sys.stdout

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
openai_client = OpenAI(api_key=OPENAI_API_KEY)
together_client = OpenAI(api_key=TOGETHER_API_KEY,
                         base_url="https://api.together.xyz/v1")

model1 = "gpt-3.5-turbo"
model2 = "gpt-4-turbo-preview"
model3 = "mistralai/Mixtral-8x7B-Instruct-v0.1"
OPENAI_MODEL = set(["gpt-3.5-turbo", "gpt-4-turbo-preview"])

ENTITY_SCHEMA = {
    "SECTOR": [],
    "COMPANY": [],
    "PEOPLE": [],
    "PRODUCTS": []
}


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
    return response.choices[0].message.content


def parse_entity(text):
    lines = text.split('\n')
    entity_dict = {}
    for line in lines:
        # Split the line into entity type and values
        parts = line.split(':')
        if len(parts) != 2:
            continue
        entity_type, values = parts
        entity_type = entity_type.strip()
        values = values.strip()

        # Check if there are values listed for the entity type
        if not values:
            entity_dict[entity_type] = []
            continue

        # Split the values by comma and strip whitespace
        entity_list = [value.strip() for value in values.split(',')]

        # Add the values to the dictionary under the correct entity type
        if entity_type in ENTITY_SCHEMA:
            entity_dict[entity_type] = entity_list

    return entity_dict


def extract_entities(text, sectors):
    query = ENTITY_PASS1.format(text=text, sectors=sectors)
    result = run_query(query, ENTITY_SYSTEM_PASS1, model3)
    entities = parse_entity(result)  # extract {category: [entity1, entity2]}
    return entities


def create_prompts_for_predefined_pairs(entities, rel_schema, input_text):
    all_results = []

    predefined_pairs = [
        'Sector-Company',
        'Company-Company',
        'Company-Products',
        'Company-People'
    ]

    # Generate prompts for each predefined pair
    for pair in predefined_pairs:
        category1, category2 = pair.upper().split('-')

        # Skip if either category in the pair is empty
        if not entities.get(category1) or not entities.get(category2):
            continue

        # Fetch the possible relations for this pair
        possible_relations = rel_schema.get(pair, [])

        # Format the entities for the prompt
        entities1 = ', '.join(entities[category1])
        entities2 = ', '.join(entities[category2])

        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
        print(f"{category1}: {entities1}\n{category2}: {entities2}")
        print("possible relation", possible_relations)

        # Construct prompt
        query = RELATION_PROMPT.format(
            text=input_text,
            cat1=f"{category1}",
            entity1=f"{entities1}",
            cat2=f"{category2}",
            entity2=f"{entities2}",
            possible_relations=f"{possible_relations}"
        )
        result = run_query(query, RELATION_SYSTEM_PROMPT, model1)

        print('intermediate result', result)
        ver_query = VERIFICATION_PROMPT1.format(
            text=input_text,
            predefined_entities=f"{category1}: {entities1}\n{category2}: {entities2}",
            predefined_relations=f"{possible_relations}",
            triplets=f"{result}"
        )
        verified_result = run_query(ver_query, VERIFICATION_SYSTEM_PROMPT1, model2)
        print('=========================>>>>>verified result\n', verified_result)
        # Store the verified triplets along with the input text
        # Collect each verified result for the document
        all_results.append(verified_result)

        # Combine all verified results for the document into one string, separated by a delimiter (e.g., "; ")
    all_results = ", ".join(all_results)

    # Store the combined verified results along with the input text
    combined_results = [(input_text, all_results)]

    return combined_results


def format_metadata(row):
    metadata = {
        "title": row.get("title", ""),
        "publisher": row.get("publisher", ""),
        "authors": row.get("authors", ""),
        "published_date": row.get("published_date", "")
    }
    return json.dumps(metadata, ensure_ascii=False)  # Use ensure_ascii=False to properly handle Unicode characters


def chunk_text(text, char_limit):
    return [text[i:i + char_limit] for i in range(0, len(text), char_limit)]


# def chunk_text_by_words(text, words_per_chunk):
#     words = text.split()
#     for i in range(0, len(words), words_per_chunk):
#         yield ' '.join(words[i:i + words_per_chunk])

def chunk_text_by_sentences(text, approx_words_per_chunk):
    words = text.split()
    chunk, word_count = "", 0

    for word in words:
        chunk += word + " "
        word_count += 1

        if word.endswith('.') and word_count >= approx_words_per_chunk:
            yield chunk
            chunk, word_count = "", 0

    if chunk:  # Yield the last chunk if there is any remaining text
        yield chunk


def log_and_save(exception, df, output_file_path):
    traceback.print_exc(file=log_file_news)
    if not df.empty:
        df.to_csv(output_file_path, index=False)
    log_file.flush()


def log_and_save_json(exception, data, output_file_path):
    traceback.print_exc(file=log_file_sec)
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    log_file.flush()


def process_news_data(input_filename, output_folder_path, sectors, rel_schema, log_file):
    sys.stdout = log_file
    try:
        all_output_rows = []
        news_data = pd.read_csv(input_filename)
        base_filename = os.path.basename(input_filename)

        news_df = pd.DataFrame(columns=['InputText', 'Relationships', 'SectorData', 'MetaData'])

        for index, row in news_data.iterrows():
            try:
                text = row['text']
                print(f'Processing index: {index}')
                entities = extract_entities(text, sectors)
                sector_data = json.dumps(entities)
                metadata = format_metadata(row)
                results = create_prompts_for_predefined_pairs(entities, rel_schema, text)
                output_rows = [(input_text, relationships, sector_data, metadata) for input_text, relationships in
                               results]
                all_output_rows.extend(output_rows)

            except Exception as e:
                news_df = pd.DataFrame(all_output_rows,
                                       columns=['InputText', 'Relationships', 'SectorData', 'MetaData'])
                error_output_file_path = os.path.join(output_folder_path, f"{base_filename[:-4]}_error.csv")
                news_df.to_csv(error_output_file_path, index=False)
                traceback.print_exc(file=log_file)
                break

        else:
            news_df = pd.DataFrame(all_output_rows, columns=['InputText', 'Relationships', 'SectorData', 'MetaData'])
            output_file_path = os.path.join(output_folder_path, f"{base_filename[:-4]}_sample_relations.csv")
            news_df.to_csv(output_file_path, index=False)
            print(f"Processed {base_filename} and saved to {output_file_path}", file=log_file)

    except Exception as e:
        traceback.print_exc(file=log_file)

    finally:
        log_file.close()


def process_sec_data(json_filename, output_folder_path, sectors, rel_schema, log_file):
    try:
        sys.stdout = log_file
        with open(json_filename, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        base_filename = os.path.basename(json_filename)
        all_output_data = []
        all_output_rows_csv = []

        for company, company_data in json_data.items():
            for item_key, text in company_data.items():
                # text_chunks = chunk_text(text, CHAR_LIMIT)
                print(f'Processing index: {item_key}')
                text_chunks = chunk_text_by_sentences(text, WORDS_PER_CHUNK)
                for chunk in text_chunks:
                    try:
                        entities = extract_entities(chunk, sectors)
                        sector_data = json.dumps(entities)
                        results = create_prompts_for_predefined_pairs(entities, rel_schema, chunk)
                        for input_text, relationships in results:
                            output_row = {
                                "Company": company,
                                "ItemKey": item_key,
                                "Text": input_text,
                                "Relationships": relationships,
                                "SectorData": sector_data
                            }
                            all_output_data.append(output_row)
                            all_output_rows_csv.append([company, item_key, input_text, relationships, sector_data])


                    except Exception as e:
                        output_json_path = os.path.join(output_folder_path,
                                                        f"{os.path.basename(json_filename)[:-5]}_error.json")
                        output_csv_path = os.path.join(output_folder_path,
                                                       f"{os.path.basename(json_filename)[:-5]}_error.csv")
                        with open(output_json_path, 'w', encoding='utf-8') as f:
                            json.dump(all_output_data, f, ensure_ascii=False, indent=4)
                        pd.DataFrame(all_output_rows_csv,
                                     columns=['Company','ItemKey', 'Text', 'Relationships', 'SectorData']).to_csv(
                            output_csv_path, index=False)
                        traceback.print_exc(file=log_file)
                        return

        output_json_path = os.path.join(output_folder_path, f"{os.path.basename(json_filename)[:-5]}_processed.json")
        output_csv_path = os.path.join(output_folder_path, f"{os.path.basename(json_filename)[:-5]}_processed.csv")
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(all_output_data, f, ensure_ascii=False, indent=4)
        pd.DataFrame(all_output_rows_csv, columns=['Company', 'ItemKey','Text', 'Relationships', 'SectorData']).to_csv(
            output_csv_path, index=False)


    except Exception as e:

        output_json_path = os.path.join(output_folder_path, f"{os.path.basename(json_filename)[:-5]}_error.json")
        output_csv_path = os.path.join(output_folder_path, f"{os.path.basename(json_filename)[:-5]}_error.csv")
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(all_output_data, f, ensure_ascii=False, indent=4)
        pd.DataFrame(all_output_rows_csv, columns=['Company','ItemKey', 'Text', 'Relationships', 'SectorData']).to_csv(
            output_csv_path, index=False)
        traceback.print_exc(file=log_file)

    finally:
        log_file.close()



# TOKEN_LIMIT = 10000
# CHARS_PER_TOKEN = 4
# CHAR_LIMIT = TOKEN_LIMIT * CHARS_PER_TOKEN
WORDS_PER_CHUNK = 1000

# All schemas and inputs
with open('eval/schema.json', 'r') as f:
    entity_schema = json.load(f)
with open('eval/relationschema.json', 'r') as f:
    rel_schema = json.load(f)

sectors = entity_schema['sectors']
product_type = entity_schema['product-type']

output_folder_path = 'Data/ProcessedData'
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)
#
# input_news_file = 'Data/GoogleNewsData/TSM_news_data.csv'

# input_news_file = 'Data/sample_input_news.csv'
input_sec_file = 'Data/SECData/NVDA.json'

# process_news_data(input_news_file, output_folder_path, sectors, rel_schema, log_file_news)
process_sec_data(input_sec_file, output_folder_path, sectors, rel_schema, log_file_sec)

# Reset stdout to its original state
sys.stdout = original_stdout
