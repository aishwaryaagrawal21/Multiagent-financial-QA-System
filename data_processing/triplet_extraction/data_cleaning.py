import pandas as pd
import re
import os
import glob

def clean_relationships(data):
    if not isinstance(data, str):
        return '', data

    tuples = re.findall(r'\(([^)]+)\)', data)
    cleaned = []
    discarded = data

    seen_pairs = set()
    for t in tuples:
        entities = [x.strip() for x in t.split(',')]

        if len(entities) != 3:
            continue

        entity_pair = tuple(sorted([entities[0], entities[2]]))
        tuple_str = f'({t})'

        if entity_pair not in seen_pairs:
            seen_pairs.add(entity_pair)
            cleaned.append(tuple_str)
            discarded = discarded.replace(tuple_str, '')

    return '\n'.join(cleaned).strip(), discarded.strip()

def clean_csv_files(folder_path):
    cleaned_folder = os.path.join(folder_path, 'Cleaned')
    if not os.path.exists(cleaned_folder):
        os.makedirs(cleaned_folder)

    for file_path in glob.glob(os.path.join(folder_path, '*.csv')):
        try:
            print(file_path)
            df = pd.read_csv(file_path)

            # Check if 'Relationships' column exists
            if 'Relationships' not in df.columns:
                print(f"Skipping {file_path}: 'Relationships' column not found.")
                continue

            new_data = df['Relationships'].apply(clean_relationships)
            df['Relationships'] = new_data.apply(lambda x: x[0])
            df['Discarded_Relationships'] = new_data.apply(lambda x: x[1])

            base_name = os.path.basename(file_path)
            new_file_name = f"{os.path.splitext(base_name)[0]}_cleaned.csv"
            new_file_path = os.path.join(cleaned_folder, new_file_name)

            df.to_csv(new_file_path, index=False)
            print(f"Processed and saved: {new_file_name}")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

folder_path = 'Data/ProcessedData'  # Replace with actual folder path
clean_csv_files(folder_path)
