import pandas as pd
from collections import Counter
import re
import os

# Load the CSV data
data_path = '/Users/vinwizard/Desktop/CMU/Sem4/Capstone/Code/finatars/TripletExtraction/FinalData'
# Construct the search pattern to match all .csv files
files = os.listdir(data_path)
print(files)

term_mapping = {
    'advanced micro devices': 'amd',
    'amd': 'amd',
    'nvidia corporation': 'nvidia',
    'nvidia': 'nvidia',
    'nvda':'nvidia',
    'intel corporation':'intel',
    'hewlett packard enterprise company':'hewlett packard',
    'hewlett-packard':'hewlett packard',
    'cisco systems':'cisco',
    'msft': 'microsoft',
    'microsoft': 'microsoft',
    'microsoft corporation': 'microsoft',
    'microsoft msft': 'microsoft',
    'meta': 'meta',
}

for file in files:
    if file.endswith('.csv'):  # Ensure to process only CSV files
        data = pd.read_csv('/Users/vinwizard/Desktop/CMU/Sem4/Capstone/Code/finatars/TripletExtraction/FinalData/AMD_news_cleaned.csv')  # Use os.path.join for proper path handling

        # Trim whitespace from the Relationships column
        # Convert all entries in 'Relationships' to string and handle missing values
        data['Relationships'] = data['Relationships'].fillna('').astype(str).str.lower()
        data['Relationships'] = data['Relationships'].str.lower().apply(lambda x: re.sub(r'\s*\((.*?)\)\s*', lambda m: '(' + m.group(1).strip() + ')', x))

        # Create a Counter object to hold the frequencies of the entities
        entity_counter = Counter()

        # Function to process each relationship entry, extract entities, and update the counter
        def process_relationships(relationships):
            # Split the relationships string into individual triplets (assuming they are separated by commas)
            triplets = re.findall(r'\((.*?)\)', relationships)
            for triplet in triplets:
                # Split each triplet into its entities (assuming they are separated by commas)
                entities = [entity.strip() for entity in triplet.split(',')]
                if len(entities) == 3:  # Ensure it's a triplet
                    # Update the counter for the first and last entity
                    entity_counter.update([entities[0].lower(), entities[2].lower()])

        # Apply the function to each row in the Relationships column
        data['Relationships'].dropna().apply(process_relationships)

        # Display the most common entities to manually identify which ones refer to AMD
        print(entity_counter.most_common())

        # Function to replace all occurrences of a company with its primary key
        def replace_terms(relationships, term_mapping):
            for term, primary_key in term_mapping.items():
                # Use regex to find whole word matches with optional surrounding whitespace
                relationships = re.sub(r'\b\s*{}\s*\b'.format(re.escape(term)), primary_key, relationships, flags=re.IGNORECASE)
            return relationships

        # Apply the replacement to the Relationships column
        data['Relationships'] = data['Relationships'].dropna().apply(lambda x: replace_terms(x, term_mapping))

        # Export the cleaned data
        output_path = os.path.join(data_path, 'TSM_news' + '_standardized.csv')
        # data.to_csv(output_path, index=False)
        exit(0)