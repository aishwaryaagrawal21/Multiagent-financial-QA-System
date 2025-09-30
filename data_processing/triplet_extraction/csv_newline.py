import pandas as pd
import os

def transform_relationships(column):
    """Transforms the relationships data by placing each entity on a new line."""
    if pd.notna(column):
        return column.replace(')(', ')\n(')
    else:
        return column 

def process_csv(file_path):
    """Reads, processes, and writes back a CSV file."""
    df = pd.read_csv(file_path)
    df['Relationships'] = df['Relationships'].apply(transform_relationships)
    df.to_csv(file_path+'_Processed', index=False)  # Overwrite the original file or specify a new file path

# Directory containing the CSV files
directory_path = '/Users/vinwizard/Desktop/CMU/Sem4/Capstone/Code/finatars/TripletExtraction/FinalData'

# Loop through all files in the directory
for filename in os.listdir(directory_path):
    if filename.endswith('.csv'):  # Check if the file is a CSV
        file_path = os.path.join(directory_path, filename)
        process_csv(file_path)
        print(f'Processed {filename}')