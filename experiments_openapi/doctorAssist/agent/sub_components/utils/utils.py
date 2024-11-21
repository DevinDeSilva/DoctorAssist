import Levenshtein
import json
import os
import pandas as pd

cwd_path = os.path.dirname(os.path.abspath(__file__))
name_synonyms = json.load(open(f"{cwd_path}/data/name_synonyms.json", 'r'))
drugbank_df = pd.read_csv(f"{cwd_path}/data/drugbank.csv", sep='\t')
drug_names = drugbank_df['name'].str.lower().tolist()


def get_drug_bank_df():
    return drugbank_df

def find_least_levenshtein_distance(target_string, array):
    array = list(array)

    # Ensure the array is not empty
    if not array:
        return None, float('inf')

    # Initialize minimum distance and corresponding string
    min_distance = float('inf')
    min_string = None

    # Iterate through each string in the array
    for string in array:
        # Calculate the Levenshtein distance
        distance = Levenshtein.distance(target_string, string)

        if distance == 0:
            print(f"Similar Name: {target_string} -> {string}, levenshtein distance: {distance}")
            return string, 0

        # Update minimum distance and string if a new minimum is found
        if distance < min_distance:
            min_distance = distance
            min_string = string
    
    print(f"Similar Name: {target_string} -> {min_string}, levenshtein distance: {min_distance}")

    return min_string, min_distance

def get_drug_synonyms(drug_name):
    drug_name = drug_name.strip().lower()
    if drug_name in name_synonyms:
        return name_synonyms[drug_name]
    else:
        return [drug_name]

def match_name(name, target_all_names):
    name_synonyms = get_drug_synonyms(name)
    for n in name_synonyms:
        if n in target_all_names:
            return n
        
    print(f"Name: {name} and its synonyms not found")
    print(f"Similary Name Matching...")

    similar_name, distance = find_least_levenshtein_distance(
        name, target_all_names)

    return similar_name

def get_SMILES(drug_name):
    drug_name = drug_name.strip().lower()

    drug_name = match_name(drug_name, drug_names)

    db_row = drugbank_df[drugbank_df['name'] == drug_name]

    if db_row.empty:
        return ""

    return db_row['smiles'].values[0]