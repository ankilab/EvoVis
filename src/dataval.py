
import os
import re
import json
import pandas as pd


################################################################################################################################################

# MODULE ENAS DATA CHECK

# The Enas Data Check Module provides functionalities to the data structure of configurations and results of a given run.
# Parameters: run (str) The identifier for the run. This is used to construct the path to the search_space.json file.
# Returns: (str) A message suggesting improvements to the data structure if validation fails or an empty string if the data structure is valid.

################################################################################################################################################


def json_to_dict(filepath):
    """
    Convert JSON data from a file to a Python dictionary.

    Parameters:
        filepath (str): The path to the JSON file.

    Returns:
        dict: A Python dictionary representing the JSON data.

    Raises:
        FileNotFoundError: If the specified file is not found.
        json.JSONDecodeError: If there is an issue decoding the JSON data.

    Example:
    >>> data = json_to_dict('example.json')
    """
    
    try:
        with open(filepath, 'r') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}")
    
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Error decoding JSON data in file {filepath}: {e}")

def validate_hyperparameters(run):
    filepath = f"{run}/config.json"
    
    # Check if config.json exists
    if not os.path.exists(filepath):
        return "Error config.json file: Config file 'config.json' not found."
    
    # Check if the file is a JSON
    try:
        data = json_to_dict(filepath)
    except json.JSONDecodeError:
        return "Error config.json file: Invalid JSON format in config.json."
    
    # Check if 'hyperparameters' key exists
    if 'hyperparameters' not in data:
        return "Error config.json file: Missing 'hyperparameters' key in config.json."
    
    hyperparameters = data['hyperparameters']
    
    # Check if hyperparameters contain a dictionary as value
    if not isinstance(hyperparameters, dict):
        return "Error config.json file: Hyperparameters must be a dictionary."
    
    message = ""
    
    # Check each hyperparameter
    for hp, details in hyperparameters.items():
        
        if not isinstance(details, dict):
            message += f"Error config.json file: Hyperparameter '{hp}' settings must be a dictionary.\n"
            continue
        
        # Check if 'value' key exists
        if 'value' not in details:
            message += f"Error config.json file: Missing 'value' key for hyperparameter '{hp}'.\n"
        
    return message

def validate_search_space(run):
    
    filepath = f"{run}/search_space.json"
    
    ### CHECK FILE ###
    # Check if search_space.json exists
    if not os.path.exists(filepath):
        return "Error search_space.json file: Search spcae file 'search_space.json' not found."
    
    # Check if the file is a JSON
    try:
        data = json_to_dict(filepath)
    except json.JSONDecodeError:
        return "Error search_space.json file: Invalid JSON format in search_space.json."
    
    
    ### CHECK KEYS ###
    # Check 'gene_pool' key
    if 'gene_pool' not in data:
        return "Error search_space.json file: Missing 'gene_pool' key in search_space.json."
    
    gene_pool = data['gene_pool']
    
    if not isinstance(gene_pool, dict):
        return "Error config.json file: Gene pool must be a dictionary."
    
    # Check 'rule_set' key
    if 'rule_set' not in data:
        return "Error search_space.json file: Missing 'rule_set' key in search_space.json."
    
    rule_set = data['rule_set']
    
    if not isinstance(rule_set, dict):
        return "Error config.json file: Rule set must be a dict."
    
    # Check 'rule_set_group' key (doesn't need to exists if no group rules)
    rule_set_group = data.get('rule_set_group', [])
    
    if not isinstance(rule_set_group, list):
        return "Error config.json file: Rule set groups must be a list."

    ### CHECK DATA STRUCTURES ###
    message = ""
    
    # Check gene pool structure
    for group, genes in gene_pool.items():
        if not isinstance(genes, list):
            message += f"Error search_space.json file: Gene pool '{group}' must be a list.\n"
        else:
            for gene in genes:
                if not isinstance(gene, dict):
                    message += f"Error search_space.json file: Gene in '{group}' must be a dictionary.\n"
                elif 'layer' not in gene:
                    message += f"Error search_space.json file: Gene in '{group}' is missing 'layer' key.\n"
                elif 'f_name' not in gene:
                    message += f"Error search_space.json file: Gene in '{group}' is missing 'f_name' key.\n"
                    
    # Check rule set structure
    start_found = False
    
    for rule_entry in rule_set:
        if not isinstance(rule_entry, str):
            message += "Error search_space.json file: Rule key in 'rule_set' must be a string.\n"
        elif 'rule' not in rule_set[rule_entry].keys():
            message += "Error search_space.json file: Rule entry in 'rule_set' is missing 'rule' key.\n"
        else:
            if rule_entry == "Start":
                start_found = True

    if not start_found:
        message += "Error search_space.json file: Rule entry with layer 'Start' is missing.\n"
        
    # Check rule set group structure
    for group_entry in rule_set_group:
        if not isinstance(group_entry, dict):
            message += "Error search_space.json file: Group rule in 'rule_set_group' must be a dictionary.\n"
        elif 'group' not in group_entry:
            message += "Error search_space.json file: Group rule entry in 'rule_set_group' is missing 'layer' key.\n"
        elif 'rule' not in group_entry:
            message += "Error search_space.json file: Group rule entry in 'rule_set_group' is missing 'rule' key.\n"
    
    return message

def validate_crossover_parents(run):
    
    filepath = f"{run}/crossover_parents.csv"
    
    # Check if config.json exists
    if not os.path.exists(filepath):
        return "Error crossover_parents.csv file: Config file 'crossover_parents.csv' not found."
    
    # Check if the file is a CSV file
    try:
        df = pd.read_csv(filepath, header=None)
    except pd.errors.EmptyDataError:
        return "Error crossover_parents.csv file: Invalid CSV format in crossover_parents.csv."
    
    message = ""
    
    for idx, row in df.iterrows():
        
        # Check if all columns are present
        if len(row) < 4:
            message += f"Error crossover_parents.csv file row {idx+1}: Not all columns are present.\n"
            continue

        # Extracting values from each row
        if "Generation: " in row[0]:
            generation = row[0].split(",")[0].replace("Generation: ", "")
                
            if not generation.isdigit():
                message += f"Error crossover_parents.csv file row {idx+1}: Generation should be a number.\n"
        else:
            message += f"Error crossover_parents.csv file row {idx+1}: 'Generation' label not found.\n"
        
        if "Parent_1: " in row[1]:
            parent1 = row[1].split(",")[0].replace("Parent_1: (", "").strip()
            parent1_value = row[1].split(",")[1].replace(")", "").strip()
                
            if not parent1:
                message += f"Error crossover_parents.csv file row {idx+1}: Parent 1 is missing.\n"
                    
            if not parent1_value.isdigit():
                message += f"Error crossover_parents.csv file row {idx+1}: Parent 1 crossover value should be a number.\n"
        else:
            message += f"Error crossover_parents.csv file row {idx+1}: 'Parent_1' label not found.\n"

        if "Parent_2: " in row[2]:
            parent2 = row[2].split(",")[0].replace("Parent_2: (", "").strip()
            parent2_value = row[2].split(",")[1].replace(")", "").strip()
                
            if not parent2:
                message += f"Error crossover_parents.csv file row {idx+1}: Parent 2 is missing.\n"
                    
            if not parent2_value.isdigit():
                message += f"Error crossover_parents.csv file row {idx+1}: Parent 2 crossover value should be a number.\n"    
        else:
            message += f"Error crossover_parents.csv file row {idx+1}: 'Parent_2' label not found.\n"

        if "New_Individual: " in row[3]:
            new_individual = row[3].replace("New_Individual: ", "").strip()
                
            if not new_individual:
                message += f"Error crossover_parents.csv file row {idx+1}: New Individual is missing.\n"
        else:
            message += f"Error crossover_parents.csv file row {idx+1}: 'New_Individual' label not found.\n"

    return message

def validate_generations_of_individuals(run):
    
    # Check for the presence of generation directories
    generation_directories = [d for d in os.listdir(run) if re.match(r"^Generation_\d+$", d)]
    
    if not generation_directories:
        return "No generation directories found."

    # Check for individual directories and their required files
    for generation_directory in generation_directories:
        generation_dir_path = os.path.join(run, generation_directory)
        generation_contents = os.listdir(generation_dir_path)
        
        for item in generation_contents:
            individual_directory = os.path.join(generation_dir_path, item)
            
            # Skip non-directory items
            if not os.path.isdir(individual_directory):
                continue 
            
            # Check for required files for individuals
            individual_files = ["chromosome.json", "results.json"]
            
            for file_name in individual_files:
                file_path = os.path.join(individual_directory, file_name)
                
                if not os.path.exists(file_path):
                    return f"Missing file in individual {item}: {file_name}"

    return ""

def validate_meas_info(run):
    filepath = f"{run}/config.json"
    
    # Check if config.json exists
    if not os.path.exists(filepath):
        return "Error config.json file: Config file 'config.json' not found."
    
    # Check if the file is a JSON
    try:
        data = json_to_dict(filepath)
    except json.JSONDecodeError:
        return "Error config.json file: Invalid JSON format in config.json."
    
    # Check if 'reslts' key exists
    if 'results' not in data:
        return "Error config.json file: Missing 'results' key in config.json."
    
    results = data['results']
    
    # Check if hyperparameters contain a dictionary as value
    if not isinstance(results, dict):
        return "Error config.json file: Results must be a dictionary."
    
    message = ""
    
    # Check each hyperparameter
    for result, details in results.items():
        
        if not isinstance(details, dict):
            message += f"Error config.json file: Result '{result}' settings must be a dictionary.\n"
            continue
        
    return message

def validate_individual_result(run, generation, individual):
    
    filepath = f"{run}/Generation_{generation}/{individual}/results.json"
    
    # Check if config.json exists
    if not os.path.exists(filepath):
        return f"Error results.json file for {individual} in {generation}: Results file 'results.json' not found."
    
    # Check if the file is a JSON
    try:
        data = json_to_dict(filepath)
    except json.JSONDecodeError:
        return f"Error results.json file for {individual} in {generation}: Invalid JSON format in results.json."
    
    return ""

def validate_individual_chromosome(run, generation, individual):
    
    filepath = f"{run}/Generation_{generation}/{individual}/chromosome.json"
    
    print(filepath)
    
    # Check if config.json exists
    if not os.path.exists(filepath):
        return f"Error chromosome.json file for {individual} in {generation}: Chromosome file 'chromosome.json' not found."
    
    # Check if the file is a JSON
    try:
        data = json_to_dict(filepath)
    except json.JSONDecodeError:
        return f"Error chromosome.json file for {individual} in {generation}: Invalid JSON format in chromosome.json."
    
    return ""