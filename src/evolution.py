import pandas as pd
import numpy as np
import os
import random
from matplotlib.colors import hex2color, rgb2hex
import json


##################################################

# MODULE EVOLUTION

# The Evolution Module provides functionalities 
# for generating the family tree cytoscape,
# for processing information about individuals, 
# and for providing run configurations. 

##################################################


### READ HELPER ###
def _json_to_dict(filepath):
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
    >>> data = _json_to_dict('example.json')
    """
    try:
        with open(filepath, 'r') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {filepath}")
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Error decoding JSON data in file {filepath}: {e}")


### RUN INFORMATION ###
def _get_configurations(run):
    """
    Retrieve configurations from the specified run.

    Args:
        run (str): The path of the ENAS run results directory.

    Returns:
        dict: A dictionary containing the configurations retrieved from the specified run.
    """
    return _json_to_dict(f'{run}/config.json')

def get_hyperparameters(run):
    """
    Retrieve hyperparameters from the specified run.

    Args:
        run (str): The path of the ENAS run results directory.

    Returns:
        dict: A dictionary containing the hyperparameters retrieved from the specified run.
    """
    configs = _get_configurations(run)
    return configs["hyperparameters"]

def get_meas_info(run):
    """
    Retrieve measurement information from the specified run.

    Args:
        run (str): The path of the ENAS run results directory.

    Returns:
        dict: A dictionary containing the measurement information retrieved from the specified run.
    """
    configs = _get_configurations(run)
    results = configs["results"]
    
    for result, setting in results.items():
        setting["displayname"] = setting.get("displayname", result) 
        setting["unit"] = setting.get("unit", None) 
        setting["run-result-plot"] = setting.get("run-result-plot", True) 
        setting["individual-info-plot"] = setting.get("individual-info-plot", True) 
        setting["pareto-optimlity-plot"] = setting.get("pareto-optimlity-plot", False) 
        setting["individual-info-img"] = setting.get("individual-info-img", "measure1-icon.png") or "measure1-icon.png"
        setting["min-boundary"] = setting.get("min-boundary", None) 
        setting["max-boundary"] = setting.get("max-boundary", None) 
    
    return configs["results"]


### NAMES OF DIRECTORIES ###
def get_generations(run, as_int=False):
    """
    Get a list of all generation names (directories) in the specified run.

    Args:
        run (str): The path of the ENAS run results directory.
        as_int (bool, optional): If True, returns a list of generation numbers as integers. Default is False.

    Returns:
        list: A list of generation names or numbers, sorted in ascending order.

    Raises:
        FileNotFoundError: If the specified run directory does not exist.
        ValueError: If 'as_int' is not a boolean.

    Example:
        >>> generation_names = get_generations('my_run')
        >>> print(generation_names)
        ['Generation_1', 'Generation_2', ...]

        >>> generation_numbers = get_generations('my_run', as_int=True)
        >>> print(generation_numbers)
        [1, 2, ...]
    """
    # Validate input values
    if not os.path.exists(run):
        raise FileNotFoundError(f"Run directory not found: {run}")

    if not isinstance(as_int, bool):
        raise ValueError("'as_int' must be a boolean.")

    # Retrieve generation names and numbers
    generation_path = run
    items = os.listdir(generation_path)
    generations = [item for item in items if os.path.isdir(os.path.join(generation_path, item))]
    generations_int = [int(gen.split("_")[1]) for gen in generations]

    generations_int.sort()
    
    # Check last generation finished processed
    take_last_gen = True
    results_last_gen = _get_individuals_of_generation(run, generations_int[-1], "results")
    
    for result in results_last_gen.values():
        if result.get("fitness", None) is None:
            take_last_gen = False
            
    if not take_last_gen:
        generations_int = generations_int[0:len(generations_int)-1]
        
    if not as_int:
        return [f"Generation_{gen}" for gen in generations_int]
    else:
        return generations_int


### SINGLE INDIVIDUAL INFORMATION ###
def get_individual_result(run, generation, individual):
    """
    Retrieve individual's objective measurements from a JSON file.

    Parameters:
        run (str): The path of the ENAS run results directory.
        generation (int): The generation number.
        individual (str): The individual's identifier.

    Returns:
        dict or None: A dictionary containing individual's objective measurements, or None if the file doesn't exist.

    Raises:
        FileNotFoundError: If the specified file path does not exist.
        Exception: If an error occurs during the processing of the JSON file.

    """
    # Construct the file path
    path = f'{run}/Generation_{generation}/{individual}/results.json'

    # Check if the file exists
    if os.path.isfile(path):
        try:
            # Load JSON data from file
            results = _json_to_dict(path)

            # Process nested dictionaries
            for key, val in results.items():
                if isinstance(val, dict):
                    numeric_values = [value for value in val.values() if isinstance(value, (int, float))]

                    if numeric_values:
                        # Calculate the average of numeric values
                        val = sum(numeric_values) / len(numeric_values)
                        results[key] = val

            return results

        except Exception as e:
            
            return {"error": str(e)}

    else:
        # File not found, return None
        return None
    
def get_individual_chromosome(run, generation, individual):
    """
    Retrieve individual's genes/layers from a JSON file representing the chromosome.

    Parameters:
        run (str): The path of the ENAS run results directory.
        generation (int): The generation number.
        individual (str): The individual's identifier.

    Returns:
        list or None: A list containing individual's genes/layers represented as dictionaries, or None if the file doesn't exist.

    Raises:
        FileNotFoundError: If the specified file path does not exist.
        Exception: If an error occurs during the processing of the JSON file.

    """
    # Construct the file path
    path = f'{run}/Generation_{generation}/{individual}/chromosome.json'

    # Check if the file exists
    if os.path.isfile(path):
        try:
            # Load JSON data from file
            chromosome = _json_to_dict(path)
            return chromosome

        except Exception as e:
            return {"error": str(e)}

    else:
        # File not found, return None
        return None

    
### INDIVIDUALS INFORMATION ###
def _get_individuals_of_generation(run, generation, value="names"):
    """
    Get the data of all individuals of a specified generation.
    
    Args:
        run (str): The path of the ENAS run results directory.
        generation (int): The generation number.
        value (str): The data of an individual that will be accessed. Available are "names", "results" or "chromosome".
    
    Returns:
        individuals_names (list) or individual_dict (dict): Get the list of names if value is set to name or a dictionary with individual names as keys.
        
    Raises:
        ValueError: If value is not one of the allowed values ("names", "results", "chromosome").

    Example:
        >>> names = _get_individuals_of_generation('my_run', 3, 'names')
        >>> results_dict = _get_individuals_of_generation('my_run', 3, 'results')
    """
    
    # Validate input values
    available_values = ["names", "results", "chromosome"]
    
    if value not in available_values:
        raise ValueError(f"Invalid value. Allowed values are {available_values}.")

    # Access individuals names
    items = os.listdir(f"{run}/Generation_{generation}")
    individuals_names = [item for item in items if os.path.isdir(os.path.join(f"{run}/Generation_{generation}", item))]

    if value == "names":
        individuals_names.sort()
        return individuals_names
    
    # Access individuals data
    individual_dict = {}

    for individual in individuals_names:

        if value == "results":
            individual_dict[individual] = get_individual_result(run, generation, individual)
        
        elif value == "chromosome":
            individual_dict[individual] = get_individual_chromosome(run, generation, individual)
    
    return individual_dict

def get_individuals(run, generation_range=None, value="names", as_generation_dict=False):
    """
    Get data from the individuals of a generation range or all the generations. 
    
    Args: 
        run (str): The path of the ENAS run results directory.
        generation_range (range): A python range of generations from which the individuals will be extracted. 
        value (str): The return values of the individuals. Choose between the values "names", "results" or "chromosome".
        as_generation_dict (bool): Specify if individuals values should be returned as generation and individual name dictionairies.
        
    Returns:
        generations (dict) or individuals (list): Outputs a dict with generation key (in int) containing dict with individual key or a list only with the values (no generation, individual name specifications). 

    Raises:
        ValueError: If the specified generation range is not a valid range.
                   If the specified value is not one of the allowed values ("names", "results", "chromosome").

    Example:
        >>> generations_dict = get_individuals('my_run', range(1, 5), 'results', as_generation_dict=True)
        >>> individuals_list = get_individuals('my_run', generation_range=range(1, 5), value='names')
    """
    
    # Validate input values
    available_values = ["names", "results", "chromosome"]
    if value not in available_values:
        raise ValueError(f"Invalid value. Allowed values are {available_values}.")
    
    # Generation range creation in case of None
    all_generations = get_generations(run)
    generation_range = range(1, len(all_generations) + 1) if generation_range is None else generation_range
    
    # Validate generation_range
    if not isinstance(generation_range, range) or len(generation_range) == 0:
        raise ValueError("Invalid generation range.")
    
    if any(g not in range(1, len(all_generations) + 1) for g in generation_range):
        raise ValueError(f"Invalid generation in range. Available generations are {all_generations}.")
    
    if as_generation_dict:
        generations = {}
        for generation in generation_range:
            generations[generation] = _get_individuals_of_generation(run, generation, value)
        return generations
    
    else: 
        individuals = []
        for generation in generation_range:
            if value != "names":
                individuals += list(_get_individuals_of_generation(run, generation, value).values())
            else:
                individuals += _get_individuals_of_generation(run, generation, value)

        return individuals
      
def _get_minmax_result_by_key(results, key, min_boundary=None, max_boundary=None):
    """
    Finds the minimum and maximum values in a list of dictionaries based on a specified key,
    while respecting optional minimum and maximum boundaries.

    Parameters:
        results (list): A list of dictionaries containing key-value pairs.
        key (str): The key to be used for comparison in dictionaries.
        min_boundary (int or float, optional): The minimum boundary for the values. Default is None.
        max_boundary (int or float, optional): The maximum boundary for the values. Default is None.

    Returns:
        tuple: A tuple containing the adjusted minimum and maximum values respecting the boundaries.

    Raises:
        ValueError: If 'key' is not present in a dictionary or if the corresponding value is not an int or float.

    Example:
    >>> my_list_of_dicts = [{'value': 3}, {'value': 'invalid'}, {'value': 5}]
    >>> min_val, max_val = _get_minmax_result_by_key(my_list_of_dicts, 'value', min_boundary=2, max_boundary=4)
    >>> print(min_val, max_val)
    (3, 4)
    """
    if not results:
        return None, None  # Return None for both min and max if the list is empty

    min_val = float('inf')
    max_val = float('-inf')

    for result in results:
        if result is not None and key in result and isinstance(result[key], (int, float)):
            
            val = result[key]

            if min_boundary is not None:
                val = max(val, min_boundary)

            if max_boundary is not None:
                val = min(val, max_boundary)

            min_val = min(min_val, val)
            max_val = max(max_val, val)
            
    return min_val, max_val

def get_individuals_min_max(run, generation_range=None):
    """
    Get the minimum and maximum values for various measurements across generations and individuals.

    Parameters:
        run (str): The path of the ENAS run results directory.
        generation_range (tuple, optional): A tuple specifying the range of generations to consider (start, end).

    Returns:
        measurements (dict): A dictionary containing the minimum and maximum values for each measurement.

    Raises:
        TypeError: If 'generation_range' is not a tuple.
        ValueError: If 'generation_range' is not a valid tuple of two integers.

    Example:
    >>> measurements = get_individuals_min_max('my_run', generation_range=(1, 3))
    >>> print(measurements)
    {'measurement_1': (min_val_1, max_val_1), 'measurement_2': (min_val_2, max_val_2), ...}
    """
    if generation_range is not None and not isinstance(generation_range, tuple):
        raise TypeError("Invalid type for 'generation_range'. Must be a tuple.")

    if generation_range is not None and (len(generation_range) != 2 or not all(isinstance(g, int) for g in generation_range)):
        raise ValueError("Invalid 'generation_range'. Must be a tuple of two integers.")

    values = get_individuals(run, generation_range, "results", as_generation_dict=False)
    
    values, _ = get_healthy_individuals_results(run, generation_range, as_generation_dict=False)
    meas_infos = get_meas_info(run)
    measurements = {}

    for measure, meas_info in meas_infos.items():
        min_boundary = meas_info.get('min_boundary', None)
        max_boundary = meas_info.get('max_boundary', None)
        
        measurements[measure] = _get_minmax_result_by_key(
            values,
            measure,
            min_boundary=min_boundary,
            max_boundary=max_boundary
        )

    return measurements
  
def get_healthy_individuals_results(run, generation_range=None, as_generation_dict=False): 
    """
    Retrieve healthy and unhealthy individuals' results.

    Args:
        run (str): The path of the ENAS run results directory.
        generation_range (tuple): A tuple specifying the range of generations to consider. Default is None, meaning all generations.
        as_generation_dict (bool): If True, returns dictionaries with generation keys containing individual results.
                                   If False, returns lists of individual results.
                                   Default is False.

    Returns:
        tuple or list: If as_generation_dict is True, returns a tuple containing dictionaries of healthy and unhealthy individuals' results.
                       If as_generation_dict is False, returns two lists of healthy and unhealthy individual results.
    """
    gen_results = get_individuals(run, generation_range, value="results", as_generation_dict=True)   
    measurements = list(get_meas_info(run).keys())
    
    healthy = {}
    unhealthy = {}
    
    for gen, results in gen_results.items():
        
        healthy[gen] = {}
        unhealthy[gen] = {}
        
        for ind, result in results.items():
            
            ### LOGIC FOR INDIVIDUAL HEALTH ###
            
            if "error" not in result or result["error"] == "False" or result["error"] == False:
                healthy[gen][ind] = result
                
            #elif result["error"] == "True" or result["error"] == True:
            else:
                unhealthy[gen][ind] = result
            
            ###################################
            
            
    healthy_list = []
    for gen, results in healthy.items():
        healthy_list += list(results.values())
                
    unhealthy_list = []
    for gen, results in unhealthy.items():
        unhealthy_list += list(results.values())
            
    if as_generation_dict:
        return healthy, unhealthy
        
    else: 
        healthy_list = []
        for gen, results in healthy.items():
            healthy_list += list(results.values())
                
        unhealthy_list = []
        for gen, results in unhealthy.items():
            unhealthy_list += list(results.values())
            
        return healthy_list, unhealthy_list
   
def get_best_individuals(run):
    """Get the individuals with the highest fitness value per generation.

    Args:
        run (str): The path of the ENAS run results directory.

    Returns:
        dict: Generation dictionnairy with dictionnairy of "individual", its "results" and "chromosome"
    """
    
    results = get_individuals(run, generation_range=None, value="results", as_generation_dict=True)
    chromosomes = get_individuals(run, generation_range=None, value="chromosome", as_generation_dict=True)
    
    best_individuals = {}
    
    for gen, individuals in results.items():
        
        best_fitness = 0
        best_ind = None
        best_result = None
        best_chromosome = None
        
        for ind, results in individuals.items():
            
            # TODO Why None
            if results is not None:
                ind_fitness = results["fitness"]
            
                if ind_fitness > best_fitness:
                    best_fitness = ind_fitness
                    best_ind = ind
                    best_result = results
                    best_chromosome = chromosomes[gen][ind]
        
        best_individuals[gen] = {
            "individual": best_ind,
            "results": best_result,
            "chromosome": best_chromosome,
        }
        
    return best_individuals


### GENES ###
def get_number_of_genes(run, generation, genename):
    """
    Get the number of genes in a certain generation.
    
    Args:
        run (str): The path of the ENAS run results directory.
        generation (int): Generation where genes will be counted.
        genename (str): The "layer" identifier of the a gene. 
        
    Returns:
        count (int): Number of genes
    """
    chromosomes = get_individuals(run, range(generation, generation+1), value="chromosome", as_generation_dict=False)

    count = 0
    for chromosome in chromosomes:
        if chromosome is not None:
            for gene in chromosome:

                if gene["layer"] == genename:
                    count += 1

    return count


### FAMILY TREE 
def _get_crossover_parents(run):
    """
    Retrieve information about crossover parents and their offspring.

    Args:
        run (str): The path of the ENAS run results directory.

    Returns:
        dict: A dictionary containing information about crossover parents and their offspring.
              Keys are the names of the new individuals, and values are dictionaries with the following keys:
              - 'generation': The generation number of the offspring.
              - 'parent1': The name of the first parent.
              - 'crossover1': The crossover point in the first parent's chromosome.
              - 'parent2': The name of the second parent.
              - 'crossover2': The crossover point in the second parent's chromosome.
    """
    df = pd.read_csv(f'{run}/crossover_parents.csv', header=None)
    df = df.rename(columns={0:"Generation", 1:"Parent 1", 2:"Parent 2", 3:"New Individual"})
    df = df.reindex(columns=["Generation", "New Individual", "Parent 1", "Crossover 1", "Parent 2", "Crossover 2"])

    crossover_dict = {}

    for idx, row in df.iterrows():
        individual = row["New Individual"].replace("New_Individual: ", "")

        generation = int(row["Generation"].replace("Generation: ", ""))
        parent1 = row["Parent 1"].replace("Parent_1: ", "").replace("(", "").replace(")", "")
        parent2 = row["Parent 2"].replace("Parent_2: ", "").replace("(", "").replace(")", "")

        parent1, crossover1 = parent1.split(",")[0], parent1.split(",")[1]
        parent2, crossover2 = parent2.split(",")[0], parent2.split(",")[1]

        crossover_dict[individual] = {
            "generation": generation,
            "parent1": parent1,
            "crossover1": crossover1, 
            "parent2": parent2,
            "crossover2": crossover2, 
        }

    return crossover_dict

def _get_crossover_parents_df(run):
    """
    Dataframe of parents of individuals (not in use)
    
    Args:
        run (str): The path of the ENAS run results directory.
        
    Returns:
        df (pandas.DataFrame): Dataframe with columns ["generation", "individual", "parent1", "crossover1", "parent2", "crossover2"]
    """

    df = pd.read_csv(f'{run}/crossover_parents.csv', header=None)
    df = df.rename(columns={0:"generation", 1:"parent1", 2:"parent2", 3:"individual"})
    df = df.reindex(columns=["generation", "individual", "parent1", "crossover1", "parent2", "crossover2"])

    df["generation"] = df["generation"].str.replace("Generation: ", "")
    df["parent1"] = df["parent1"].str.replace("Parent_1: ", "")
    df["parent2"] = df["parent2"].str.replace("Parent_2: ", "")
    df["individual"] = df["individual"].str.replace("New_Individual: ", "")

    df["generation"] = df["generation"].astype('int64')

    for idx, row in df.iterrows():
       
        p1 = row["parent1"].replace("(", "").replace(")", "")
        p2 = row["parent2"].replace("(", "").replace(")", "")

        df.loc[idx, "parent1"] = p1.split(",")[0]
        df.loc[idx, "crossover1"] = p1.split(",")[1]
        df.loc[idx, "parent2"] = p2.split(",")[0]
        df.loc[idx, "crossover2"] = p2.split(",")[1]

    return df 

def _get_upstream_tree(run, generation, individual, generation_range):
    """
    Helper funnction for family tree: Create upstream famliy tree with nodes, edges and root elements starting from selected individual.
    
    Args:
        run (str): The path of the ENAS run results directory.
        generation (int): Generation of individual
        individual (str): Individual from where family tree evolves from. 
        generation_range (range): A python range of generations from which the individuals will be extracted. 
        
    Returns:
        elements (list): List of nodes and edges for element param in cytoscape.
        root (list): List of roots nodes to create right tree structure.
    """
    
    min_generation = generation_range[0]

    # End condition with only one individual
    if min_generation == generation:
        return ([{'data': {'id': individual, 'label': individual[0:3], 'generation': generation, 'extinct': False}}], [individual])

    # Nodes and edges for individual
    crossovers = _get_crossover_parents(run)
    
    parent1 = crossovers[individual]["parent1"]
    crossover1 = crossovers[individual]["crossover1"]
    
    parent2 = crossovers[individual]["parent2"]
    crossover2 = crossovers[individual]["crossover2"]

    individual_el = [
        {'data': {'id': individual, 'label': individual[0:3], 'generation': generation, 'extinct': False}},
        {'data': {'source': parent1, 'target': individual, 'edgelabel': crossover1}},
        {'data': {'source': parent2, 'target': individual, 'edgelabel': crossover2}}
    ]

    # Recursion for moving up the tree
    parent1_tree, roots1 = _get_upstream_tree(run, generation-1, parent1, generation_range)
    parent2_tree, roots2 = _get_upstream_tree(run, generation-1, parent2, generation_range)

    return (parent1_tree + individual_el + parent2_tree, roots1 + roots2)

def _get_downstream_tree(run, generation, individual, generation_range):
    """
    Helper funnction for family tree: Create upstream famliy tree with nodes and edges starting from selected individual.
    
    Args:
        run (str): The path of the ENAS run results directory.
        generation (int): Generation of individual
        individual (str): Individual from where family tree evolves from. 
        generation_range (range): A python range of generations from which the individuals will be extracted. 
        
    Returns:
        elements (list): list of nodes and edges for element param in cytoscape.
    """

    max_generation = generation_range[-1]
    
    # Get children of individual
    crossovers = _get_crossover_parents_df(run)
    children = list(crossovers[crossovers['parent1'].str.contains(individual)]["individual"].values)
    crossover = list(crossovers[crossovers['parent1'].str.contains(individual)]["crossover1"].values)
    
    children += list(crossovers[crossovers['parent2'].str.contains(individual)]["individual"].values)
    crossover += list(crossovers[crossovers['parent2'].str.contains(individual)]["crossover2"].values)
    
    extinct = False if children else True

    # End condition with only one individual
    if max_generation == generation:
        return [{'data': {'id': individual, 'label': individual[0:3], 'generation': generation, 'extinct': extinct}}]

    # Nodes and edges for individual
    individual_el = [{'data': {'id': individual, 'label': individual[0:3], 'generation': generation, 'extinct': extinct}}]

    for idx, child in enumerate(children):
        individual_el.append({'data': {'source': individual, 'target': child, 'edgelabel': crossover[idx]}})

    # Recursion for moving down the tree
    children_tree = []

    for child in children: 
        children_tree += _get_downstream_tree(run, generation+1, child, generation_range)

    return children_tree + individual_el

def get_family_tree(run, generation, individual, generation_range=None):
    """
    Create famliy tree with nodes, edges and roots elements starting from selected individual.
    
    Args:
        run (str): The path of the ENAS run results directory.
        generation (int): Generation of individual
        individual (str): Individual from where family tree evolves from. 
        generation_range (range): A python range of generations from which the individuals will be extracted. 
        
    Returns:
        elements (list): list of nodes and edges for element param in cytoscape.
        root (list): List of roots nodes to create right tree structure.
    """

    # Set default value
    if generation_range is None:
        generation_range = range(generation-2, generation+1)

    # Get the entire tree without duplicates
    upstream_tree, roots = _get_upstream_tree(run, generation, individual, generation_range)
    downstream_tree = _get_downstream_tree(run, generation, individual, generation_range)

    family_tree = []
    unique_roots = []

    for el in downstream_tree + upstream_tree:
        if el not in family_tree: family_tree.append(el)

    for el in roots:
        if el not in unique_roots: unique_roots.append(el)
    
    return (family_tree, unique_roots)


### OTHER HELPER FUNCTIONS 
def get_random_individual(run, generation=None):
    """
    Get a random individual from a specified generation or from all generations if no generation is specified.

    Args:
        run (str): The path of the ENAS run results directory.
        generation (int, optional): The generation number from which to select the individual. If not specified, a random generation will be chosen. Defaults to None.

    Returns:
        tuple: A tuple containing the selected generation number and the random individual's name.
    """
    if generation is None:
        generations = get_generations(run)
        generation = random.choice(generations)
        generation = int(generation.split("_")[1])
    
    individuals = get_individuals(run, range(generation, generation+1), value="names", as_generation_dict=False)
    
    return generation, individuals[0]
