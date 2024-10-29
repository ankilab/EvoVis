import json
from matplotlib.colors import hex2color, rgb2hex
import numpy as np


##########################################################################################

# MODULE SEARCH SPACE

# The Search Space Module provides functionalities for defining, managing, and exploring 
# search spaces in the context of evolutionary neural network architectures.
# It offers tools to represent the possible configurations, constraints, and relationships 
# between components within a predefined solution space.

###########################################################################################


### READ DATA FROM SEARCH SPACE JSON ###
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

def _get_search_space(run):
    """
    Retrieve the search space configuration from a JSON file for a specific run.

    Parameters:
        run (str): The path of the ENAS run results directory.

    Returns:
        dict: A Python dictionary representing the search space configuration for the specified run.

    Raises:
        FileNotFoundError: If the search_space.json file for the given run is not found.
        json.JSONDecodeError: If there is an issue decoding the JSON data in the search_space.json file.

    Example:
    >>> search_space = _get_search_space('evonas_run')
    """
    return _json_to_dict(f"{run}/search_space.json")

def _get_groups(run):
    """
    Extract layer identifiers from the 'gene_pool' section of the search space.

    Args:
        run (str): The path of the ENAS run results directory.
        
    Returns:
        dict: A dictionary mapping group types to lists of layer identifiers.
    """
    search_space = _get_search_space(run)
    
    layer_dict = {}
    
    for key, values in search_space.get("gene_pool", []).items():
        
        layer_identifiers = [val['layer'] for val in values]
        layer_dict[key] = layer_identifiers
        
    return layer_dict

def _get_genes_flattened(run):
    """
    Flatten the genes in the search space by adding the group information to each gene.

    Args:
        run (str): The path of the ENAS run results directory.

    Returns:
        list: A list of dictionaries representing flattened genes with group information.
    """
    
    search_space = _get_search_space(run)
    groups = search_space.get("gene_pool", [])
    
    layers = []
    
    for group, group_layers in groups.items():
        
        for layer in group_layers:
            layer_with_group = layer.copy()
            layer_with_group['group'] = group
            layers.append(layer_with_group)
            
    return layers
  
    
### GRAPH CREATED FROM RULESETS ###         
def _get_layer_graph(run, group_connections=True):
    """
    Retrieves the layer graph based on the search space defined for a given run.

    Args:
        run (str): The path of the ENAS run results directory.
        group_connections (bool, optional): If True, includes group connections in the graph. 
            Defaults to True.

    Returns:
        dict: A dictionary representing the layer graph. Keys are source layers, 
              and values are lists of target layers connected to the source layer.

    Raises:
        KeyError: If the expected keys ('rule_set', 'rule', 'layer', 'rule_set_groups', 'group') are not present in the search space data.
        TypeError: If the data structure of the search space is not as expected.

    Example:
        >>> _get_layer_graph("run_123")
        {'STFT_2D': ['MAG_2D'], 'MAG_2D': ['FB_2D'], 'FB_2D': ['C_2D', 'DC_2D', 'MAG2DEC_2D'], ...}
    """
    try:
        # Read search space from JSON
        search_space = _get_search_space(run)

        # Store layer connections in dictionary
        graph = {}

        # Process layers connections
        for src_layer in search_space.get("rule_set", []):
                     
                # Identify target layers
                target_layers = search_space["rule_set"].get(src_layer).get("rule", [])
                graph[src_layer] = target_layers
        
        # Return graph without group connections
        if not group_connections:
            return graph       

        # Process group connections
        for group_rule in search_space.get("rule_set_group", []):
            
            # Identify source and target group
            for target_group in group_rule.get("rule", []):
                source_groups = group_rule.get("group", [])

                # Identify source and target layers in source and target groups
                groups = _get_groups(run)
                source_layers = groups.get(source_groups, [])
                target_layers = groups.get(target_group, [])

                for src_layer in source_layers:
                    if src_layer in graph:
                        graph[src_layer] += target_layers
                    else:
                        graph[src_layer] = target_layers
        
        return graph

    except KeyError as key_error:
        raise KeyError(f"Expected key not found in search space data: {key_error}")

    except TypeError as type_error:
        raise TypeError(f"Unexpected data structure in search space data: {type_error}")

def _get_group_graph(run):
    """
    Builds the group graph based on the search space defined for a given run.

    Args:
        run (str): The path of the ENAS run results directory.

    Returns:
        dict: A dictionary representing the group graph. Keys are source groups, 
              and values are lists of target groups connected to the source group.

    Raises:
        KeyError: If the expected keys ('rule_set_groups', 'rule', 'group') are not present in the search space data.
        TypeError: If the data structure of the search space is not as expected.

    Example:
        >>> _get_group_graph("run_123")
        {'Feature Extraction 1D': ['Global Pooling 1D'], 'Feature Extraction 2D': ['Global Pooling 2D']}
    """
    try:
        # Read search space from JSON
        search_space = _get_search_space(run)

        # Store groups connections in dictionary
        group_graph = {}

        # Process group connections
        for group_rule in search_space.get("rule_set_group", []):
            excluded = group_rule.get("exclude", False)

            if not excluded:
                # Identify source and target group
                source_group = group_rule["group"]
                tar_get_groups = group_rule.get("rule", [])
                group_graph[source_group] = tar_get_groups

        return group_graph

    except KeyError as key_error:
        raise KeyError(f"Expected key not found in search space data: {key_error}")

    except TypeError as type_error:
        raise TypeError(f"Unexpected data structure in search space data: {type_error}")
 
 
### CONNECTED LAYERS ### 
def _dfs(graph, layer, visited, result):
    """
    Perform depth-first search (_DFS) on the given graph starting from the specified layer.

    Args:
        graph (dict): A dictionary representing the layer graph.
        layer (str): The starting layer for DFS.
        visited (set): A set to keep track of visited layers.
        result (list): A list to store the layers visited in DFS order.

    Returns:
        None

    Example:
        >>> graph = {'A': ['B', 'C'], 'B': ['D'], 'C': ['E'], 'D': [], 'E': []}
        >>> visited = set()
        >>> result = []
        >>> _dfs(graph, 'A', visited, result)
        >>> print(result)
        ['A', 'B', 'D', 'C', 'E']
    """
    if layer not in visited:
        visited.add(layer)
        result.append(layer)

        for neighbor in graph.get(layer, []):
            _dfs(graph, neighbor, visited, result)

def _get_connected_layers(run, start_layer="Start"):
    """
    Retrieve layers connected to the specified starting layer in the layer graph for a given run.

    Args:
        run (str): The path of the ENAS run results directory.
        start_layer (str): The layer from which to start exploring connected layers.

    Returns:
        list: A list of layers connected to the starting layer.

    Raises:
        ValueError: If the specified start_layer is not found in the layer graph.

    Example:
        >>> _get_connected_layers('run_123', 'STFT_2D')
        ['STFT_2D', 'MAG_2D', 'FB_2D', 'C_2D', 'DC_2D', 'MAG2DEC_2D', ...]
    """
    try:
        graph = _get_layer_graph(run, group_connections=True)
        if start_layer not in graph:
            raise ValueError(f"The specified start_layer '{start_layer}' is not found in the layer graph.")

        visited = set()
        result = []

        _dfs(graph, start_layer, visited, result)
        
        return result

    except KeyError as key_error:
        raise KeyError(f"Expected key not found in layer graph data: {key_error}")

    except TypeError as type_error:
        raise TypeError(f"Unexpected data structure in layer graph data: {type_error}")


### DASH CYTOSCAPE FORMAT ###
def _get_node_element(gene):
    """
    Create a node element from a gene, representing a layer or group.

    Args:
        gene (dict): A dictionary representing a gene.

    Returns:
        dict: A dictionary representing a node element with appropriate attributes.

    Example:
        >>> gene = {'layer': 'STFT_2D', 'group': 'Feature Extraction 2D', 'exclude': False}
        >>> _get_node_element(gene)
        {'id': 'STFT_2D', 'label': 'STFT 2D', 'parent': 'Feature Extraction 2D', 'layer': 'STFT_2D'}
    """
    layer = gene["layer"]
    node = gene.copy()

    # Assigning ID and label based on the layer
    node["id"] = layer
    node["label"] = layer.replace("_", " ")

    # Group is optional
    if "group" in gene:
        # If group is present, move it to 'parent' and remove it from the node itself
        group = node.pop("group")
        node["parent"] = group

    return node

def get_genepool(run):
    """
    Create Cytoscape elements representing layers and group connections in the search space for a given run.

    Args:
        run (str): The path of the ENAS run results directory.
        
    Returns:
        tuple: A tuple containing a list of Cytoscape elements and a list of unique group names.

    Raises:
        KeyError: If the expected keys ('gene_pool') are not present in the search space data.
        TypeError: If the data structure of the search space is not as expected.

    Example:
        >>> get_genepool('run_123')
        ([{'data': {'id': 'Start', 'label': 'Start', 'f_name': 'Start', 'layer': 'Start'}},
          {'data': {'id': 'Feature Extraction 2D', 'label': 'Feature Extraction 2D', 'parent': 'Feature Extraction 2D'}},
          ...
         ],
         ['Feature Extraction 2D', ...])
    """
    try:
        # Initialize elements with a start node
        elements = [{'data': {'id': 'Start', 'label': 'Start', 'f_name': 'Start', 'layer': 'Start'}}]
        group_elements = []
        groups = []

        # Use connected layers to create nodes of layers and groups
        start_layer = "Start"
        connected_layers = _get_connected_layers(run, start_layer)
        genes = _get_genes_flattened(run)

        for gene in genes:
            layer = gene.get("layer")
            excluded = gene.get("exclude", False)

            if not excluded and layer in connected_layers:
                
                # Add layer node
                layer_data = _get_node_element(gene)
                element = {"data": layer_data}

                if element not in elements:
                    elements.append(element)

                # Add group node
                group = gene.get("group")
                if group:
                    group_data = {'id': group, 'label': group}
                    element = {"data": group_data}

                    if element not in elements:
                        group_elements.append(element)
                        groups.append(group)

        # Combine group elements with layer elements
        elements = group_elements + elements

        # Build layer connections
        layer_graph = _get_layer_graph(run, group_connections=False)

        for layer, edges in layer_graph.items():
            for edge in edges:
                element = {'data': {'source': layer, 'target': edge}, 'classes': f'{layer} {edge}'}

                if element not in elements and layer in connected_layers:
                    elements.append(element)

        # Build group connections
        group_graph = _get_group_graph(run)

        for group_source, group_targets in group_graph.items():
            for group_target in group_targets:
                element = {'data': {'source': group_source, 'target': group_target}, 'classes': 'class-connect'}

                if element not in elements and group_source in groups:
                    elements.append(element)

        return elements, groups

    except KeyError as key_error:
        raise KeyError(f"Expected key not found in search space data: {key_error}")

    except TypeError as type_error:
        raise TypeError(f"Unexpected data structure in search space data: {type_error}")
    

### UNIQUE GENES WITH COLORS ###
def _generate_color_scale(start_color, end_color, num_colors):
    """
    Generate a color scale between two given colors.

    Args:
        start_color (str): The starting color in hexadecimal format.
        end_color (str): The ending color in hexadecimal format.
        num_colors (int): The number of colors in the scale.

    Returns:
        list: A list of hexadecimal color codes representing the color scale.
    """
    start_rgb = hex2color(start_color)
    end_rgb = hex2color(end_color)

    r = np.linspace(start_rgb[0], end_rgb[0], num_colors)
    g = np.linspace(start_rgb[1], end_rgb[1], num_colors)
    b = np.linspace(start_rgb[2], end_rgb[2], num_colors)

    color_scale = [rgb2hex((r[i], g[i], b[i])) for i in range(num_colors)]
    return color_scale

def get_unique_gene_colors(run):
    """
    Get unique colors for each gene layer in a run.

    Args:
        run (str): The path of the ENAS run results directory.

    Returns:
        dict: A dictionary where keys are unique gene layers and values are unique colors in hexadecimal format.
    """
    connected_genes = _get_connected_layers(run)
    unique_genes = {connected_gene: "" for connected_gene in connected_genes}
    
    opacity_step = 1 / len(unique_genes)
    
    start_color = '#6173E9'
    end_color = '#B70202'
    
    num_colors = len(unique_genes)
    color_scale = _generate_color_scale(start_color, end_color, num_colors)

    for idx, gene in enumerate(unique_genes):
        
        opacity = 1 - idx * opacity_step
        unique_genes[gene] = color_scale[idx]

    return unique_genes
