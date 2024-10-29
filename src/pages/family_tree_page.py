import dash
from dash import html, callback, Input, Output, dcc
import dash_cytoscape as cyto
import dash_mantine_components as dmc
from dash_iconify import DashIconify
from dotenv import load_dotenv
import os
from evolution import get_family_tree, get_generations, get_individuals, get_random_individual, get_individuals_min_max, get_individual_result, get_individual_chromosome, get_meas_info
from components import dot_heading, bullet_chart_card, bullet_chart_card_basic, warning, information, chromosome_sequence
from dataval import validate_generations_of_individuals, validate_crossover_parents, validate_meas_info, validate_individual_chromosome, validate_individual_result

### LOAD PATH FROM ENVIRONMENT VARIABLES
load_dotenv()
run = os.getenv("RUN_RESULTS_PATH")


### REGISTER DASH APP
dash.register_page(__name__, path='/family-tree')


### STYLES
CYTOSCAPE_STYLE = [
    {
        'selector': 'node',
        'style': {
            'background-color': '#6173E9',
            'content': 'data(label)',
            'width':'50px',
            'height':'50px',
        }
    },
    {
        'selector': 'edge',
        'style': {
            'line-color': '#6173E9',
            'label': 'data(edgelabel)',
        }
    },
    {
        'selector': 'label',
        'style': {
            'font-family': 'sans-serif',
            'color': '#FFFFFF',
            'font-size': '12px',
            'font-weight': 'bold',
            'text-valign': 'center',
        }
    },
    {
        'selector': 'edgelabel',
        'style': {
            'font-family': 'sans-serif',
            'color': '#FFFFFF',
            'font-size': '12px',
            'font-weight': 'bold',
            'text-valign': 'center',
            'text-background-color': '#6173E9',
            'text-background-opacity': '0.5',
            'text-background-shape': 'round-rectangle',
            'text-background-padding': '5px'
        }
    },
]
MARKS_STYLE = {
    'color': '#6173E9', 
    'background-color': '#D1D6F8', 
    'padding': '2px', 
    'border-radius': '3px',
    'font-size': '12px',
    'font-weight': '300',
}


### FAMILY TREE COMPONENTS 
def family_tree_cytsocape():
    """
    Generates a Dash Cytoscape component representing the family tree.

    Returns:
        dash_cytoscape.Cytoscape: Dash Cytoscape component representing the family tree.
    """
    return cyto.Cytoscape(
        id='cytoscape-family-tree',
        className="wrapper",
        style={'height': '450px', 'max-width': '100%'},
        stylesheet=CYTOSCAPE_STYLE,
    )

def generation_slider():
    """
    Generates a Dash RangeSlider component for selecting generations.

    Returns:
        dash_core_components.RangeSlider: Dash RangeSlider component.
    """
    generations_int = get_generations(run, as_int=True)
    random_generation, _ = get_random_individual(run, generations_int[round(len(generations_int)/3)])
    
    return dcc.RangeSlider(
        min(generations_int), 
        max(generations_int), 
        1, 
        marks={
            min(generations_int): {"label": f"Generation_{min(generations_int)}", "style": MARKS_STYLE}, 
            max(generations_int): {"label": f"Generation_{max(generations_int)}", "style": MARKS_STYLE}
        }, 
        #pushable=1, 
        allowCross=False,
        id='gen-range-slider',
        tooltip={"placement": "top", "always_visible": True},
        value=[random_generation-4, random_generation, random_generation+1], 
    )

def individual_select(): 
    """
    Generates a Dash Mantine Select component for selecting individuals.

    Returns:
        dash_mantine_components.Select: Dash Mantine Select component.
    """
    return dmc.Select(
        label="Select Individual",
        placeholder="Select Individual",
        icon=DashIconify(icon="material-symbols-light:circle", height=10, width=10, color="#6173E9"),
        id="ind-select",
        className="circle-select",
    )


### FAMILY TREE MODIFICATION CALLBACKS 
@callback( Output("ind-select", "data"), Output("ind-select", "value"), Input("gen-range-slider", "value"))
def set_individuals_select(gen_range):
    """
    Sets the options and default value for the individual selection dropdown based on the selected generation range.

    Args:
        gen_range (list): List containing the selected generation (idx 1) and minimum (idx 0), maximum (idx 2) generation values selected on the RangeSlider.

    Returns:
        list: Data options for the individual selection dropdown.
        str: Default value for the individual selection dropdown.
    """
    gen = gen_range[1]
    
    data = [{"value": ind, "label": ind} for ind in get_individuals(run, generation_range=range(gen, gen+1), value="names", as_generation_dict=False)]
    gen, value = get_random_individual(run, generation=gen)
    
    return data, value

@callback( Output("cytoscape-family-tree", "elements"), Output("cytoscape-family-tree", "layout"), Output("cytoscape-family-tree", "stylesheet"), Input("gen-range-slider", "value"), Input("ind-select", "value"), Input("cytoscape-family-tree", "tapNodeData"), Input("cytoscape-family-tree", "tapEdgeData"))
def set_cytoscape(gen_range, ind, ind_clicked, edge_clicked):
    """
    Sets the elements, layout, and stylesheet for the family tree visualization based on user interactions.

    Args:
        gen_range (list): List containing the selected generation (idx 1) and minimum (idx 0), maximum (idx 2) generation values selected on the RangeSlider.
        ind (str): Selected individual from the dropdown.
        ind_clicked (dict): Data of the individual node clicked on the Cytoscape component.
        edge_clicked (dict): Data of the edge clicked on the Cytoscape component.

    Returns:
        list: Nodes and edges of Cytoscape component.
        dict: Layout configuration for the Cytoscape component.
        list: Stylesheet for the Cytoscape component.
    """
    
    # Get Family tree through individual selection
    generation_range = range(gen_range[0], gen_range[2]+1)
    gen = gen_range[1]
    elements, roots = get_family_tree(run, gen, ind, generation_range)
    
    cytoscape_layout = {
        'name': 'breadthfirst', 
        'roots': roots
    }
    
    # Creating new stylesheet with white selected node
    new_cytoscape_style = CYTOSCAPE_STYLE.copy()

    # Add styling for 
    if ind is not None: 
        new_cytoscape_style.append({
            'selector': f'[id = "{ind}"]',
            'style': {
                'background-color': '#6173E9',
                'border-color': '#FFFFFF',
                'border-width': '3px',
                'content': 'data(label)',
                'color': '#FFFFFF',
            }, 
        })
    
    if ind_clicked is not None: 
        ind_clicked_id = ind_clicked['id']
        
        new_cytoscape_style.append({
            'selector': f'[id = "{ind_clicked_id}"]',
            'style': {
                'background-color': '#FFFFFF',
                'border-color': '#6173E9',
                'border-width': '3px',
                'content': 'data(label)',
                'color': '#000000',
            }
        })   
        
    if edge_clicked is not None: 
        edge_id = edge_clicked['id']
        source = edge_clicked['source']
        target = edge_clicked['target']

        new_cytoscape_style.append({
            'selector': f'[id = "{source}"]',
            'style': {
                'background-color': '#FFFFFF',
                'border-color': '#6173E9',
                'border-width': '3px',
                'content': 'data(label)',
                'color': '#000000',
            }
        }) 
        
        new_cytoscape_style.append({
            'selector': f'[id = "{target}"]',
            'style': {
                'background-color': '#FFFFFF',
                'border-color': '#6173E9',
                'border-width': '3px',
                'content': 'data(label)',
                'color': '#000000',
            }
        }) 
    
    return elements, cytoscape_layout, new_cytoscape_style

@callback( 
    Output("individual-heading", "children"),  Output("individual-exceptions", "children"), Output("individual-genes", "children"), Output("individual-results", "children"), 
    Input("cytoscape-family-tree", "tapNodeData"), Input("ind-select", "value"), Input("gen-range-slider", "value"))
def set_values(ind_clicked, ind_select, gen_range):
    """
    Sets the information to be displayed about the selected individual.

    Args:
        ind_clicked (dict): Data of the individual node clicked on the Cytoscape component.
        ind_select (str): Selected individual from the dropdown.
        gen_range (tuple): Tuple containing the minimum and maximum generation values selected on the RangeSlider.

    Returns:
        list: Name of the selected individual.
        list: Exception information about the selected individual.
        list: Genes of the selected individual.
        list: Metrics of the selected individual.
    """
    
    # Individual selected in cytoscape
    ind = None
    gen = None
    extinct = None
    
    if ind_clicked is None: 
        ind = ind_select
        gen = gen_range[1]
        extinct = False
    
    else:
        ind = ind_clicked["id"]
        gen = ind_clicked["generation"]
        extinct = ind_clicked["extinct"]
        
    # Get individual information from selected node
    ind_meas = get_individual_result(run, gen, ind)
    ind_genome = get_individual_chromosome(run, gen, ind)
        
    ### 1 HEADING ###
    ind_heading = [html.H2(ind, style = {'margin': '10px'})]
    
    ### 2 EXCEPTIONS ###
    ind_exceptions = []

    if extinct:
        ind_exceptions.append(information("Individual became extinct."))

    if "error" in ind_meas and (ind_meas["error"] == "True" or ind_meas["error"] == True):
        ind_exceptions.append(warning("Individual errored."))
        
    val_msg = validate_individual_result(run, gen, ind) + validate_individual_chromosome(run, gen, ind)
    
    if val_msg:
        ind_exceptions.append(warning(val_msg))
        return ind_heading, ind_exceptions, None, None
        

    ### 3 CHROMOSOME ###
    ind_genes = [dot_heading("Genes", style={"margin": "10px",'flex': '100%'}), chromosome_sequence(chromosome=ind_genome)]
    
    ### 4 RESULTS ###
    ind_fitness = [dot_heading("Fitness", style={"margin": "10px",'flex': '100%'})]
    
    if "fitness" in ind_meas:
        ind_fitness += [bullet_chart_card_basic(ind_meas['fitness'], 0, 1, metric_card_id="bullet-chart-basic")]

    border_meas = get_individuals_min_max(run, generation_range=None)
    meas_info = get_meas_info(run)
    del meas_info['fitness']
    
    for meas_key in list(meas_info.keys()):
        if meas_key in ind_meas and meas_info[meas_key]["individual-info-plot"]:
            
            if isinstance(ind_meas[meas_key], (float, int)):
                
                ind_fitness += [(
                    bullet_chart_card(
                        meas_info[meas_key]["displayname"], 
                        meas_info[meas_key]["individual-info-img"], 
                        ind_meas[meas_key], 
                        border_meas[meas_key][0], 
                        border_meas[meas_key][1], 
                        unit=meas_info[meas_key]["unit"], 
                        constraint=None, 
                        metric_card_id="bullet-chart-card"
                    )
                )]
            
    return ind_heading, ind_exceptions, ind_genes, ind_fitness


### FAMILY TREE PAGE LAYOUT  
def family_tree_header():
    """
    Generates the header for the family tree page.

    Returns:
        dash_html_components.H1: Header for the family tree page.
    """
    return html.H1('Family Tree', style = {"margin-bottom": "20px", "margin-top": "20px"})

def family_tree():
    """
    Generates the layout for the family tree page.

    Returns:
        dash_mantine_components.Col: Column layout for the family tree page.
    """
    return dmc.Col(html.Div(
        [ 
            family_tree_header(), 
            individual_select(), 
            family_tree_cytsocape(),
            generation_slider()
        ]), 
        span='auto',
        style={'max-width': '100%'} 
    )
    
def individual_information():
    """
    Generates the layout for individual information display.

    Returns:
        dash_mantine_components.Col: Column layout for individual information display.
    """
    return dmc.Col(
        [
            html.Div([], id='individual-heading'), 
            html.Div([], id='individual-exceptions'), 
            dmc.Grid(
                [
                    dmc.Col([html.Div([], id='individual-results')], span=10),
                    dmc.Col([html.Div([], id='individual-genes')], span='auto')
                ],
                gutter="xs",
                grow=True
            )
        ], 
        span=2, 
        className='cytoscape-values', 
        id='values-col'
    )

def family_tree_layout():
    """
    Generates the layout for the family tree page based on data validation.
    If data fails validation, it displays a warning message.

    Returns:
        dash_html_components.Div: Layout for the family tree page.
    """
    validation_result = validate_generations_of_individuals(run) + validate_crossover_parents(run)
    validation_result_ind_inf = validate_meas_info(run)
    layout = None
    
    if validation_result:
        layout=html.Div(
            children=[
                family_tree_header(),
                warning(validation_result)
            ]
        )
        print(validation_result)
        
    else:
        if validation_result_ind_inf:
            layout = dmc.Grid(
                children=[
                    family_tree(),
                    dmc.Col(html.Div([warning(validation_result_ind_inf)]), span='auto', style={'max-width': '100%'})  
                ],
                gutter="s",
                grow=True
            )
            print(validation_result)
            
        else:  
            layout = dmc.Grid(
                children=[
                    family_tree(),
                    individual_information()
                ],
                gutter="s",
                grow=True
            )
            
    return layout

layout = family_tree_layout