import dash
from dash import html, Input, Output, callback, dcc
import dash_mantine_components as dmc
import dash_cytoscape as cyto
import plotly.express as px
from dotenv import load_dotenv
import os
from evolution import get_number_of_genes, get_generations
from genepool import get_genepool
from components import parameter_card, warning
from dataval import validate_search_space

### LOAD PATH FROM ENVIRONMENT VARIABLES
load_dotenv()
run = os.getenv("RUN_RESULTS_PATH")


### REGISTER DASH APP
dash.register_page(__name__, path='/genepool')


### GENE POOL PAGE COMPONENTS
def cytoscape_stylesheet(groups):
    """
    Generates the stylesheet for the gene pool cytoscape component.

    Args:
        groups (list): List of gene groups for styling group nodes.

    Returns:
        list: Stylesheet for the cytoscape component.
    """
    cytoscape_stylesheet = [
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
                'width': '2px'
            }
        },
        {
            'selector': 'label',
            'style': {
                'font-family': 'sans-serif',
                'color': '#FFFFFF',
                'font-size': '12px',
                'font-weight': '500',
                'text-valign': 'center',
            }
        },
        {
            'selector': '[id = "Start"]',
            'style': {
                'background-color': '#FFFFFF',
                'border-color': '#6173E9',
                'border-width': '2px',
                'content': 'data(label)',
                'color': '#000000', # Font color
            }
        },
        {
            'selector': '[id = "end"]',
            'style': {
                'background-color': '#FFFFFF',
                'border-color': '#6173E9',
                'border-width': '2px',
                'content': 'data(label)',
                'color': '#000000', # Font color
            }
        },
        {
            'selector': '[id = "class-connect"]',
            'style': {
                'width': '2px',
                'line-color': '#6173E9',
            }
        },
    ]

    # Style for groupe nodes
    for group in groups:
    
        group_node_style = {
            'selector': f'[id = "{group}"]',
            'style': {
                #Box
                'background-fill': 'radial-gradient', 
                'background-gradient-stop-colors': '#EFEFEF #6173E9',
                'background-gradient-stop-positions': '0% 50%',
                'background-opacity': '0.5',
                'shape': 'round-rectangle',
                'border-color': '#6173E9',
                'border-width': '2px',
                'width':'400px',
                'height':'400px',
                #Font
                'color': '#6173E9',
                'text-background-color': '#FFFFFF',
                'text-background-shape': 'round-rectangle',
                'text-background-padding': '5px',
                'text-background-opacity': '1',
                'text-border-color':'#6173E9',
                'text-border-opacity':'1',
                'text-border-width': '1px',
                'text-valign': 'bottom',
                'line-height': '10',
                'font-size': '15px',
            }
        }

        cytoscape_stylesheet.append(group_node_style)
    
    return cytoscape_stylesheet

def cytoscape_search_space():
    """
    Generates the cytoscape component for the gene search space.

    Returns:
        dash_cytoscape.Cytoscape: Cytoscape component for gene search space.
    """
    
    elements, groups = get_genepool(run)
    stylesheet = cytoscape_stylesheet(groups)
    
    cytoscape = cyto.Cytoscape(
        id='cytoscape-genepool',
        elements=elements,
        style={'height': '600px', 'width': '100%'},
        stylesheet=stylesheet,
        layout={
            'name': 'cose',
        }
    )
    
    return cytoscape

def genepool_header():
    """
    Generates a header for the Gene Pool page.

    Returns:
        dash.html.H1: A dash HTML H1 element representing the header.
    """
    return html.H1('Gene Pool', id="genepool-header")

def gene_overview():
    """
    Generates an overview section for genes.

    Returns:
        dash.html.Div: A dash HTML Div element representing the gene overview section.
    """
    return html.Div(
        [
            html.Div([], id="gene-amount"),
            html.Div([], id="gene-name"),
            html.Div([], id="gene-type"),
            html.Img(src="assets/media/gene-overview-img.png", height="250px", id="gene-overview-img"),
        ],
        id="gene-overview-back"
    )

def gene_distribution_plot():
    """
    Generates the section for gene distribution plot updated on gene pool node click.

    Returns:
        dash.html.Div: Plot for gene distribution.
    """
    return html.Div([], id="number-of-genes-graph")

def gene_parameters():
    """
    Generates the section for gene parameters updated on gene pool node click.

    Returns:
        dash_mantine_components.Grid: Section for gene parameters.
    """
    return dmc.Grid(
        children=[],
        gutter="l",
        id="metric-cards-section",
        grow=True,
        style={"width": "100%"}
    )

def gene_insights():
    """
    Combines gene insights components for clicked gene in gene pool graph.

    Returns:
        dash_mantine_components.Stack: Combined gene insights components.
    """
    return dmc.Stack(
        [
            genepool_header(),
            gene_overview(),
            gene_distribution_plot(),
            gene_parameters(),
        ],
        align="flex-start",
        justify="flex-start",
        spacing="xl",
        style={"margin-right": "20px"}
    )


### GENE POOL CYTOSCAPE INTERACTION
@callback(
    Output('gene-name', 'children'), 
    Output('gene-amount', 'children'),
    Output('gene-type', 'children'),
    Output('metric-cards-section', 'children'), 
    Output('number-of-genes-graph', 'children'),
    Output('cytoscape-genepool', 'stylesheet'),
    
    Input('cytoscape-genepool', 'tapNodeData'))
def display_node_data(data):
    """
    Displays data for the clicked node in the cytoscape component.

    Args:
        data (dict): Data of the clicked node.

    Returns:
        tuple: Tuple containing gene name, gene amount, gene type, parameter cards, graph, and cytoscape stylesheet.
    """
    gene = data
    
    if data is None:
        gene = {'id': 'Start', 'label': 'Start', 'f_name': 'Start', 'layer': 'Start', 'parent': 'Starting point'}
    
    # Gene name
    gene_name = gene["f_name"]
    
    # Gene dimension
    gene_type = ""
    if "parent" in gene:
        gene_type = gene["parent"]
    
    # Metric cards
    parameter_cards = []
    for key, value in gene.items():
        
        not_metric = ["id", "label", "f_name", "layer", "parent", "exclude"]
        
        if key not in not_metric:
            
            mc = parameter_card(key, str(value), "mdi:input")
            parameter_cards.append(mc)   
            
    # Number of genes per generation
    numb_of_genes = []
    
    for generation in range(1, len(get_generations(run))+1):
        numb_of_genes.append(get_number_of_genes(run, generation, gene["layer"]))
    
    fig = px.bar(
        x = list(range(1, len(get_generations(run))+1)), 
        y = numb_of_genes, 
        labels={"x": "Generation", "y": f"{gene['layer']} layers"}
    )
    graph = dcc.Graph(figure=fig, id="gene-distribution-plot")
    
    fig.update_layout(
        plot_bgcolor='white',
        margin=dict(l=20, r=20, t=20, b=20),
        paper_bgcolor="white",
        font_color="black",
    )
    fig.update_xaxes(
        title_standoff=10
    )
    fig.update_yaxes(
        title_standoff=10
    )
    
    # Amount of genes
    gene_amount = 0
    
    for numb_gen in numb_of_genes:
        gene_amount += numb_gen
        
    gene_amount = f"{gene_amount} Count"
            
    # New node style when node is clicked
    _, groups = get_genepool(run)
    cytoscape_stylesheet_copy = cytoscape_stylesheet(groups)
    
    cytoscape_stylesheet_copy.append({
        'selector': f'[id = "{gene["layer"]}"]',
        'style': {
            'background-color':  '#6173E9',
            'border-color': '#FFFFFF',
            'border-width': '2px',
            'content': 'data(label)',
            'color': '#FFFFFF',
        }})
    
    cytoscape_stylesheet_copy.append({
        'selector': f'.{gene["layer"]}',
        'style': {
            'line-color': '#FFFFFF',
        }})
    
    return gene_name, gene_amount, gene_type, parameter_cards, graph, cytoscape_stylesheet_copy


### GENE POOL PAGE LAYOUT
def genepool_layout():
    """
    Generates the real-time layout for the gene pool page.
    If the search space data fails validation, it displays a warning message.

    Returns:
        dash_mantine_components.Grid: Layout for the gene pool page.
    """
    validation_result = validate_search_space(run)
    layout = None
    
    if validation_result:
        layout=html.Div(
            children=[
                genepool_header(),
                warning(validation_result)
            ]
        )
        print(validation_result)
        
    else:
        layout = dmc.Grid(
            children=[
                dmc.Col(gene_insights(), span='auto', style={ 'min-width': '525px'}),
                dmc.Col(cytoscape_search_space(), span='auto', style={ 'min-width': '525px'}),
            ],
            gutter="l",
        )
    
    return layout

layout = genepool_layout
