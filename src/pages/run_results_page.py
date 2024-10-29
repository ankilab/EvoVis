import dash
from dash import html, dcc
import dash_mantine_components as dmc
import plotly.graph_objects as go
import numpy as np
from dotenv import load_dotenv
import os
from components import dot_heading, bullet_chart_card_basic, parameter_card, chromosome_sequence, warning
from evolution import get_generations, get_meas_info, get_healthy_individuals_results, get_best_individuals, get_hyperparameters
from genepool import get_unique_gene_colors
from dataval import validate_generations_of_individuals, validate_meas_info


### LOAD PATH FROM ENVIRONMENT VARIABLES
load_dotenv()
run = os.getenv("RUN_RESULTS_PATH")


### REGISTER DASH APP
dash.register_page(__name__, path='/results')


### STYLES
grid_gutter = 'xl'
fitn_obj_height = 180


### HELPER FUNCTIONS FOR PLOTS ###
def add_meas_trace(fig, run, meas, generation_range=None, min=None, max=None, show_std=True, linecolor='#6173E9'):
    """
    Add a measurement trace to a Plotly figure.

    Args:
        fig (plotly.graph_objs.Figure): Plotly figure to which the measurement trace will be added.
        run (str): Path to the run results.
        meas (str): Measurement to be plotted.
        generation_range (tuple): Tuple containing the minimum and maximum generation.
        min (float): Minimum boundary for valid values.
        max (float): Maximum boundary for valid values.
        show_std (bool): Whether to show standard deviation.
        linecolor (str): Color of the measurement trace.

    Returns:
        None
    """
    
    measurements = get_meas_info(run)
    
    # Defining borders to exclude and include invalid values
    if min is None:
        min = -100000000000
    if max is None:
        max = 100000000000

    # Get the indivuals' information by sotring the valid values in arrays
    gen_results, _ = get_healthy_individuals_results(run, generation_range, as_generation_dict=True)
    
    generations = []
    avg_results = []
    std_results = []
    
    for gen, results in gen_results.items():
        
        values = []
        
        for ind, result in results.items():
            
            if result is not None and meas in result:
                value = result[meas]

                if (type(value) == int or type(value) == float) and (value <= max) and (value >= min): 
                    values.append(value)
            
        generations.append(gen)
        avg_results.append(np.mean(np.array(values)))
        std_results.append(np.std(np.array(values)))
    
    avg_results = np.array(avg_results)
    std_results = np.array(std_results)
    
    # Add standard deviation in background
    std_top = avg_results - std_results
    std_bottom = (avg_results + std_results)[::-1]
    
    if show_std:
        fig.add_trace(go.Scatter(
            x=np.concatenate([generations, generations[::-1]]),
            y=np.concatenate([std_top, std_bottom]),
            fill='toself',
            fillcolor='rgba(239,239,239,0.5)',
            line={'color':'rgba(239,239,239,0.5)'},
            name='Standard Deviation',
            hoverinfo='x+y'
        ))
    
    # Add mean in forground
    fig.add_trace(go.Scatter(
        x=generations,
        y=avg_results,
        mode='lines+markers',
        name=measurements[meas]["displayname"],
        line=go.scatter.Line(color=linecolor),
        hoverinfo='x+y'
    ))     
    
    # Set xaxis ticks to generations to avoid non int values
    fig.update_layout(
        xaxis={'tickvals': generations},
    )

def add_constraint_trace(fig, constraint):
    """
    Add a constraint trace to a Plotly figure.

    Args:
        fig (plotly.graph_objs.Figure): Plotly figure to which the constraint trace will be added.
        constraint (float): Value of the constraint.

    Returns:
        None
    """
    generations = get_generations(run, as_int=True)
    fig.add_trace(
        go.Scatter(
            x=generations, 
            y=[constraint] * len(generations), 
            name='Constraint',
            mode='lines', 
            line=go.scatter.Line(color='#B70202')
        )
    )

def figure_meas_over_gen(run, measures, generation_range=None, min=None, max=None, show_std=True, show_constraint=True, title=None, xaxis_title=None, yaxis_title=None):
    """
    Generate a Plotly figure showing measurement trends over generations.

    Args:
        run (str): Path to the run results.
        measures (str or list): Measurement(s) to be plotted.
        generation_range (tuple): Tuple containing the minimum and maximum generation values.
        min (float): Minimum boundary for valid values.
        max (float): Maximum boundary for valid values.
        show_std (bool): Whether to show standard deviation.
        show_constraint (bool): Whether to show constraint trace.
        title (str): Title of the figure.
        xaxis_title (str): Title of the x-axis.
        yaxis_title (str): Title of the y-axis.

    Returns:
        plotly.graph_objs.Figure: Plotly figure showing measurement trends over generations.
    """
    
    # Check if measures is a list or a string
    if type(measures) is str:
        measures = [measures]
    
    # Don't show standard deviation if more than one measure
    showlegend = False
    
    if len(measures) > 1:
        show_std=False
        show_constraint=False
        showlegend=True
        
    # Create figure
    fig = go.Figure()
    
    # Empty figure with no measures
    if len(measures) == 0:
        
        gens = list(range(1, 26))
        fig = go.Figure(go.Scatter(x=gens, mode='markers', marker=dict(color='rgba(0,0,0,0)')))
    
    # Measure traces
    else:   
        opacity_step = 1 / len(measures)
        for idx, meas in enumerate(measures):
        
            opacity = 1 - idx * opacity_step
            linecolor = f'rgba(97,115,233,{opacity})'
        
            add_meas_trace(fig, run, meas, generation_range, min, max, show_std, linecolor)
    
            # Constraint trace
            #if show_constraint:
            #    constraint = get_meas_info(run)[meas][1]
                
                #if constraint != None:
                #    add_constraint_trace(fig, constraint)
    
    # Layout
    t = 10 if title is None else 50
    
    fig.update_layout(
        title=title,
        title_font_color='#717171',
        title_font_size=15,
        title_font=dict(family='sans-serif'),
        xaxis={'title': xaxis_title, 'tickfont':{'color': '#D0D0D0'}, 'showline':True},
        yaxis={'title': yaxis_title, 'showgrid':True, 'gridcolor':'#D0D0D0', 'tickfont':{'color': '#D0D0D0'}},
        margin={'l': 10, 'b': 10, 't': t, 'r': 10},
        showlegend=showlegend,
        plot_bgcolor='rgba(0,0,0,0)',
        hovermode="x",
    )
    
    return fig

def graph_meas_over_gen(run, measures, generation_range=None, min=None, max=None, show_std=True, max_width=600, height=200, width=None, show_constraint=True, title=None, xaxis_title=None, yaxis_title=None, id="graph-meas-over-gen"):
    """
    Generate a Dash Graph component showing measurement trends over generations.

    Args:
        run (str): Path to the run results.
        measures (str or list): Measurement(s) to be plotted.
        generation_range (tuple): Tuple containing the minimum and maximum generation values.
        min (float): Minimum boundary for valid values.
        max (float): Maximum boundary for valid values.
        show_std (bool): Whether to show standard deviation.
        max_width (int): Maximum width of the graph.
        height (int): Height of the graph.
        width (int): Width of the graph.
        show_constraint (bool): Whether to show constraint trace.
        title (str): Title of the graph.
        xaxis_title (str): Title of the x-axis.
        yaxis_title (str): Title of the y-axis.
        id (str): ID of the graph component.

    Returns:
        dash_core_components.Graph: Dash Graph component showing measurement trends over generations.
    """
    
    fig = figure_meas_over_gen(run, measures, generation_range, min, max, show_std, show_constraint, title, xaxis_title, yaxis_title)
    
    graph_div = dcc.Graph(
        figure=fig, 
        style={'height': height, 'max-width': max_width, 'min-width': 200, 'width': width},
        id=id
    )
    
    return graph_div

def get_pareto_optimality_fig(run, generation_range=None, max_width=600, height=200):
    """
    Generate a Dash Graph component showing multi-objective mappingto identify pareto optimal neural architectures.

    Args:
        run (str): Path to the run results.
        generation_range (tuple): Tuple containing the minimum and maximum generation values.
        max_width (int): Maximum width of the graph.
        height (int): Height of the graph.

    Returns:
        dash_core_components.Graph: Dash Graph component showing  multi-objective mapping of architectures.
    """
    
    measurements = get_meas_info(run)
    
    # Font of axis
    tickfont = {
        'color': '#D0D0D0', 
        'family': 'sans-serif',
    }
    
    titlefont = {
        'family': 'sans-serif', 
        'color': '#D0D0D0',     
        'size': 12,            
    }
        
    # Get all fitness objectives
    fitness_objectives = []
    
    for measurement, rules in measurements.items():
        if rules["pareto-optimlity-plot"]:
            fitness_objectives.append(measurement)
      
    numb_fo = len(fitness_objectives)
    obj1, obj2, obj3 = None, None, None
    marker = None
    
    # Special Cases
    if numb_fo == 0 or numb_fo == 1:
        return None
    elif numb_fo > 3:
        fitness_objectives = fitness_objectives[:3]
    
    # Objectives
    results, _ = get_healthy_individuals_results(run, as_generation_dict=False)
    
    if numb_fo == 2:
        obj1 = [result[fitness_objectives[0]] for result in results]
        obj2 = [result[fitness_objectives[1]] for result in results]
        
    elif numb_fo == 3:
        obj1 = [result[fitness_objectives[0]] for result in results]
        obj2 = [result[fitness_objectives[1]] for result in results]
        obj3 = [result[fitness_objectives[2]] for result in results]
        
        custom_colorscale = [
            [0.0, '#ACB5ED'], 
            [0.2, '#7D8CEF'], 
            [0.4, '#5666CD'], 
            [0.6, '#293AAA'], 
            [0.8, '#001075'], 
            [1.0, '#000B51']  
        ] 
        
        marker = dict(
            size=3,
            color=obj3,
            colorscale=custom_colorscale,
            reversescale=False,
            symbol='x',
            colorbar={
                'title':f"{measurements[fitness_objectives[2]]['displayname']} [{measurements[fitness_objectives[2]]['unit']}]",
                'thickness': 10,
                'tickfont': tickfont,
                'titlefont': titlefont,
                #'title_standoff': 20,
                #'title_textangle': -90,
                'orientation': 'h',
                'outlinecolor': '#D0D0D0'
            }
        )
    
    # Add pareto optimality scatter plot
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=obj1,
        y=obj2,
        mode='markers',
        marker=marker
    ))
    
    # Update layout of figure
    fig.update_layout(
        title="", # Pareto optimality
        title_font_color='#717171',
        title_font_size=15,
        title_font=dict(family='sans-serif'),
        xaxis={'title': f"{measurements[fitness_objectives[0]]['displayname']} [{measurements[fitness_objectives[0]]['unit']}]", 'showline':True , 'tickfont': tickfont, 'title_font': titlefont},
        yaxis={'title': f"{measurements[fitness_objectives[1]]['displayname']} [{measurements[fitness_objectives[1]]['unit']}]", 'showgrid':True, 'gridcolor':'#D0D0D0', 'tickfont': tickfont, 'title_font': titlefont},
        coloraxis=dict(
            colorbar=dict(
                tickfont=tickfont,  # Adjust tick font color and size
                titlefont=titlefont,  # Adjust title font color and size
                thickness=15  # Adjust thickness of the color axis
            )
        ),
        margin={'l': 10, 'b': 0, 't': 0, 'r': 10},
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        hovermode="x",
    )
    
    # Add plot to a dash component
    graph_div = dcc.Graph(
        figure=fig, 
        style={'height': height, 'max-width': max_width},
    )
    
    return graph_div


### GENERAL RUN OVERVIEW ###
def general_overview():
    """
    Generate a Dash Grid component containing the count of healthy and unhealthy individuals, fitness plot, 
    mulit-objective individual mapping plot.

    Returns:
        dash_mantine_components.Grid: Dash Grid component containing general overview.
    """
    
    tot_gen = get_hyperparameters(run)["num_generations"]["value"]
    processed_gen = len(get_generations(run))
    healthy, unhealthy = get_healthy_individuals_results(run, as_generation_dict=False)
    
    gen_processed = bullet_chart_card_basic(processed_gen, 1, tot_gen, unit='Generations processed', info='Generations', back_color='#6173E9', bar_color='#A4B0FE', load_color='#FFFFFF', margin='0px', min_width='260px', flex='1')
    ind_healthy = parameter_card("Healthy Individuals", len(healthy), icon='icon-park-outline:health', margin='0px', width='100%')
    ind_unhealthy = parameter_card("Unhealthy Individuals", len(unhealthy), icon='mdi:robot-dead-outline', margin='0px', width='100%')
    fitness_plot = graph_meas_over_gen(run, 'fitness', generation_range=None, min=0, max=1, height=222.5, title="Fitness over generations", xaxis_title="", yaxis_title="")
    pareto_optimality_plot = get_pareto_optimality_fig(run, height=222.5)

    general_overview = dmc.Grid(
        [
            dmc.Col(dmc.Stack([gen_processed, ind_healthy, ind_unhealthy]), span="auto"),
            dmc.Col(fitness_plot, span="auto", className="col-results-page"), 
            dmc.Col(pareto_optimality_plot, span="auto", className="col-results-page"),
        ],
        gutter=grid_gutter,
        grow=True,
        justify='flex-start',
        className='general-overview'
    )
    
    return general_overview


### INDIVIDUAL RUN RESULTS PLOT ###
def objectives_overview():
    """
    Generate a Dash Grid component containing the overview of metrics over generations plots.

    Returns:
        dash_mantine_components.Grid: Dash Grid component containing overview of metrics over generations plots.
    """
    measurements = get_meas_info(run)
    objective_trends = []
    
    for measurement, meas_info in measurements.items():
        if meas_info.get("run-result-plot", True):
            
            heading = meas_info.get("displayname", measurement)
            
            if meas_info.get("unit", None):
                heading += f" [{meas_info.get('unit')}]"
            
            objective_trends.append(
                dmc.Col(
                    [
                        dot_heading(heading, style={"font-size": "14px"}, className='dot-heading-results-page'), 
                        graph_meas_over_gen(run, measurement, generation_range=None, min=meas_info.get("min-boundary", None), max=meas_info.get("max-boundary", None), height=fitn_obj_height, width=250)
                    ], 
                    className="col-results-page"
                )
            )
        
    objectives_overview = dmc.Grid(
        objective_trends,
        gutter=grid_gutter,
        grow=True,
        justify='flex-start'
    )
    
    return objectives_overview


### BEST INDIVIDUALS PLOT ###
def best_individuals_overview():
    """
    Generate a Dash Group component containing the overview of the best individuals chromosomes.

    Returns:
        dash_mantine_components.Group: Dash Group component containing best individuals chromosomes.
    """
    genomes = []
    best_individuals = get_best_individuals(run)
    unique_genes = get_unique_gene_colors(run)
    
    for gen, ind in best_individuals.items():
        
        if ind["individual"] is None:
            continue
        
        splits = ind["individual"].split("_")
        
        ind_abbrev = ""
        for split in splits:
            ind_abbrev += split[0].upper()
        
        ind_results = ind.copy()
        del ind_results["chromosome"]
        ind_results = str(ind_results)
        ind_results = ind_results.replace('{', '').replace('}', '').replace("'", '')
        
        ind_overview = html.Div(
            [
                dmc.Tooltip(
                    label=ind_results,
                    position="right",
                    offset=3,
                    transition="slide-up",
                    color='gray',
                    multiline=True,
                    width="300px",
                    children=[dmc.Avatar(ind_abbrev, radius="xl", style={"color": "#000000", "background-color": "#FFFFFF", "margin": "5px"})]
                ),
                html.P(f"GEN {gen}", style={"margin": "5px", "font-weight": "bold", "font-size": "11px"}),
                
                chromosome_sequence(ind["chromosome"], justify="flex-start", align="center", compromised=True, unique_genes=unique_genes),
            ],
            className="best-individual"
        )
        genomes.append(ind_overview)

    genomes_div = dmc.Group(
        genomes, 
        position='left', 
        align='start', 
        style={"gap":"5px"}
    )
    
    return genomes_div


### PAGE LAYOUT ###
def performance_plots_div():
    """
    Generate a Dash Div component containing performance plots.

    Returns:
        dash_html_components.Div: Dash Div component containing performance plots.
    """
    return html.Div(
        children=[
            html.H1("Run Result Plots", style={'margin-bottom': '25px', 'margin-top': '25px'}),
            general_overview(),
            objectives_overview(),
        ]
    )

def best_individuals_div():
    """
    Generate a Dash Div component containing chromsomes of the best individuals.

    Returns:
        dash_html_components.Div: Dash Div component containing chromsomes of the best individuals.
    """
    return html.Div(
        children=[
            html.H1("Fittest Individuals", style={'margin-bottom': '25px', 'margin-top': '25px'}),
            best_individuals_overview()
        ]
    )

def run_results_layout():
    """
    Generate the layout for the run results page.

    Returns:
        dash_mantine_components.Tabs: Layout for the run results page.
    """
    validation_result = validate_generations_of_individuals(run) + validate_meas_info(run)
    layout = None
    
    if validation_result:
        layout=html.Div(
            children=[
                html.H1("Run Results", style={'margin-bottom': '25px', 'margin-top': '25px'}),
                warning(validation_result)
            ]
        )
        print(validation_result)
        
    else:
        layout = dmc.Tabs(
            [
                dmc.TabsList(
                    [   
                        dmc.Tab("Run results plots", value="plots"),
                        dmc.Tab("Fittest individuals", value="best-individuals"),
                    ]
                ),
                dmc.TabsPanel(performance_plots_div(), value="plots"),
                dmc.TabsPanel(best_individuals_div(), value="best-individuals"),
            ],
            color="indigo",
            orientation="horizontal",
            variant="default",
            value="plots"
        )  
    
    return layout


layout = run_results_layout