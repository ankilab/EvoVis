from dash import html
from dash_iconify import DashIconify
import plotly.graph_objects as go
import dash_mantine_components as dmc


####################################################################################

# MODULE COMPONENTS

# Module provides importable HTML components that are used across all EvoVis pages 
# to reduce code duplication and promote consistency in design.

####################################################################################


### HEADING ###
def dot_heading(heading, className='dot-heading', style=None):
    """
    Generate a heading with a purple dot in front.

    Args:
        heading (str): The text content of the heading.
        className (str, optional): CSS class name for the heading div. Defaults to 'dot-heading'.
        style (dict, optional): CSS styles for the heading div. Defaults to None.

    Returns:
        dash.html.Div: Div containing the dot heading.
    """
    heading_div = html.Div(
        [
            DashIconify(icon="material-symbols:circle", height=12, width=12, color="#6173E9", 
                style={ 
                    "display": "inline-block", 
                    "vertical-align":"center" 
                }),
            html.H4(heading, 
                style={ 
                    "display": "inline-block",
                    "margin": "5px", 
                    "vertical-align":"center" 
                })
        ],
        className=className,
        style=style
    )
    return heading_div


### METRIC ###
def parameter_card(metrictype, metric, icon=None, unit=None, description=None, min_width=280, max_width="100%", width="100%", margin="5px", metric_card_id="metric-card"):
    """
    Generate a card to display a metric parameter.

    Args:
        metrictype (str): The type of metric.
        metric (float): The value of the metric.
        icon (str, optional): The icon for the metric card button. Defaults to None.
        unit (str, optional): The unit of measurement for the metric. Defaults to None.
        description (str, optional): The description of the metric. Defaults to None.
        min_width (int or str, optional): The minimum width of the card. Defaults to 280.
        max_width (str, optional): The maximum width of the card. Defaults to "100%".
        width (str, optional): The width of the card. Defaults to "100%".
        margin (str, optional): The margin around the card. Defaults to "5px".
        metric_card_id (str, optional): The id of the metric card. Defaults to "metric-card".

    Returns:
        dash.html.Div: Card containing the metric parameter.
    """
    
    # Styles
    parameter_card_style = {
        'max-width': f"{max_width}px",
        'width': f"{width}px",
        'min-width': f"{min_width}px",
        'display': 'inline-flex',
        'margin': margin,
        'vertical-align': 'center',
        'background-color': '#FFFFFF',
        'border-radius': '20px',
        'align-items': 'center',
        'justify-content': 'flex-start',
        'box-shadow': '3px 3px 10px 1px rgba(0, 0, 0, 0.05)',
        'flex-grow': '1',
    }

    text_block_style = {
        "margin": "5px", 
        "vertical-align":"center",
        "background-color": "#FFFFFF",
    } 
    
    metric_type_style = {
        "margin": "5px", 
        "font-weight": "lighter",
        "font-size": "15px", 
        "overflow": "auto", 
        "width": "200px",
        "height":"18px"
    }
    
    metric_style = { 
        "margin": "5px", 
        "overflow": "auto", 
        "width": "200px",
        "height":"19px"
    }
    
    # Button with and withoout tooltip depending if description is given
    button = html.Button(
        children=DashIconify(icon=icon, height=25, width=25, color="#000000"), 
        className="metric-btn",
        id=f"metric-btn-{metrictype}"
    )
    
    tooltip_button = dmc.Tooltip(
        label=description,
        position="bottom",
        offset=5,
        radius=15,
        transition="pop-top-left",
        color="#6173E9",
        multiline=True,
        width=280,
        children=[button]
    ) 
    
    button = button if description is None else tooltip_button

    # Metric card div of button, metrictype and metric
    parameter_card_div = html.Div([
        
        button,
        html.Div([   
            html.P(metrictype, style=metric_type_style, id=f"{metric_card_id}-label"),
            html.H4(metric if unit is None else f"{metric} {unit}", style=metric_style, id=f"{metric_card_id}-value")], 
            style= text_block_style)   
        ], 
        
        style = parameter_card_style, 
        id = metric_card_id
    )

    return parameter_card_div

def bullet_chart_card(metrictype, img, metric, min, max, constraint=None, unit=None, min_width='180px', metric_card_id="bullet-chart-card"):
    """
    Generate a card to display a metric as a bullet chart.

    Args:
        metrictype (str): The type of metric.
        img (str): The filename of the image associated with the metric.
        metric (float): The value of the metric.
        min (float): The minimum value of the metric range.
        max (float): The maximum value of the metric range.
        constraint (float, optional): The constraint value for the metric. Defaults to None.
        unit (str, optional): The unit of measurement for the metric. Defaults to None.
        min_width (str, optional): The minimum width of the card. Defaults to '180px'.
        metric_card_id (str, optional): The id of the bullet chart card. Defaults to "bullet-chart-card".

    Returns:
        dash.html.Div: Card containing the bullet chart.
    """
    metric = round(metric, 3)
    
    bullet_chart_card_style = {
        'border-radius': '20px',
        'border': '3px solid #FFFFFF',
        'box-shadow': '3px 3px 10px 1px rgba(0, 0, 0, 0.05)',
        'background': '#EFEFEF',
        'min-width': min_width,
        'margin': '10px',
        'margin-top': '30px',
        'flex': 1,
        
        'position': 'relative',
        'display': 'inline-flex',
        'justify-content': 'center',
    }
    
    bullet_chart_style = {
        'border-radius': '15px',
        'border': '1px solid #FFFFFF',
        'background': 'linear-gradient(180deg, rgba(217, 217, 217, 0) 0%, rgba(97, 115, 233, 0.40) 178.92%)',
        'backdrop-filter': 'blur(3px)',
        'width': '95%',
        'height': '50%',
        'margin': '3px',
        'margin-top': '20px',
        'text-align': 'center',
        'z-index': 0,
    }
    
    img_style = {
        'width': '70px',
        'height': '70px',
        'flex-shrink': 0,
        'top': '-30px',
        'position': 'absolute',
        'z-index': 1,
    }
    
    bar_style = {
        'border-radius': '30px',
        'background': '#FFFFFF',
        'position': 'relative',
        'height': 20,
        'margin': '5px',
    }
    
    range = max - min
    load_percent = ((metric - min) / range * 100) if range != 0 else 0
    load_percent = 100 if load_percent > 100 else load_percent
    
    load_style = {
        'border-radius': '30px',
        'background': '#6173E9',
        
        'position': 'absolute',
        'z-index': 0,
        'height': 20,
        'width': f'{load_percent}%',
        
    }
    
    constraint_style = None
    
    if constraint is not None:
        constraint_percent = ((constraint - min) / range * 100) if range != 0 else 0
        if constraint_percent > 100: constraint_percent = 100
        if constraint_percent < 0: constraint_percent = 0

        constraint_style = {
            'background': '#B70202',
            'border-radius': '3px',
        
            'position': 'absolute',
            'z-index': 1,
            'height': 20,
            'width': 3,
            'left': f'{constraint_percent}%',
        }
    
    fig = go.Figure(go.Indicator(
        mode = "number+gauge+delta",
        gauge={
            'shape': "bullet",
            'axis': {'visible': False},
            'bar': {'color': "lightblue", 'line': {'color': "darkblue", 'width': 2}},
        },
        value = 220,
        delta = {'reference': 300},
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': ""})
    )
    
    fig.update_layout(
        yaxis={'visible': False},
        yaxis_title_text='',
        yaxis_showgrid=False,
        yaxis_zeroline=False,
        yaxis_showticklabels=False,
        xaxis={'visible': False},
        xaxis_title_text='',
        xaxis_showgrid=False,
        xaxis_zeroline=False,
        xaxis_showticklabels=False,
        height = 50,
        margin=dict(t=5, b=5, l=5, r=5),
        plot_bgcolor='rgba(0,0,0,0)', 
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    metric_str = f'{metric} {unit}' if unit is not None else str(metric)
    
    bullet_chart_card_div = html.Div(
        [
            html.Img(src=f"assets/icons/{img}", style=img_style, id=img),
            html.Div(
                [
                    html.P(metrictype, style={"margin": "5px", "margin-top": "20px", "font-weight": "lighter", "font-size": "15px", 'white-space': 'nowrap', 'min-width':'180px'}, id=f"{metric_card_id}-label"),
                    html.H4(metric_str, style={ "margin": "5px" , 'white-space': 'nowrap'}, id=f"{metric_card_id}-value"),
                    #dcc.Graph(figure=fig, className="bullet_chart"),
                    html.Div( [ html.Div([], style=load_style), html.Div([], style=constraint_style) ] ,style=bar_style)
                ],
                style = bullet_chart_style
            ),
            
        ],
        style = bullet_chart_card_style,
        id = metric_card_id
    )
    
    return bullet_chart_card_div

def bullet_chart_card_basic(metric, min, max, unit='', info='', back_color="#FFFFFF", bar_color='#EFEFEF', load_color='#6173E9', margin='10px', flex='100%', min_width=None,metric_card_id="bullet-chart-basic"):
    """
    Generate a basic bullet chart card to display a metric.

    Args:
        metric (float): The value of the metric.
        min (float): The minimum value of the metric range.
        max (float): The maximum value of the metric range.
        unit (str, optional): The unit of measurement for the metric. Defaults to ''.
        info (str, optional): Additional information about the metric. Defaults to ''.
        back_color (str, optional): Background color of the card. Defaults to "#FFFFFF".
        bar_color (str, optional): Color of the bar in the bullet chart. Defaults to '#EFEFEF'.
        load_color (str, optional): Color of the loaded part of the bar. Defaults to '#6173E9'.
        margin (str, optional): The margin around the card. Defaults to '10px'.
        flex (str, optional): Flex value of the card. Defaults to '100%'.
        min_width (str, optional): The minimum width of the card. Defaults to None.
        metric_card_id (str, optional): The id of the bullet chart basic card. Defaults to "bullet-chart-basic".

    Returns:
        dash.html.Div: Basic bullet chart card containing the metric.
    """
    metric = round(metric, 3)
    range = max - min
    load_percent = ((metric - min) / range * 100) if range != 0 else 0
    
    bullet_chart_basic_style = {
        'border-radius': '20px',
        'background': back_color,
        'box-shadow': '3px 3px 10px 1px rgba(0, 0, 0, 0.05)',
        'padding': '10px',
        'margin': margin,
        'display': 'block',
        'flex': flex,
        'min-width':min_width,
    }
    
    bar_style = {
        'border-radius': '30px',
        'background': bar_color,
        'position': 'relative',
        'height': 15,
        'border': f'3px solid {bar_color}'
    }
    
    load_style = {
        'border-radius': '30px',
        'background': load_color,
        'height': 15,
        'width': f'{load_percent}%',
    }
    
    bullet_chart_basic_div = html.Div(
        [
            dmc.Group(
                [
                    html.P(str(metric), style={"margin": "5px", "font-weight": "bold", "font-size": "15px", 'color': load_color}, id=f"{metric_card_id}-label"),
                    html.P(unit, style={"margin": "5px", "font-weight": "lighter", "font-size": "15px", 'color': load_color}, id=f"{metric_card_id}-label"),
                ],
                position='left',
                spacing='s'
            ),
            html.Div( [ html.Div([], style=load_style)], style=bar_style)
        ],
        style = bullet_chart_basic_style
    )
    
    return bullet_chart_basic_div  


### FEEDBACK ###
def warning(message):
    """
    Generate a warning message.

    Args:
        message (str): The text content of the warning message.

    Returns:
        dash.html.Div: Div containing the warning message.
    """
    children = [DashIconify(icon='mingcute:information-line', height=25, width=25), html.P(message, className='feedback-text')]
    warning_div = html.Div(children=children, className="feedback feedback-warning")
    
    return warning_div

def information(message):
    """
    Generate an information message.

    Args:
        message (str): The text content of the information message.

    Returns:
        dash.html.Div: Div containing the information message.
    """
    children = [DashIconify(icon='mingcute:information-line', height=25, width=25), html.P(message, className='feedback-text')]
    warning_div = html.Div(children=children, className="feedback feedback-information")
    
    return warning_div


### CHROMOSOME SEQUENCE ###
def chromosome_sequence(chromosome, justify="flex-start", align="flex-start", compromised=False, unique_genes=None):
    """
    Generate a sequence of chromosome genes.

    Args:
        chromosome (list): List of genes in the chromosome.
        justify (str, optional): Justification of the sequence. Defaults to "flex-start".
        align (str, optional): Alignment of the sequence. Defaults to "flex-start".
        compromised (bool, optional): Whether the chromosome is compromised or not. Defaults to False.
        unique_genes (dict, optional): Dictionary of unique genes with corresponding colors. Defaults to None.

    Returns:
        dash_mantine_components.Stack: Stack of chromosome genes.
    """
    chromosome_sequence = []
    
    for gene in chromosome:
        
        color = '#6173E9'
        if unique_genes is not None:
            color = unique_genes[gene["layer"]]
        
        gene_params = str(gene)
        gene_params = gene_params.replace('{', '').replace('}', '').replace("'", '').replace(",", '\n')
        gene_name = gene["layer"][0] if compromised else gene["layer"].replace('_', '')
        
        tooltip = dmc.Tooltip(
            label=gene_params,
            position="right",
            offset=3,
            transition="slide-up",
            color='gray',
            multiline=True,
            children=[dmc.Badge(gene_name, variant='light', color='indigo', style={'flex': '100%', 'background-color': f"{color}33", 'color': color})]
        )
        
        chromosome_sequence.append(tooltip)
        
    return dmc.Stack(chromosome_sequence, justify=justify, align=align, spacing="0px")


### NOT IN USE ###
def fitness_function():
    
    style_backrgound = {
        'border-radius': '20px',
        'border': '3px solid #FFFFFF',
        'background': '#FFFFFF',
        'box-shadow': '3px 3px 10px 1px rgba(0, 0, 0, 0.05)',
        'margin': '5px',
        'display': 'inline-flex',
        'vertical-align': 'center',
        'width': '100%',
        'align-items': 'center',
        'justify-content': 'flex-start',  
    }
    
    img_style = {
        'max-width': '100%',
        'max-height': '40px',
        'margin': '10px'
    }
    
    img_background_style = {
        'border-radius': '15px',
        'display': 'flex',
        'justify-content': 'center',
        'align-items': 'center',
        'background': '#EFEFEF',
        #'width': '100%',
        'height': '100%',
        'margin': '2px',
        'padding': '10px',
        'flex-grow': '3'
    }
    
    description_style = {
        'display': 'inline-flex',
        'flex-grow': '1',
        'margin': '5px'
    }
    
    fitness_function_div = dmc.Grid(
        [
            html.Div(html.Img(src=f"assets/media/fitness-function.png", style=img_style), style=img_background_style),
            html.Div(
                [
                    html.Div([html.P('Weights', style={"margin": "5px", "font-weight": "lighter","font-size": "15px",}), html.H4('a b c', style={ "margin": "5px" })]),
                    html.Div([html.P('Objectives', style={"margin": "5px", "font-weight": "lighter","font-size": "15px",}), html.H4('M T W', style={ "margin": "5px" })])
                ],
                style=description_style
            )
        ],
        style=style_backrgound,
    )
    
    return fitness_function_div

def fitnes_function_latex(dashlatex):
    
    style_backrgound = {
        'border-radius': '15px',
        'border': '5px solid white',
        'margin': '5px',
        'display': 'flex',
        'justify-content': 'center',
        'align-items': 'center',
        'background': '#EFEFEF',
        'width': '100%',
        'height': '100%',
        'padding': '10px',
        'flex-grow': '3'
    }
    
    fitness_function_div = html.Div(dashlatex, style=style_backrgound)
    
    return fitness_function_div
    