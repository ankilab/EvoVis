import dash
from dash import html
import dash_mantine_components as dmc
from dotenv import load_dotenv
import os
from components import dot_heading, parameter_card, warning
from evolution import get_hyperparameters
from dataval import validate_hyperparameters

### LOAD PATH FROM ENVIRONMENT VARIABLES
load_dotenv()
run = os.getenv("RUN_RESULTS_PATH")


### REGISTER DASH APP
dash.register_page(__name__, path='/')


### HYPERPARAMETER PAGE COMPONENTS
def grouped_metriccards():
    """Generate hyperparameter metric card divs assigned to their specified group divs.

    This function retrieves hyperparameters for a specified EvoNAS run and organizes them into groups based on their 'group' attribute.
    Each group is displayed as a header along with the corresponding hyperparameter metric cards.

    Returns:
        list: A list of HTML div elements representing the hyperparameter groups, 
              where each group includes the group name as the header and the associated hyperparameter metric cards.
    """
    
    # Retrieve hyperparameters for the specified run
    hps = get_hyperparameters(run)
    
    # Initialize dictionary to store hyperparameters grouped by their respective groups
    groups = {}
    
    # Initialize list to store hyperparameters without a group
    other_params = []

    for hp_key, hp in hps.items():
        # Extract values of hyperparameter and handle missing keys
        value = hp.get("value", None)
        display = hp.get("display", True)

        if value is not None and display: 
            # Extract values of hyperparameter and handle missing keys
            unit = hp.get("unit", None)
            icon = hp.get("icon", "icon-park:expand-text-input") or "icon-park:expand-text-input"
            displayname = hp.get("displayname", hp_key) or hp_key
            group = hp.get("group", None) 
            description = f"{displayname}: {hp.get('description', displayname)}"
            metrictype = displayname or hp_key.replace("_", " ").capitalize() 
            metric = str(hp.get("value", "")).replace("\n", "")
            
            # Create metric card div component for hyperparameter visual
            mc = parameter_card(metrictype=metrictype, metric=metric, icon=icon, unit=unit, description=description, max_width=280)

            # Add the metric card to its group or to the list of hyperparameters without a group
            if group is None:
                other_params.append(mc)
            elif group not in groups:
                groups[group] = [mc]
            else:
                groups[group].append(mc)
    

    # Add hyperparameters without a group to the end
    if len(other_params) != 0:   
        groups["Other parameters"] = other_params

    # Convert the group dictionaries to list of divs  
    group_divs = []

    # Create group div with group heading and metric cards for each group
    for group_key, group in groups.items():   
        group_div = html.Div([dot_heading(group_key)] + group, className="metrics")
        group_divs.append(group_div)

    return group_divs

def parameter_overview_header():
    """Generate hyperparameter page header.

    Returns:
        list: A list of HTML elements representing the header, including title text and a cover image.
    """
    return [
        html.H1("Parameters"),
        html.H1("Overview"),
        html.Img(src="assets/media/evolution-cover.png", id="evolution-cover"),
    ]


### HYPERPARAMETER PAGE LAYOUT 
def hyperparameter_layout():
    """
    Generates the layout for the hyperparameter page.
    If hyperparameters fail validation, it displays a warning message.

    Returns:
        dash_mantine_components.Grid: A Mantine Grid component representing the layout of the hyperparameter page.
    """
    
    validation_result = validate_hyperparameters(run)
    children = None
    
    if validation_result:
        children=[
            dmc.Col(parameter_overview_header(), span='auto', className='hyperparameters-header'),
            dmc.Col(warning(validation_result), span=8)
        ]
        print(validation_result)
        
    else:
        children=[
            dmc.Col(parameter_overview_header(), span='auto', className='hyperparameters-header'),
            dmc.Col(grouped_metriccards(), span=8)
        ]

    return dmc.Grid(
        children=children,
        justify="center",
        gutter="sm",
    )
    
layout = hyperparameter_layout
