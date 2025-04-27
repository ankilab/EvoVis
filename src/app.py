import dash
from dash import html
from dash_iconify import DashIconify
import dash_mantine_components as dmc


############################################################

# MODULE EVOVIS APP

# The Module combines the four EvoVis pages into a web 
# application where the user can navigate from page to page.

############################################################


### LAYOUT COMPONENTS
def navbar():
    """
    Generate the navigation bar of EvoVis with tooltips for better usability.
    
    Returns:
        dash.html.Div: Navigation bar containing links to different pages with tooltips.
    """
    # Header descriptions and configurations
    nav_items = [
        {"desc": "Hyperparameter Overview", "icon": "streamline:input-box-solid", "id": "hyperparameter-link", 
         "path": dash.page_registry['pages.hyperparameters_page']['relative_path']},
        {"desc": "Gene Pool", "icon": "jam:dna", "id": "genepool-link", 
         "path": dash.page_registry['pages.genepool_page']['relative_path']},
        {"desc": "Family Tree", "icon": "mdi:graph", "id": "family-tree-link", 
         "path": dash.page_registry['pages.family_tree_page']['relative_path']},
        {"desc": "Run Results", "icon": "entypo:bar-graph", "id": "results-link", 
         "path": dash.page_registry['pages.run_results_page']['relative_path']}
    ]
    
    # Create nav links with tooltips
    nav_links = []
    for item in nav_items:
        nav_links.append(
            dmc.Tooltip(
                label=item["desc"],
                position="bottom",
                offset=5,
                radius=15,
                transition="pop-top-left",
                color="#6173E9",
                multiline=False,
                width="auto",
                children=[
                    html.A(
                        html.Button(
                            children=DashIconify(icon=item["icon"], height=25, width=25, color="#000000"), 
                            className="circle-btn", 
                            id=item["id"]
                        ), 
                        href=item["path"],
                        style={"display": "inline-block"}
                    )
                ]
            )
        )
    
    return html.Div(
        [
            html.Div(
                [html.A(children=html.Img(src="assets/media/evonas-logo.png", height="50px"), 
                        href=dash.page_registry['pages.hyperparameters_page']['relative_path'])],
                id="navrun"
            ),
            html.Div(nav_links, id="navlinks", style={"display": "flex", "align-items": "center"})
        ],
        id="navbar",
    )

def page():
    """
    Generate the page content container.

    Returns:
        dash.html.Div: Page content container.
    """
    return html.Div([ dash.page_container], id="page-content")

def app_layout():
    """
    Generate the layout of the application.

    Returns:
        dash.html.Div: Layout of the application containing the navigation bar and page content.
    """
    return html.Div([
        navbar(), 
        page()
    ])

### DASH APP & LAYOUT 
app = dash.Dash(__name__, use_pages=True)
app.layout = app_layout

if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port="8050", debug=True)
