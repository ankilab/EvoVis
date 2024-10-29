import dash
from dash import html
from dash_iconify import DashIconify


############################################################

# MODULE EVOVIS APP

# The Module combines the four EvoVis pages into a web 
# application where the user can navigate from page to page.

############################################################


### LAYOUT COMPONENTS
def navbar():
    """
    Generate the navigation bar of EvoVis.

    Returns:
        dash.html.Div: Navigation bar containing links to different pages.
    """
    return html.Div(
    [
        html.Div(
            [   
                html.A(children=html.Img(src="assets/media/evonas-logo.png", height="50px"), href=dash.page_registry['pages.hyperparameters_page']['relative_path']),
            ],
            id="navrun"
        ),
        html.Div(
            [
                html.A(html.Button(children=DashIconify(icon="streamline:input-box-solid", height=25, width=25, color="#000000"), className="circle-btn", id="hyperparameter-link"), href=dash.page_registry['pages.hyperparameters_page']['relative_path']),
                html.A(html.Button(children=DashIconify(icon="jam:dna", height=25, width=25, color="#000000"), className="circle-btn", id="genepool-link"), href=dash.page_registry['pages.genepool_page']['relative_path']),
                html.A(html.Button(children=DashIconify(icon="mdi:graph", height=25, width=25, color="#000000"), className="circle-btn", id="family-tree-link"), href=dash.page_registry['pages.family_tree_page']['relative_path']),
                html.A(html.Button(children=DashIconify(icon="entypo:bar-graph", height=25, width=25,color="#000000"), className="circle-btn", id="results-link"), href=dash.page_registry['pages.run_results_page']['relative_path']), 
            ],
            id="navlinks",
        )    
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
    app.run_server(host="0.0.0.0", port="8050", debug=False)
