from dash import html, dcc, Output, Input, State
import dash_bootstrap_components as dbc
import dash
import dash_auth
import pandas as pd

df = pd.read_csv("mcb2.csv")
 
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SKETCHY], title='Portfolio ')

server = app.server

# VALID_USERNAME_PASSWORD_PAIRS = {
#     'test': 'test123'
# }

# auth = dash_auth.BasicAuth(
#     app,
#     VALID_USERNAME_PASSWORD_PAIRS
# )

app.layout = dbc.Container([
    dbc.Row(
        [
            html.Div([html.P(['Portfolio recommender Model'],
                             style={'marginBottom': 15, 'marginTop': 10, 'text-align': 'center', 'color': 'Green',
                                    'fontSize': 18})]),
            dbc.Col([
                html.P(['What is the age of the Customer'],
                       style={'marginBottom': 15, 'marginTop': 5, 'color': 'Green', 'fontSize': 14}),
                dcc.Input(placeholder="Age", type="number", value=34,
                          style={'color': '#F71AF7', 'fontSize': 14}, id="drp"),

                html.P(['What is the average salary?'],
                       style={'marginBottom': 15, 'marginTop': 5, 'color': 'Green', 'fontSize': 14}),
                dcc.Input(placeholder="Average Salary", type="number", value=120000,
                          style={'color': 'LimeGreen', 'fontSize': 14}, id="drp1"),

                html.P(['How much is the incoming Transactions?'],
                       style={'marginBottom': 15, 'marginTop': 5, 'color': 'Green', 'fontSize': 14}),
                dcc.Input(placeholder="Incoming Transactions", type="number", value=300000,
                          style={'color': '#0EE3D1', 'fontSize': 14}, id="drp2"),

                html.P(['How much are the outgoing Transcations?'],
                       style={'marginBottom': 15, 'marginTop': 5, 'color': 'Green', 'fontSize': 14}),
                dcc.Input(placeholder="Outgoing Transactions", value=50000, type="number",
                          style={'color': '#FF11F1', 'fontSize': 14}, id="drp3"),

                html.P(['What is your current Saving Balance?'],
                       style={'marginBottom': 15, 'marginTop': 5, 'color': 'Green', 'fontSize': 14}),
                dcc.Input(placeholder="Current saving Balance", value=100000, type="number",
                          style={'color': '#F40C0F', 'fontSize': 14}, id="drp4"),

                html.P(['What Customer Market does the customer Fall in?'],
                       style={'marginBottom': 15, 'marginTop': 5, 'color': 'Green', 'fontSize': 14}),
                dcc.Dropdown(["segment1825", "eamfi", "icpsstaffsegmentA", "icpsstaffsegmentB", "privatebanking",
                              "privatebankinginternational", "privatebankinglocal", "rupys012", "rupys1317", "mass",
                              "massAfluent", "mcbSelcet", "mcbInternational", "mcbselectInternational"],
                             placeholder="Customer Market", clearable=True, value='mass',
                             style={'color': '#0C34F4', 'fontSize': 14}, id="drp5"),
                html.Br(),
                html.P([html.U([html.Cite(
                    [html.Button('Recommended Accounts', id='refer',
                                 style={'backgroundColor': 'F0E68C', 'marginTop': '10px',
                                        'marginRight': '100px', 'marginBottom': '50px',
                                        'marginLeft': '90px',
                                        "border": "2px LightGreen"})], id="cite")])],
                    style={'backgroundColor': 'ff79c6'})
            ], md=3, sm=12, lg=3, style={'fontSize': 14}),

            dbc.Col([html.P(id='parsed', style={'backgroundColor': 'F0E68C', 'fontSize': 15}),
                     html.Div(id="explanations")], md=4, sm=12, lg=4,
                    style={'display': 'inline-block', 'backgroundColor': 'F0E68C',
                           'color': 'LimeGreen', 'fontSize': 14}),

            dbc.Col([], md=1, sm=1, lg=1, className="m-5"),

            dbc.Col([html.Div([
            ])], id="app1", md=3, sm=12, lg=3,
                style={'display': 'inline-block', 'backgroundColor': 'F0E68C',
                       'border': '2px Green', 'marginTop': '20px',
                       'color': 'LimeGreen', 'fontSize': 14})
        ], style={'height': '100vh'}),
], id="container", fluid=True)


@app.callback(
    Output('parsed', 'children'),
    Output('explanations', 'children'),
    Output('app1', 'children'),
    Input('refer', 'n_clicks'),
    State('drp', 'value'),
    State('drp1', 'value'),
    State('drp2', 'value'),
    State('drp3', 'value'),
    State('drp4', 'value'),
    State('drp5', 'value'), prevent_initial_call=True)
def update_output(n_clicks, state, state0, state1, state2, state3, state4):
    if n_clicks:
        q = state
        w = state0
        e = state1
        r = state2
        t = state3
        y = state4

        # creating out of sample data (Prompt from the Relationship Manager)
        # The schema of the X_test + the input from the application (The PROMPTS)

        schema = {'customerAge': {71569: 0.0},
                  'averageSalary': {150000: 0.0},
                  'incomingTransactions': {150000: 0.0},
                  'outgoingTransactions': {150000: 0.0},
                  'currentsavingBalance': {150000: 0.0},
                  'customerMarket_eamfi': {150000: 0.0},
                  'customerMarket_icpsstaffsegmentA': {150000: 0.0},
                  'customerMarket_icpsstaffsegmentB': {150000: 0.0},
                  'customerMarket_mass': {150000: 0.0},
                  'customerMarket_massAfluent': {150000: 0.0},
                  'customerMarket_mcbInternational': {150000: 0.0},
                  'customerMarket_mcbSelcet': {150000: 0.0},
                  'customerMarket_mcbselectInternational': {150000: 0.0},
                  'customerMarket_privatebanking': {150000: 0.0},
                  'customerMarket_privatebankinginternational': {150000: 0.0},
                  'customerMarket_privatebankinglocal': {150000: 0.0},
                  'customerMarket_rupys012': {150000: 0.0},
                  'customerMarket_rupys1317': {150000: 0.0},
                  'customerMarket_segment1825': {150000: 0.0}}

        userInputs = {'customerAge': q,
                      'averageSalary': w,
                      'incomingTransactions': e,
                      'outgoingTransactions': r,
                      'currentsavingBalance': t,
                      'customerMarket': y}

        userInputs = pd.DataFrame(userInputs, index=[150000])

        encode_Inputs = pd.get_dummies(userInputs)

        encode_Inputs = encode_Inputs.to_dict()

        # Convert Merged Dictionary to Models format

        new_df = {**schema, **encode_Inputs}

        pred = pd.DataFrame(new_df, index=[150000])

        def mrLgbm():
            # Load the imports
            import joblib
            import pandas as pd
            import plotly.express as px
            from sklearn.neighbors import NearestNeighbors
            from sklearn.preprocessing import StandardScaler

            scaler = StandardScaler()

            # Load the model
            mrLgbm = joblib.load('mrLgbm.pkl')

            pre = mrLgbm.predict(pred)

            columns = ['localcurrencyAccount', 'foreigncurrencyAccount',
                       'fixeddepositforeigncurrencyAccount',
                       'fixeddepositlocalcurrencyAccount', 'carloanAccount',
                       'educationloanAccount', 'generalloanAccount', 'housingloanAccount',
                       'personalloanAccount', 'onetimeinvestmentAccount',
                       'monthlyeducationPlan', 'monthlyretirementPlan ',
                       'americanexpressgoldAccounts', 'americanexpressgreenbundleAccounts',
                       'mastercardclassicAccounts', 'mastercardforeignAccounts',
                       'mastercardgoldAccounts', 'mastercardprimoAccounts',
                       'visaclassicAccounts', 'visagoldAccounts', 'neobundleAccounts']

            newdf = pd.DataFrame(pre, columns=columns)

            def NN():
                from dash import dash_table
                # import and build the NN recommender
                df = pd.read_csv("mcb2.csv")
                df.drop("date", axis=1, inplace=True)
                df = df.dropna()
                y = df.iloc[:, 7:]
                df_scaledy = pd.DataFrame(scaler.fit_transform(y), columns=y.columns)
                y = df_scaledy

                # Create a NearestNeighbors model
                modelNN = NearestNeighbors(n_neighbors=4, metric='euclidean', n_jobs=-1)

                modelNN = modelNN.fit(y)

                # Use the loaded model for recommendation
                # Get the distances and indices of the 7 nearest neighbors
                distances, indices = modelNN.kneighbors(newdf)

                # Get the labels of the nearest neighbors
                nearest_labels = y.iloc[indices[0]]

                # Euclidean distance of the all neighbouring accounts
                recommend = nearest_labels.sum()

                # print(recommend)

                color = ["blu", "red", "blue", "gen", "purpl", "orange",
                         "bl", "green", "purpe", "ange", "ack", "rd",
                         "pple", "onge", "ack", "reed", "bluee", "geen",
                         "blacck", "rted", "bllue"]

                fig = px.bar(recommend, text_auto=True, color=color, width=700, height=700,
                             title="Sum of euclidean distance of 4 Neighbours")

                fig.update_layout(showlegend=False)

                fig.update_layout(
                    plot_bgcolor='#ffffff',
                    paper_bgcolor='#ffffff')

                fig.update_layout(
                    title={
                        'text': "Sum of euclidean distance of Accounts of 4 Neighbours",
                        'y': 0.9,
                        'x': 0.5,
                        'xanchor': 'center',
                        'yanchor': 'top'})

                graphRecommenderExplainer = dcc.Graph(figure=fig, config={'displaylogo': False,
                                                                          'modeBarButtonsToRemove': ['lasso2d',
                                                                                                     'resetViewMapbox',
                                                                                                     'zoom2d',
                                                                                                     'select2d']})

                # Recommend the top 3 accounts for each customer
                recommendation = nearest_labels.sum().nlargest(3).index.tolist()

                r, t, q = recommendation

                # Create df for storing live the predictions for audit
                graphDict = f"{r}, {t}, {q}"

                graphPd = pd.DataFrame([graphDict], index=[150000], columns=["modelPrediction"])

                graph_pd = pd.concat([userInputs, graphPd], axis=1)

                # Save all the Prompts and the corresponding Recommendations.
                # write the dataframe to a csv file row by row

                graph_pd.to_csv('reco.csv', index=False, mode='a', header=False)

                reco = pd.read_csv("reco.csv")

                dashTable = html.Div([
                    dash_table.DataTable(
                        id='table',
                        columns=[{"name": i, "id": i, "renamable": True, "hideable": True} for i in reco.columns],
                        data=reco.to_dict('records'),
                        style_table={'overflowX': 'auto'},
                        export_format='xlsx',
                        editable=True,
                        include_headers_on_copy_paste=True,
                        sort_action='native',
                        page_action="native",
                        page_size=8,
                        style_cell={
                            'height': 'auto',
                            'minWidth': '140px', 'width': '150px', 'maxWidth': '180px',
                            'whiteSpace': 'normal'
                        },
                        style_header={
                            'backgroundColor': 'rgb(230, 230, 230)',
                            'color': 'black'
                        },
                        style_data={
                            'backgroundColor': 'rgb(230, 230, 230)',
                            'color': 'black'
                        }
                    )
                ])

                recommendation = f"Recommended accounts: {r}, {t}, {q}"

                return recommendation, graphRecommenderExplainer, dashTable

            return NN()

        return mrLgbm()

    else:
        return dash.no_update, dash.no_update


if __name__ == "__main__":
    app.run_server(debug=True, port=5050)
