import pandas as pd
import plotly.express as px
import dash
import dash_core_components as dcc
import dash_html_components as html
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import plotly.graph_objs as go
from matplotlib.patches import Ellipse
import numpy as np
import plotly.io as pio
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template
from dash.dependencies import Input, Output
from statistics import mean
import math
from scipy.stats import t,ttest_ind
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import warnings



warnings.simplefilter(action='ignore', category=FutureWarning)




def stdev(X):
    m = mean(X)
    return math.sqrt(sum((x-m)**2 for x in X) / (len(X)-1))

def degreesOfFreedom(X, Y):
    s1 = (stdev(X)**2)
    s2 = (stdev(Y)**2)
    df = (s1 / len(X) + s2 / len(Y))**2 / ((s1 / len(X))**2 / (len(X) - 1) + (s2 / len(Y))**2 /
(len(Y) - 1))
    return(df)



df=pd.read_csv("homicides.csv")
dff=df.copy()

known_relationship = df[df['Relationship'] != 'Unknown']
relationship_counts = known_relationship['Relationship'].value_counts()
df['Perpetrator Age'] = pd.to_numeric(df['Perpetrator Age'], errors='coerce')
df = df.dropna()
df = df[(df['Victim Age'] != 998)&(df['Perpetrator Age'] != 0)&(df['Perpetrator Age'].notnull())]


fig1=go.Figure()

fig1.add_trace(go.Violin(y=df["Perpetrator Age"], name="Perpetrator Age",box_visible=True,meanline_visible=True))
fig1.add_trace(go.Violin(y=df["Victim Age"], name="Victim Age",box_visible=True,meanline_visible=True))

fig1.update_layout(title="Violin Plots of Perpetrator Age and Victim Age",yaxis_title='Age',title_x=0.5)

df2 = df.drop(["Record ID", "Incident"], axis=1)

corr_df = df2.corr().round(2)

fig2=px.imshow(corr_df, x=corr_df.columns, y=corr_df.columns, color_continuous_scale='Sunset', title='Correlation Heatmap',text_auto=True)

fig2.update_layout(title="Correlation Heatmap", title_x=0.5, width=800, height=600)

new_df = df[['Victim Age', 'Perpetrator Age']].head(50000)


fig3 = px.scatter(new_df, x="Victim Age", y="Perpetrator Age")
fig3.update_layout(xaxis_title='Victim Age', yaxis_title='Perpetrator Age',title='Scatterplot',title_x=0.5)

kmeans = KMeans(n_clusters=2,n_init="auto").fit(new_df)
labels = kmeans.predict(new_df)
fig4 = px.scatter(new_df, x='Victim Age', y='Perpetrator Age', color=labels)
fig4.update_layout(xaxis_title='Victim Age', yaxis_title='Perpetrator Age',title='K-means Clustering',title_x=0.5)


centroids = kmeans.cluster_centers_

cluster_0 = new_df[labels == 0]
cluster_1 = new_df[labels == 1]

centroid_0 = cluster_0.mean(axis=0)
centroid_1 = cluster_1.mean(axis=0)

radius_0_90 = np.percentile(np.linalg.norm(cluster_0 - centroid_0, axis=1), 90)
radius_1_90 = np.percentile(np.linalg.norm(cluster_1 - centroid_1, axis=1), 90)

circle_0 = go.layout.Shape(
    type="circle",
    x0=centroid_0[0]-radius_0_90,
    y0=centroid_0[1]-radius_0_90,
    x1=centroid_0[0]+radius_0_90,
    y1=centroid_0[1]+radius_0_90,
    line=dict(color="red", width=2),
    fillcolor="rgba(0,0,0,0)")

circle_1 = go.layout.Shape(
    type="circle",
    x0=centroid_1[0]-radius_1_90,
    y0=centroid_1[1]-radius_1_90,
    x1=centroid_1[0]+radius_1_90,
    y1=centroid_1[1]+radius_1_90,
    line=dict(color="green", width=2),
    fillcolor="rgba(0,0,0,0)")


fig5 = go.Figure()

fig5.add_trace(go.Scatter(x=new_df['Victim Age'], y=new_df['Perpetrator Age'], mode='markers', marker=dict(color=labels)))

fig5.update_layout(
    xaxis_range=[min(new_df["Victim Age"]), max(new_df["Victim Age"])],
    yaxis_range=[min(new_df["Perpetrator Age"]), max(new_df["Perpetrator Age"])]
)

fig5.add_trace(go.Scatter(x=[centroid_0[0]], y=[centroid_0[1]], marker=dict(size=10, color="red", symbol="x")))
fig5.add_trace(go.Scatter(x=[centroid_1[0]], y=[centroid_1[1]], marker=dict(size=10, color="green", symbol="x")))

fig5.add_shape(circle_0)
fig5.add_shape(circle_1)
fig5.update_layout(xaxis_title='Victim Age', yaxis_title='Perpetrator Age',title='K-means Clustering',title_x=0.5)


wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(new_df)
    wcss.append(kmeans.inertia_)

fig6= px.line(x=range(1, 11), y=wcss, title='Elbow Method', labels={'x': 'Number of clusters', 'y': 'WCSS'})
fig6.update_layout(title_x=0.5)

perpetrator_age=df['Perpetrator Age'].head(1000)
victim_age=df['Victim Age'].head(1000)

testStatistic,pValue=ttest_ind(perpetrator_age,victim_age,equal_var=False)

a=perpetrator_age
b=victim_age

degFreedom=degreesOfFreedom(a,b)

x = np.linspace(t.ppf(0.001, degFreedom),t.ppf(0.999, degFreedom), 100)

y = t.pdf(x, degFreedom)

fig7= go.Figure(
    go.Scatter(x=x, y=y, mode='lines', line=dict(color='black'))
)

critical_points = t.ppf(0.025, degFreedom)
fig7.add_shape(
    go.layout.Shape(
        type='line',
        x0=-critical_points,
        y0=0,
        x1=-critical_points,
        y1=np.max(y),
        line=dict(color='red', dash='dash')
    )
)
fig7.add_shape(
    go.layout.Shape(
        type='line',
        x0=critical_points,
        y0=0,
        x1=critical_points,
        y1=np.max(y),
        line=dict(color='red', dash='dash')
    )
)

fig7.add_trace(go.Scatter(x=[testStatistic], y=[0], mode='markers', marker=dict(color='blue')))

fig7.update_layout(title='Two-sample t-test for Perpetrator Age and Victim Age',title_x=0.5)

popped_df=dff.pop("Crime Solved")
dff.insert(len(dff.columns), 'Crime Solved', popped_df)

no_samples = dff[dff['Crime Solved']=='No'].sample(n=10000, random_state=42)
yes_samples = dff[dff['Crime Solved']=='Yes'].sample(n=10000, random_state=42)

combined_samples=pd.concat([no_samples,yes_samples],ignore_index=True)

label_encoder = LabelEncoder()
combined_samples['Agency Name'] = label_encoder.fit_transform(combined_samples['Agency Name'])
combined_samples['Agency Type'] = label_encoder.fit_transform(combined_samples['Agency Type'])
combined_samples['City'] = label_encoder.fit_transform(combined_samples['City'])
combined_samples['State'] = label_encoder.fit_transform(combined_samples['State'])
combined_samples['Crime Type'] = label_encoder.fit_transform(combined_samples['Crime Type'])
combined_samples['Perpetrator Sex'] = label_encoder.fit_transform(combined_samples['Perpetrator Sex'])
combined_samples['Perpetrator Race'] = label_encoder.fit_transform(combined_samples['Perpetrator Race'])
combined_samples['Perpetrator Ethnicity'] = label_encoder.fit_transform(combined_samples['Perpetrator Ethnicity'])
combined_samples['Relationship'] = label_encoder.fit_transform(combined_samples['Relationship'])
combined_samples['Weapon'] = label_encoder.fit_transform(combined_samples['Weapon'])
combined_samples['Record Source'] = label_encoder.fit_transform(combined_samples['Record Source'])
combined_samples['Victim Sex'] = label_encoder.fit_transform(combined_samples['Victim Sex'])
combined_samples['Victim Race'] = label_encoder.fit_transform(combined_samples['Victim Race'])
combined_samples['Victim Ethnicity'] = label_encoder.fit_transform(combined_samples['Victim Ethnicity'])
combined_samples['Month'] = label_encoder.fit_transform(combined_samples['Month'])

combined_samples.replace(" ", float("nan"), inplace=True)

combined_samples.dropna(inplace=True)

x=combined_samples.loc[:,'Agency Name':'Weapon']


y=combined_samples['Crime Solved']


X_train,X_test,Y_train,Y_test=train_test_split(x,y, test_size=0.2, random_state=42)

dt = DecisionTreeClassifier()


dt.fit(X_train, Y_train)

Y_pred = dt.predict(X_test)

accuracy = accuracy_score(Y_test, Y_pred)
precision = precision_score(Y_test, Y_pred,pos_label="Yes")
recall = recall_score(Y_test, Y_pred,pos_label="Yes")

cm = confusion_matrix(Y_test, Y_pred)

fig8= px.imshow(cm,
                labels=dict(x="Predicted label", y="True label", color="Count"),
                x=["No", "Yes"],
                y=["No", "Yes"],
                color_continuous_scale=px.colors.sequential.Blues,
                title="Confusion Matrix for a Decision Tree Classifier regarding Crime Solved",text_auto=True)

fig8.update_layout(title_x=0.5)




Y_prob = dt.predict_proba(X_test)[:, 1]

fpr, tpr, thresholds = roc_curve(Y_test, Y_prob,pos_label='Yes')

auc_score = roc_auc_score(Y_test, Y_prob)

fig9 = px.line(x=fpr, y=tpr, title='ROC Curve')
fig9.update_layout(title_x=0.5)
fig9.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)
fig9.update_xaxes(title_text='False Positive Rate')
fig9.update_yaxes(title_text='True Positive Rate')

df_counts = df.groupby(['State', 'City'])['Record ID'].count().reset_index()
df_counts = df_counts.rename(columns={'Record ID': 'Homicide Cases'})


fig10 = px.sunburst(df_counts, path=['State', 'City'], values='Homicide Cases')
fig10.update_layout(title="Distribution of Homicide Cases by State and City",title_x=0.5)






app = dash.Dash(__name__)

app.layout = html.Div(className='row', children=[
    html.H1('Homicide Analysis', style={"text-align":'center'}),
    html.Br(),
    html.Div(className='col-6', children=[
        dcc.Graph(id="graph1", figure=fig1),
    ],style={'display': 'inline-block','width': '50%'}),
    html.Div(className='col-6', children=[
        html.Div(className='row', children=[
            dcc.Dropdown(
                id='dropdown',
                options=[
                    {'label': 'Victim Race', 'value': 'Victim Race'},
                    {'label': 'Victim Sex', 'value': 'Victim Sex'},
                    {'label': 'Weapon', 'value': 'Weapon'},
                    {'label': 'Perpetrator Race', 'value': 'Perpetrator Race'},
                    {'label': 'Perpetrator Sex', 'value': 'Perpetrator Sex'},
                    {'label': 'Crime Solved', 'value': 'Crime Solved'}
                ],
                value='Victim Race'
            ),
        ]),
        html.Div(className='row', children=[
            dcc.Graph(id="piechart")
        ])
    ],style={'display': 'inline-block','width': '50%'}),
    html.Div(className='col-6',children=[
        html.Div(className='row',children=[
            dcc.Slider(1980,2014,1,
                       marks={i: '{}'.format(i) for i in range(1980,2015,1)},
                       value=1980,id='slider')]),
        html.Div(className='row', children=[
            dcc.Graph(id="choropleth")
        ])
        
],style={'margin-left':'auto','margin-right':'auto','height':'80%','width':'80%'}),

    html.Div(className='col-6',children=[
        html.Div(className='row',children=[
            dcc.RadioItems(
                id='buttons',
                options=[
                    {'label': 'Relationship', 'value': 'Relationship'},
                    {'label': 'State', 'value': 'State'}],
                value='Relationship',inline=True),
            ],style={'margin-left':'20px'}),
        html.Br(),
        html.Div(className='row',children=[
            dcc.Graph(id='barchart')])
        ],style={'display': 'inline-block','width': '50%','float': 'left'}),

    html.Div(className='col-6',children=[
        html.Div(className='row',children=[
            dcc.Dropdown(
                id='multidropdown',
                options=[{'label':i,'value':i} for i in df['State'].unique()],
                value=df['State'].unique()[0],
                multi=True),
            ]),
        html.Div(className='row',children=[
            dcc.Graph(id='linegraph')])

        ],style={'display': 'inline-block','width': '50%','float': 'left'}),

    html.Div(className='col-6', children=[
        dcc.Graph(id="graph2", figure=fig2),
        ],style={'margin-left':'auto','margin-right':'auto','height':'80%','width':'80%','clear':'both'}),
    html.Div(className='col-6', children=[
        dcc.Graph(id="graph3", figure=fig3),
    ],style={'display': 'inline-block','width': '25%'}),
    html.Div(className='col-6', children=[
        dcc.Graph(id="graph4", figure=fig4),
    ],style={'display': 'inline-block','width': '25%'}),
    html.Div(className='col-6', children=[
        dcc.Graph(id="graph5", figure=fig5),
    ],style={'display': 'inline-block','width': '25%'}),
    html.Div(className='col-6', children=[
        dcc.Graph(id="graph6", figure=fig6),
    ],style={'display': 'inline-block','width': '25%'}),
    html.Div(className='col-6',children=[
        dcc.Graph(id="graph10",figure=fig10),
        ],style={'margin-left':'auto','margin-right':'auto','height':'80%','width':'80%'}),
    html.Div(className='col-6', children=[

        html.P("Enter the number of samples:"),

         html.Div(className='row', children=[
            dcc.Input(id='range', type='number', min=100, max=df.shape[0], step=1,value=100)
            ]),

        
        
        html.Div(className='row', children=[
            dcc.Graph(id='t-test')
            ])
        ],style={'margin-left':'auto','margin-right':'auto','height':'80%','width':'80%'}),

    html.Div(className='col-6', children=[
        dcc.Graph(id="graph8", figure=fig8),
        html.P("The accuracy score of the Decision Tree Classifier is {:.3f}".format(accuracy)),
        html.P("The precision score of the Decision Tree Classifier is {:.3f}".format(precision)),
        html.P("The recall score of the Decision Tree Classifier is {:.3f}".format(accuracy))


    ],style={'display': 'inline-block','width': '50%'}),

     html.Div(className='col-6', children=[
        dcc.Graph(id="graph9", figure=fig9),
        html.P("The AUC score of the Decision Tree Classifier is {:.3f}".format(auc_score))
    ],style={'display': 'inline-block','width': '50%'})
    


])

@app.callback(Output('piechart', 'figure'), [Input('dropdown', 'value')])
def update_pie_chart(selection):
    chosen_attribute = df[selection].value_counts()
    fig = px.pie(values=chosen_attribute.values, names=chosen_attribute.index)
    fig.update_layout(title="Distribution of {}".format(selection),title_x=0.5)

    return fig

@app.callback(Output('choropleth', 'figure'), [Input('slider', 'value')])
def update_choropleth(year):
    state_counts=df[df['Year'] == year]['State'].value_counts()
    us_state_abbrev = {
    'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR',
    'California': 'CA', 'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE', 'District of Columbia':'DC',
    'Florida': 'FL', 'Georgia': 'GA', 'Hawaii': 'HI', 'Idaho': 'ID',
    'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA', 'Kansas': 'KS',
    'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD',
    'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS',
    'Missouri': 'MO', 'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV',
    'New Hampshire': 'NH', 'New Jersey': 'NJ', 'New Mexico': 'NM', 'New York': 'NY',
    'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH', 'Oklahoma': 'OK',
    'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhodes Island': 'RI', 'South Carolina': 'SC',
    'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT',
    'Vermont': 'VT', 'Virginia': 'VA', 'Washington': 'WA', 'West Virginia': 'WV',
    'Wisconsin': 'WI', 'Wyoming': 'WY'
    }
    state_codes = [us_state_abbrev[state] for state in state_counts.index]

    fig=px.choropleth(locations=state_codes,
                    locationmode='USA-states',
                    color=state_counts.values,
                    scope="usa")
    
    fig.update_layout(title='Homicide Cases By State in {}'.format(year),title_x=0.5)

    return fig

@app.callback(Output('barchart', 'figure'), [Input('buttons', 'value')])
def update_bar_chart(category):
    specific_category=df[df[category] != 'Unknown']
    category_count=specific_category[category].value_counts()
    fig=px.bar(x=category_count.index, y=category_count.values, color_discrete_sequence=['pink'], labels={'x': '{}'.format(category), 'y': 'Count'})
    fig.update_layout(title='Distribution of {} in Homicide Cases'.format(category), xaxis_tickangle=-45,title_x=0.5)

    return fig

@app.callback(Output('linegraph', 'figure'), [Input('multidropdown', 'value')])
def update_line_graph(state):
    if isinstance(state, str):
        state = [state]
    state_year_counts = df[df['State'].isin(state)].groupby(['State', 'Year'])['Record ID'].count().reset_index()
    fig = px.line(state_year_counts, x='Year', y='Record ID', color='State', title='Homicide Count by State and Year')
    fig.update_layout(xaxis_title='Year', yaxis_title='Number of Homicides',title_x=0.5)
    return fig

@app.callback(Output('t-test', 'figure'), [Input('range', 'value')])
def update_ttest(number):
    perpetrator_age=df['Perpetrator Age'].head(number)
    victim_age=df['Victim Age'].head(number)
    testStatistic,pValue=ttest_ind(perpetrator_age,victim_age,equal_var=False)
    a=perpetrator_age
    b=victim_age
    degFreedom=degreesOfFreedom(a,b)
    x = np.linspace(t.ppf(0.001, degFreedom),t.ppf(0.999, degFreedom), 100)
    y = t.pdf(x, degFreedom)

    fig= go.Figure(
    go.Scatter(x=x, y=y, mode='lines', line=dict(color='black'))
    )

    critical_points = t.ppf(0.025, degFreedom)
    fig.add_shape(
        go.layout.Shape(
            type='line',
            x0=-critical_points,
            y0=0,
            x1=-critical_points,
            y1=np.max(y),
            line=dict(color='red', dash='dash')
            )
        )

    fig.add_shape(
        go.layout.Shape(
            type='line',
            x0=critical_points,
            y0=0,
            x1=critical_points,
            y1=np.max(y),
            line=dict(color='red', dash='dash')
            )
        )
    fig.add_trace(go.Scatter(x=[testStatistic], y=[0], mode='markers', marker=dict(color='blue')))
    fig.update_layout(title='Two-sample t-test for Perpetrator Age and Victim Age',title_x=0.5)

    return fig
    

    



app.run_server(debug=True,
    port=8085, 
    threaded=True)

