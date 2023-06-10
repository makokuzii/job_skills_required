# import necessary libraries
import pandas as pd
import nltk
import plotly.graph_objects as go
import plotly.express as px
import dash_bootstrap_components as dbc
import holoviews as hv
hv.extension('plotly')

from dash.dependencies import Input, Output
from dash import Dash
from dash import html
from dash import dcc
from dash_bootstrap_templates import load_figure_template

# load the template that i personally liked
load_figure_template("bootstrap")

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

# create a markdown text on the web application
markdown_text = '''
### Realized by Sotheara SOK


This project aims to assist MSc students studying Data Management for Finance at Audencia in identifying crucial skills 
to align with their desired career path after graduation. The program mandates a **4-6 month internship**, acting as a vital 
bridge between theoretical knowledge and real-world application. By providing valuable guidance, this project aids 
students in prioritizing the skills necessary for their chosen career trajectory.


'''

# Clear the layout and do not display exception till callback gets executed
app.config.suppress_callback_exceptions = True

# Read the automobiles data into pandas dataframe
word_tokenizer = nltk.tokenize.word_tokenize
all_job = pd.read_csv('joined_job_title.csv')
all_job.fillna('', inplace=True)
all_job['description_tokens'] = all_job['description_tokens'].str.replace("'", "")
all_job['description_tokens'] = all_job['description_tokens'].apply(lambda x: word_tokenizer(x.lower()))

# Picked out keywords based on all keywords for the data analyst intern
keywords_programming = [
'sql', 'python', 'r', 'c', 'c#', 'javascript', 'js',  'java', 'scala', 'sas', 'matlab', 
'c++', 'c/c++', 'perl', 'go', 'typescript', 'bash', 'html', 'css', 'php', 'powershell', 'rust', 
'kotlin', 'ruby',  'dart', 'assembly', 'swift', 'vba', 'lua', 'groovy', 'delphi', 'objective-c', 
'haskell', 'elixir', 'julia', 'clojure', 'solidity', 'lisp', 'f#', 'fortran', 'erlang', 'apl', 
'cobol', 'ocaml', 'crystal', 'javascript/typescript', 'golang', 'nosql', 'mongodb', 't-sql', 'no-sql',
'visual_basic', 'pascal', 'mongo', 'pl/sql',  'sass', 'vb.net', 'mssql', 
]

keywords_libraries = [
'scikit-learn', 'jupyter', 'theano', 'openCV', 'spark', 'nltk', 'mlpack', 'chainer', 'fann', 'shogun', 
'dlib', 'mxnet', 'node.js', 'vue', 'vue.js', 'keras', 'ember.js', 'jse/jee',
]

keywords_analyst_tools = [
'excel', 'tableau',  'word', 'powerpoint', 'looker', 'powerbi', 'outlook', 'azure', 'jira', 'twilio',  'snowflake', 
'shell', 'linux', 'sas', 'sharepoint', 'mysql', 'visio', 'git', 'mssql', 'powerpoints', 'postgresql', 'spreadsheets',
'seaborn', 'pandas', 'gdpr', 'spreadsheet', 'alteryx', 'github', 'postgres', 'ssis', 'numpy', 'power_bi', 'spss', 'ssrs', 
'microstrategy',  'cognos', 'dax', 'matplotlib', 'dplyr', 'tidyr', 'ggplot2', 'plotly', 'esquisse', 'rshiny', 'mlr',
'docker', 'linux', 'jira',  'hadoop', 'airflow', 'redis', 'graphql', 'sap', 'tensorflow', 'node', 'asp.net', 'unix',
'jquery', 'pyspark', 'pytorch', 'gitlab', 'selenium', 'splunk', 'bitbucket', 'qlik', 'terminal', 'atlassian', 'unix/linux',
'linux/unix', 'ubuntu', 'nuix', 'datarobot',
]

keywords_cloud_tools = [
'aws', 'azure', 'gcp', 'snowflake', 'redshift', 'bigquery', 'aurora',
]

keywords = keywords_programming + keywords_libraries + keywords_analyst_tools + keywords_cloud_tools 

# Picked out keywords based on all keywords for the finance intern
soft_skills = [
    'interpersonal', 'editing', 'project_management', 'adaptable', 'proactive', 'curious', 'self_directed', 'flexible',
    'leader', 'communication', 'multitask', 'collaborative', 'meticulous', 'pro_active', 'attention_to_detail',
    'organization', 'prioritize', 'tech_savvy', 'decision_making', 'time_management', 'self_confident', 'self_starter', 'goal_oriented', 
    'administrative', 'strategic', 'logical', 'articulate', 'solution_forward', 'work_independently', 'accuracy', 'timeliness', 
    'critical_thinking', 'self_motivated', 'energetic' 'problem_solver', 'team_work', 'relationship_management', 'versatility'
]

# Analytical Skills
financial_skills = [
    'financial_reporting', 'financial_modeling', 'financial_analysis', 'valuation_techniques', 'ad_hoc_analysis', 
    'financial_planning', 'budgets', 'reconciliations', 'bookkeeping'
]

# Technical Skills
technical_skills = [
    'statistical_analysis', 'documentation', 'research', 'quantitative', 'analytical', 'data_driven', 'data_analysis', 'excel',
    'word', 'powerpoint', 'data_entry', 'r', 'stata', 'gis','canva', 'sql', 'access', 'power_bi', 'python', 'gaap', 'ifrs', 'outlook', 
    'data_science', 'javascript', 'node', 'react', 'noble_development', 'macros', 'erp', 'sap', 'tableau', 'spreadsheet', 'pdf',
    'google_suite', 'databases'
]

# Language Skills
language_skills = [
    'spanish', 'french', 'german', 'english'
]

# Total Skills
total_skills = soft_skills + financial_skills + technical_skills + language_skills

# Layout Section of Dash
app.layout = html.Div(children=[
    html.H1('Top Skills Needed for the MSc Students in Data Management for Finance 👩‍🎓',
            style={'textAlign': "center", 'color': '#0D0D0D',
                   'background-color': '#F0F0F0', 'padding': '10px',
                   'border-radius': '5px', 'margin': '10px',
                   'font': 'bold', 'font-size': '35px'}),
    html.Br(), # create space after heading
    html.Br(), # create space after heading
    html.Div([
    # customized the style of the text using markdown
    dcc.Markdown(children=markdown_text, style = {'text-align': 'justify', 'text-align-last': 'left', 
                                                  'margin-left' : '2.3em', 'margin-right' : '5em'}),
    ]),
    html.Br(), # create space after markdown
    html.Div([
        html.Div([
            html.Div([
                html.H2('Job Title:', style={'margin-right': '2em', 'margin-left' : '1em', 'padding' : '5px'}),
            ]),
            # customize the drop down for job title
            dcc.Dropdown(
                id='input-job-title',
                options=[
                    {'label': 'Data Analyst Intern', 'value': 'data analyst intern'},
                    {'label': 'Finance Intern', 'value': 'finance intern'}
                ],
                placeholder='Select Job Title',
                style={'padding': '2px', 'font-size': '20px', 'text-align-last': 'center', 'flex' : 0.855}
            ),
        ], style={'display': 'flex'}),
        html.Div([
            html.Div([
                html.H2('Skills:', style={'margin-right': '3.45em', 'margin-left' : '1em', 'padding' : '5px'})
            ]),
            # customize the drop down for skills
            dcc.Dropdown(
                id='input-skills',
                placeholder="Select Skill",
                style={ 'padding': '2px', 'font-size': '20px', 'text-align-last': 'center', 'flex' : 0.855}
            ),
        ], style={'display': 'flex'}),
    ]),
    html.Br(), # create space after dropdown
    html.Div([
        html.Div([], id='bar-plot-container'),
        html.Div([], id='map-plot-container'),
    ], style={'display': 'flex', 'flex-direction': 'row', 'justify-content': 'space-around', 
              'margin-right' : '2em', 'margin-left' : '2em', 'padding' : '5px'}),
])

# create callback component 
@app.callback(
    Output(component_id='input-skills', component_property='options'),
    [Input(component_id='input-job-title', component_property='value')]
)
# update the drop down for skills based on the job title
def update_skill_dropdown(job_title):
    if job_title == 'data analyst intern':
        return [
            {'label': 'Programming Languages', 'value': 'keywords_programming'},
            {'label': 'Libraries', 'value': 'keywords_libraries'},
            {'label': 'Analysis Tools', 'value': 'keywords_analyst_tools'},
            {'label': 'Cloud Tools', 'value': 'keywords_cloud_tools'},
            {'label': 'All Skills', 'value': 'keywords'}
        ]
    elif job_title == 'finance intern':
        return [
            {'label': 'Soft Skills', 'value': 'soft_skills'},
            {'label': 'Financial Skills', 'value': 'financial_skills'},
            {'label': 'Technical Skills', 'value': 'technical_skills'},
            {'label': 'Language Skills', 'value': 'language_skills'},
            {'label': 'All Skills', 'value': 'total_skills'}
        ]
    else:
        return ['Please Select the Job Title First']
# create callback component
@app.callback(
    [Output(component_id='bar-plot-container', component_property='children'),
     Output(component_id='map-plot-container', component_property='children')],
    [Input(component_id='input-job-title', component_property='value'),
     Input(component_id='input-skills', component_property='value')]
    
    #[State("bar-plot-container", "children"), State("map-plot-container", "children")]
)
# update the bar plot based on the job title and skills
def get_graph(job_title, skills):
    keyword_lists = {
        'keywords_programming': keywords_programming,
        'keywords_libraries': keywords_libraries,
        'keywords_analyst_tools': keywords_analyst_tools,
        'keywords_cloud_tools': keywords_cloud_tools,
        'keywords': keywords
    }
    
    finance_keyword_lists = {
        'soft_skills': soft_skills,
        'financial_skills': financial_skills,
        'technical_skills': technical_skills,
        'language_skills': language_skills,
        'total_skills': total_skills
    }

    if job_title == 'data analyst intern':
        # get keywords in a column
        count_keywords = pd.DataFrame(all_job[all_job['job_title'] == 'data analyst intern']\
                                      .description_tokens.sum()).value_counts()\
                                        .rename_axis('keywords').reset_index(name='counts')

        # get frequency of occurrence of word (as the word only appears once per line)
        length = len(all_job[all_job['job_title'] == 'data analyst intern'])   # number of job postings
        count_keywords['percentage'] = 100 * count_keywords.counts / length

        # create a map dataframe group by location and count the number of job
        map_data = all_job[all_job['job_title'] == 'data analyst intern'].groupby('job_loc')['job_title'].count().reset_index()


        # plot the results
        if skills in keyword_lists:
            count_keywords = count_keywords[count_keywords.keywords.isin(keyword_lists[skills])]
        else:
            return None
        # display only the top 10 keywords
        count_keywords = count_keywords.head(10)
        # Reverse the order of the data
        count_keywords = count_keywords[::-1]  
        # Color that i personally liked
        specific_colors = ["#5a189a", "#7b2cbf", "#9d4edd", "#c77dff", "#e0aaff", 
                           "#48bfe3", "#56cfe1", "#64dfdf", "#5aedc9", "#80ffdb"]
        # update the parameters for plotly bar chart
        bar_fig = go.Figure(go.Bar(
            x=count_keywords['percentage'],
            y=count_keywords['keywords'],
            orientation='h',
            marker=dict(color=specific_colors[:len(count_keywords.keywords)]),
            text=count_keywords['percentage'].apply(lambda x: f'{x:.2f}%'),  # Label percentages as text
            textposition='auto',  # Show labels inside the bars
        ))

        # update map parameters
        map_fig = px.choropleth(map_data, 
            locations='job_loc', 
            color='job_title',
            hover_data=['job_loc'], 
            locationmode='USA-states',  # Set to plot as US States
            color_continuous_scale= 'purp',
            labels={'job_title':'# of Job'},
            range_color=[0, map_data['job_title'].max()])

        # update map layout
        map_fig.update_layout(
            title_text = 'Location of Data Analyst Intern', 
            font = dict(size=16),
            legend=dict(x=0.029, y=1.038, font_size=10),
            margin=dict(l=100, r=20, t=70, b=30),
            geo_scope='usa') # Plot only the USA instead of globe
        
        # update the layout of the bar chart
        bar_fig.update_layout(
            title='Top Skills for Data Analyst Intern',
            font=dict(size=16),
            xaxis=dict(
                tickfont=dict(size=14),
                zeroline=False,
                showline=False,
                showticklabels=False,
                showgrid=False,
                domain=[0, 1],
            ),
            yaxis=dict(
                showgrid=False,
                showline=False,
                showticklabels=True,
                domain=[0, 1],
            ),
            legend=dict(x=0.029, y=1.038, font_size=10),
            margin=dict(l=100, r=20, t=70, b=30),
            #paper_bgcolor='rgb(248, 248, 255)',
            #plot_bgcolor='rgb(248, 248, 255)',
        )        
        return dcc.Graph(id='bar-plot',figure=bar_fig), dcc.Graph(id='map-plot', figure=map_fig)
    
    if job_title == 'finance intern':
        # get keywords in a column
        count_keywords = pd.DataFrame(all_job[all_job['job_title'] == 'finance intern']\
                                      .description_tokens.sum()).value_counts()\
                                        .rename_axis('keywords').reset_index(name='counts')

        # get frequency of occurrence of word (as the word only appears once per line)
        length = len(all_job[all_job['job_title'] == 'finance intern'])   # number of job postings
        count_keywords['percentage'] = 100 * count_keywords.counts / length

        # create a map dataframe group by location and count the number of job
        map_data = all_job[all_job['job_title'] == 'finance intern'].groupby('job_loc')['job_title'].count().reset_index()

        # plot the results
        if skills in finance_keyword_lists:
            count_keywords = count_keywords[count_keywords.keywords.isin(finance_keyword_lists[skills])]
        else:
            return None
        # display only the top 10 keywords
        count_keywords = count_keywords.head(10)
        # Reverse the order of the data
        count_keywords = count_keywords[::-1]
        # Color that i personally liked
        specific_colors = ["#5a189a", "#7b2cbf", "#9d4edd", "#c77dff", "#e0aaff", 
                           "#48bfe3", "#56cfe1", "#64dfdf", "#5aedc9", "#80ffdb"]
        
        # update map parameters
        map_fig = px.choropleth(map_data, 
            locations='job_loc', 
            color='job_title',
            hover_data=['job_loc'], 
            locationmode='USA-states',  # Set to plot as US States
            color_continuous_scale= 'purp',
            labels={'job_title':'# of Job'},
            range_color=[0, map_data['job_title'].max()])

        # update map layout
        map_fig.update_layout(
            title_text = 'Location of Finance Intern',
            font = dict(size=16), 
            legend=dict(x=0.029, y=1.038, font_size=10),
            margin=dict(l=100, r=20, t=70, b=30),
            geo_scope='usa') # Plot only the USA instead of globe
        # update the parameters for plotly bar chart
        bar_fig = go.Figure(go.Bar(
            x=count_keywords['percentage'],
            y=count_keywords['keywords'],
            orientation='h',
            marker=dict(color=specific_colors[:len(count_keywords.keywords)]),
            text=count_keywords['percentage'].apply(lambda x: f'{x:.2f}%'),  # Label percentages as text
            textposition='auto',  # Show labels inside the bars
        ))
        # update the layout of the bar chart
        bar_fig.update_layout(
            title='Top Skills for Finance Intern',
            font=dict(size=16),
            xaxis=dict(
                tickfont=dict(size=14),
                zeroline=False,
                showline=False,
                showticklabels=False,
                showgrid=False,
                domain=[0, 1],
            ),
            yaxis=dict(
                showgrid=False,
                showline=False,
                showticklabels=True,
                domain=[0, 1],
            ),
            legend=dict(x=0.029, y=1.038, font_size=10),
            margin=dict(l=100, r=20, t=70, b=30),
            #paper_bgcolor='rgb(248, 248, 255)',
            #plot_bgcolor='rgb(248, 248, 255)',
        )

        return dcc.Graph(id='bar-plot',figure=bar_fig), dcc.Graph(id='map-plot', figure=map_fig)

    return 'Please Select a Job Title and Skill above to see the chart below*'

if __name__ == '__main__':
    app.run_server(debug = False)
    
