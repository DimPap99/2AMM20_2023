import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import plotly.graph_objs as go 
import plotly.offline as py 
import math
from helpers import read_csv

def SetColor(y):
        if(y == 1):
            return "blue"
        elif(y == 2):
            return "red"
        elif(y == 3):
            return "yellow"

def visualize_graph_df(df_txs_features, df_txs_edgelist, plot_str):
    
    all_ids = df_txs_features['txId']

    short_edges = df_txs_edgelist[df_txs_edgelist['txId1'].isin(all_ids)]
    graph = nx.from_pandas_edgelist(short_edges, source = 'txId1', target = 'txId2', 
                                    create_using = nx.DiGraph())
    
    pos = nx.spring_layout(graph)
    df_txs_features['colors'] = df_txs_features['class'].apply(lambda x: "gray" if x==1 else ("Red" if x==2 else "green"))
    edge_x = []
    edge_y = []
    for edge in graph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='blue'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    node_text=[]
    for node in graph.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            color=df_txs_features.colors,
            size=10,
            colorbar=dict(
                thickness=15,
                title='Transaction graph',
                xanchor='left',
                titleside='right',
                tickmode='array',
                tickvals=[3, 1, 2],
                ticktext=['Unknown','Illicit','Licit']
            ),
            line_width=2))
    node_trace.text=node_text
    node_trace.marker.color = pd.to_numeric(df_txs_features[df_txs_features['txId'].isin(list(graph.nodes()))]['class'])

    fig = go.Figure(data=[edge_trace, node_trace],
                layout=go.Layout(
                    title=plot_str,
                    titlefont_size=16,
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    annotations=[ dict(
                        showarrow=True,
                        xref="paper", yref="paper",
                        x=0.005, y=-0.002 ) ],
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    fig.show()



def visualize_graph_from_list(class_pairs, edges, plot_str ):

    graph = nx.from_edgelist(edges, create_using = nx.DiGraph())
    edge_x = []
    edge_y = []
    pos = nx.spring_layout(graph)

    for edge in graph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='blue'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    node_text=[]
    for node in graph.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
    node_x = []
    node_y = []
    node_text=[]
    for node in graph.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        marker=dict(
            #color=df_txs_features.colors,
            size=10,
            colorbar=dict(
                thickness=15,
                title='Transaction graph',
                xanchor='left',
                titleside='right',
                tickmode='array',
                tickvals=[3, 1, 2],
                ticktext=['Unknown','Illicit','Licit']
            ),
            line_width=2))
    node_trace.text=node_text
    
    node_trace.marker.color = pd.to_numeric(class_pairs)

    fig = go.Figure(data=[edge_trace, node_trace],
                layout=go.Layout(
                    title=plot_str,
                    titlefont_size=16,
                    showlegend=False,
                    hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    annotations=[ dict(
                        showarrow=True,
                        xref="paper", yref="paper",
                        x=0.005, y=-0.002 ) ],
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    fig.show()
    print(len(graph.nodes()))