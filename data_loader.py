import collections, re
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
import os.path
import re
import random
pd.set_option('mode.chained_assignment','raise')

# orig_speeds - original pd.DataFrame of speeds data
# speeds - merged pd.DataFrame of uber/osm speeds data
# orig_graph - original nx.Graph before merging
# graph - merged nx.Graph of uber/osm data
# gmmc - greedy modularity maximization communitites (of graph, not line graph)
# line_graph - line graph of graph
CityData = collections.namedtuple('CityData', 'orig_speeds speeds orig_graph graph gmmc line_graph')

def load_data_for_city(city, root_dir='.'):
    speeds_filename = 'movement-speeds-quarterly-by-hod-%s-2019-Q2.csv.zip' % city
    graph_filename = '%s.gpickle.gz' % city

    graph = nx.read_gpickle(os.path.join(root_dir, graph_filename))
    print('OSM MultiDiGraph has %d nodes, %d edges' %
        (nx.number_of_nodes(graph), nx.number_of_edges(graph)))
    graph = nx.Graph(graph)
    print('OSM Graph has %d nodes, %d edges' %
        (nx.number_of_nodes(graph), nx.number_of_edges(graph)))

    speeds = pd.read_csv(os.path.join(root_dir, speeds_filename))
    # Print basic stats on the data.
    speeds_num_rows = len(speeds)
    speeds_num_distinct_segment_ids = speeds['segment_id'].nunique()
    print('Speeds has %d rows, %d distinct segment IDs' %
        (speeds_num_rows, speeds_num_distinct_segment_ids))
    # Get p85 before dropping hours
    p85 = speeds.groupby('osm_way_id').mean()['speed_mph_p85']
    # Drop speeds with hour not in 7-10
    speeds_to_drop = (speeds['hour_of_day'] < 7) | (speeds['hour_of_day'] > 10)
    speeds.drop(speeds[speeds_to_drop].index, inplace=True)
    print('Dropped %d/%d Uber speeds with hour not in 7-10' % (speeds_to_drop.sum(), len(speeds_to_drop)))
    # For each OSM way ID, we'll just take the average of all the rows present
    speeds = speeds.groupby('osm_way_id').mean()[ ['speed_mph_mean', 'speed_mph_stddev']]
    speeds = speeds.join(p85)
    print('After processing, %d distinct OSM way IDs in the speeds dataset' % len(speeds))

    orig_graph = graph
    orig_speeds = speeds
    graph, speeds = merge_uber_osm_data(graph, speeds)
    print('After merging, graph has %d nodes, %d edges' %
        (nx.number_of_nodes(graph), nx.number_of_edges(graph)))

    # Do this after merging
    # Compute 1 - mean / p85 as the measure of traffic
    speeds['traffic'] = (1 - speeds['speed_mph_mean'] / speeds['speed_mph_p85']).clip(0.0, 1.0)
    # Group traffic into 5 classes like uber does
    speeds['traffic_class'] = speeds['traffic'].floordiv(0.2).astype('int')
    # Add traffic as edge attribute in the graph
    for v1, v2, edge in graph.edges(data=True):
        edge['traffic'] = speeds.loc[edge['osmid'], 'traffic']
        edge['traffic_class'] = speeds.loc[edge['osmid'], 'traffic_class']

    data = CityData(
        orig_speeds = orig_speeds,
        speeds = speeds,
        orig_graph = orig_graph,
        graph = graph,
        gmmc = nx.algorithms.community.greedy_modularity_communities(graph),
        line_graph = nx.line_graph(graph),
    )
    print('Greedy modularity maximization produced %d communities' % len(data.gmmc))
    print('Line graph has %d nodes, %d edges' %
        (nx.number_of_nodes(data.line_graph), nx.number_of_edges(data.line_graph)))
    return data

def load_cesna_communities(filename):
    cs = {}
    for line in open(filename):
        m = re.match(r'\D*(\d+)\D*(\d+)\D*(\d+)\D*', line)
        cs[(int(m.group(1)), int(m.group(2)))] = int(m.group(3))
    num_bidir = 0
    num_mismatch = 0
    for (a,b), c in cs.items():
        if (b,a) in cs:
            num_bidir += 1
            if cs[(b,a)] != c: 
                num_mismatch += 1
    print('Out of %d total roads, %d are bidirectional and out of those, %d have different communities for each direction' % (len(cs), num_bidir, num_mismatch))
    return cs

def add_cesna_communities_to_graph(cs, graph):
    total = 0
    notfound = 0
    for v1, v2, edge in graph.edges(data=True):
        total += 1
        if (v1,v2) in cs:
            edge['cesna'] = cs[(v1,v2)]
        elif (v2,v1) in cs:
            edge['cesna'] = cs[(v2,v1)]
        else:
            notfound += 1
            edge['cesna'] = -1
    print('Out of %d edges, %d not found in cesna communities' % (total, notfound))

def plot_map_with_cesna_communities(graph):
    graph = nx.MultiGraph(graph) # Need MultiGraph to plot
    attr_values = pd.Series([data['cesna'] for u, v, key, data in graph.edges(keys=True, data=True)])
    values = sorted(attr_values.drop_duplicates())
    color_map = ox.get_colors(len(values), cmap='jet')
    # Try to reduce likelihood of neighboring communities having similar color.
    random.shuffle(color_map)
    cv_map = {v:color_map[k] for k, v in enumerate(values)}
    edge_colors = attr_values.map(lambda x: cv_map[x])
    ox.plot_graph(graph, edge_color=edge_colors, fig_height=12)

def plot_map_with_gmmc_communities(data):
    print('Got %d communities' % len(data.gmmc))
    graph = nx.MultiGraph(data.graph) # Need MultiGraph to plot
    color_map = ox.get_colors(len(data.gmmc), cmap='jet')
    # Try to reduce likelihood of neighboring communities having similar color.
    random.shuffle(color_map)
    n2c = {}
    for c, ns in enumerate(data.gmmc):
        col = color_map[c]
        for n in ns:
            n2c[n] = col
    node_colors = [n2c[n] for n in graph.nodes()]
    ox.plot_graph(graph, node_color=node_colors, fig_height=12)

def plot_map_for_discrepancy(data):
    """Plots map showing discrepancy between Uber Movement and OSM data"""
    osm_way_ids_in_speeds_data = set(data.orig_speeds.index)

    graph = nx.MultiGraph(data.orig_graph) # Need MultiGraph to plot
    edge_colors = []
    edges_in_uber = 0
    total_edges = 0
    for v1, v2, edge in graph.edges(data=True):
        osmid = edge['osmid']
        edge_in_uber = False
        if isinstance(osmid, list):
            edge_in_uber = any(o in osm_way_ids_in_speeds_data for o in osmid)
        else:
            edge_in_uber = osmid in osm_way_ids_in_speeds_data
        edge_colors.append('black' if edge_in_uber else 'red')
        if edge_in_uber: edges_in_uber += 1
        total_edges += 1
    print('Out of %d total edges in the OSM graph, %d are in the Uber data' %
        (total_edges, edges_in_uber))
    ox.plot_graph(graph, edge_color=edge_colors, fig_height=12)

def plot_map_with_traffic(graph, attr='traffic_class', fig_height=12, drop_tertiary_roads=False):
    graph = nx.MultiGraph(graph) # Need MultiGraph to plot

    if drop_tertiary_roads:
        edges_to_drop = []
        for v1, v2, edge in graph.edges(data=True):
            highway = edge['highway']
            dropit = False
            if isinstance(highway, list):
                dropit = any(x.startswith('tertiary') or x.startswith('residential') for x in edge['highway'])
            else:
                dropit = highway.startswith('tertiary') or highway.startswith('residential')
            if dropit:
                edges_to_drop.append((v1, v2))
        graph.remove_edges_from(edges_to_drop)
        # Drop graph nodes with zero degree
        nodes_to_drop = [node for node, degree in graph.degree if degree==0]
        graph.remove_nodes_from(nodes_to_drop)

    # Plot percentiles instead of get_edge_colors_by_attr
    # which seems to have some issues
    #edge_colors = ox.get_edge_colors_by_attr(graph, attr, cmap='PiYG')
    #cmap = cm.get_cmap('PiYG')
    #vals = pd.Series([edge[attr] for _, _, edge in graph.edges(data=True)])
    #vals = vals.rank(pct=True)
    #vals = list(vals)
    #edge_colors = [cmap(x) for x in vals]
    def get_color(traffic):
        # use colors from uber movement
        return ('#43B982', '#EFD756', '#FF7D49', '#E54937', '#AE0000')[traffic]
    edge_colors = [get_color(edge[attr]) for _, _, edge in graph.edges(data=True)]
    fig, ax = ox.plot_graph(graph, edge_color=edge_colors, node_size=0, fig_height=fig_height, show=False, close=False)
    custom_lines = [
        Line2D([0], [0], color='#43B982', lw=4),
        Line2D([0], [0], color='#EFD756', lw=4),
        Line2D([0], [0], color='#FF7D49', lw=4),
        Line2D([0], [0], color='#E54937', lw=4),
        Line2D([0], [0], color='#AE0000', lw=4),
    ]
    attrs = pd.Series([edge[attr] for _, _, edge in graph.edges(data=True)])
    ax.legend(custom_lines, [
        '<20%% slower than free-flow (%.1f%% of segments)' % ((attrs==0).sum()/len(attrs)*100.0),
        '<40%% slower than free flow (%.1f%% of segments)' % ((attrs==1).sum()/len(attrs)*100.0),
        '<60%% slower than free-flow (%.1f%% of segments)' % ((attrs==2).sum()/len(attrs)*100.0),
        '<80%% slower than free-flow (%.1f%% of segments)' % ((attrs==3).sum()/len(attrs)*100.0),
        '<100%% slower than free-flow (%.1f%% of segments)' % ((attrs==4).sum()/len(attrs)*100.0),
    ])
    plt.show()

def plot_degree_distribution(graph, title):
    degrees = collections.Counter(degree for node, degree in graph.degree)
    deg, cnt = zip(*sorted(degrees.items()))
    plt.bar(deg, cnt)
    plt.title(title)
    plt.show()

def merge_uber_osm_data(graph, speeds):
    """Only keep data when edges are present in both Uber and OSM data"""
    graph = graph.copy()
    speeds = speeds.copy()
    osm_way_ids_in_speeds_data = set(speeds.index)
    osm_way_ids_to_keep = set()

    # Drop graph edges that aren't present in the Uber data
    edges_to_drop = []
    for v1, v2, edge in graph.edges(data=True):
        osmid = edge['osmid']
        edge_in_uber = False
        if isinstance(osmid, list):
            for o in osmid:
                if not o in osm_way_ids_in_speeds_data: continue
                osm_way_ids_to_keep.add(o)
                edge['osmid'] = o
                edge_in_uber = True
                break
        else:
            if osmid in osm_way_ids_in_speeds_data:
                osm_way_ids_to_keep.add(osmid)
                edge_in_uber = True
        if not edge_in_uber:
            edges_to_drop.append((v1, v2))
    graph.remove_edges_from(edges_to_drop)

    # Drop graph nodes with zero degree
    nodes_to_drop = [node for node, degree in graph.degree if degree==0]
    graph.remove_nodes_from(nodes_to_drop)
    print('Dropped %d vertices, %d edges from graph' % (len(nodes_to_drop), len(edges_to_drop)))

    # Drop Uber speeds rows that aren't in OSM data
    speeds_to_drop = set(speeds.index) - osm_way_ids_to_keep
    speeds.drop(speeds_to_drop, inplace=True)
    print('Dropped %d Uber speeds not in OSM map' % len(speeds_to_drop))
    return graph, speeds

def create_dataframe_for_baseline(data):
    def getprop(edge, prop, default):
        if prop not in edge: return default
        x = edge[prop]
        if isinstance(x, list):
            x = x[0]
        return x

    # Vertex to community ID
    n2c = {n:c for c, ns in enumerate(data.gmmc) for n in ns}

    rows = []
    road_types = {} # Use consistent road type for osmid
    for v1, v2, edge in data.graph.edges(data=True):
        maxspeed = 30 # Assume 30 by default
        m = re.match(r'(\d+).*', getprop(edge, 'maxspeed', ''))
        if m:
            maxspeed = int(m.group(1))
        lanes = getprop(edge, 'lanes', 1) # Assume 1 by default
        if isinstance(lanes, str):
            m = re.match(r'(\d+).*', lanes)
            if m:
                lanes = int(m.group(1))
            else:
                lanes = 1
        line_graph_node = (v1,v2) if data.line_graph.has_node((v1,v2)) else (v2,v1)
        if edge['osmid'] not in road_types:
            road_types[edge['osmid']] = get_road_type(edge)
        road_type = road_types[edge['osmid']]
        obj = {
            'osm_way_id': edge['osmid'],
            'traffic_class': edge['traffic_class'],
            'lanes': lanes,
            'length': getprop(edge, 'length', 0), # Assume 0 by default
            'maxspeed': maxspeed,
            'diff_community': 0 if n2c[v1] == n2c[v2] else 1,
            'degree': data.line_graph.degree[line_graph_node],
        }
        for i in range(13):
            obj['road_type_%d'%i] = 1 if road_type==i else 0
        rows.append(obj)
    df = pd.DataFrame(rows)
    # Just take the average of all fields over a given OSM way ID
    df = df.groupby('osm_way_id').mean()
    return df

def get_road_type(p3):
    # p3 is edge attribute dict
    result = p3['highway']
    if isinstance(result, list):
        if 'footway' in result or 'steps' in result or 'track' in result or 'path' in result:
            result = 'residential'
        elif 'residential' in result:
            result = 'residential'
        elif 'primary' in result:
            result = 'primary'
        elif 'secondary' in result:
            result = 'secondary'
        elif 'tertiary' in result:
            result = 'tertiary'
        elif 'motorway' in result:
            result = 'motorway'
        elif 'secondary_link' in result:
            result = 'secondary_link'
        elif 'tertiary_link' in result:
            result = 'tertiary_link'
        elif 'motorway_link' in result:
            result = 'motorway_link'
        elif 'service' in result:
            result = 'service'
    if result in ['living_street', 'pedestrian', 'razed']:
        result = 'residential'
    return {
        'residential': 0,
        'primary_link': 1,
        'service': 2,
        'motorway_link': 3,
        'trunk': 4,
        'secondary': 5,
        'motorway': 6,
        'secondary_link': 7,
        'trunk_link': 8,
        'unclassified': 9,
        'tertiary_link': 10,
        'primary': 11,
        'tertiary': 12,
    }[result]
