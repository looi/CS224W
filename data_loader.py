import collections, re
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
pd.set_option('mode.chained_assignment','raise')

CityData = collections.namedtuple('CityData', 'speeds graph')

def load_data_for_city(city):
    speeds_filename = 'movement-speeds-quarterly-by-hod-%s-2019-Q2.csv.zip' % city
    graph_filename = '%s.gpickle.gz' % city

    graph = nx.read_gpickle(graph_filename)
    print('OSM MultiDiGraph has %d nodes, %d edges' %
        (nx.number_of_nodes(graph), nx.number_of_edges(graph)))
    graph = nx.Graph(graph)
    print('OSM Graph has %d nodes, %d edges' %
        (nx.number_of_nodes(graph), nx.number_of_edges(graph)))

    speeds = pd.read_csv(speeds_filename)
    # Print basic stats on the data.
    speeds_num_rows = len(speeds)
    speeds_num_distinct_segment_ids = speeds['segment_id'].nunique()
    print('Speeds has %d rows, %d distinct segment IDs' %
        (speeds_num_rows, speeds_num_distinct_segment_ids))
    # Drop speeds with hour not in 7-10
    speeds_to_drop = (speeds['hour_of_day'] < 7) | (speeds['hour_of_day'] > 10)
    speeds.drop(speeds[speeds_to_drop].index, inplace=True)
    print('Dropped %d/%d Uber speeds with hour not in 7-10' % (speeds_to_drop.sum(), len(speeds_to_drop)))
    # For each OSM way ID, we'll just take the average of all the rows present
    speeds = speeds.groupby('osm_way_id').mean()[
        ['speed_mph_mean', 'speed_mph_stddev', 'speed_mph_p50', 'speed_mph_p85']]
    # Compute mean / p85 as the measure of traffic
    speeds['traffic'] = speeds['speed_mph_mean'] / speeds['speed_mph_p85']
    print('After processing, %d distinct OSM way IDs in the speeds dataset' % len(speeds))
    data = CityData(
        speeds = speeds,
        graph = graph,
    )
    return data

def plot_map_for_discrepancy(data):
    """Plots map showing discrepancy between Uber Movement and OSM data"""
    osm_way_ids_in_speeds_data = set(data.speeds.index)

    graph = nx.MultiGraph(data.graph) # Need MultiGraph to plot
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

def plot_map_with_traffic(graph, attr='traffic'):
    graph = nx.MultiGraph(graph) # Need MultiGraph to plot
    edge_colors = ox.get_edge_colors_by_attr(graph, attr, cmap='PiYG')
    ox.plot_graph(graph, edge_color=edge_colors, fig_height=12)

def plot_degree_distribution(graph):
    degrees = collections.Counter(degree for node, degree in graph.degree)
    deg, cnt = zip(*sorted(degrees.items()))
    plt.bar(deg, cnt)
    plt.title('Degree distribution')

def merge_uber_osm_data(data):
    """Only keep data when edges are present in both Uber and OSM data"""
    osm_way_ids_in_speeds_data = set(data.speeds.index)
    osm_way_ids_to_keep = set()

    # Drop graph edges that aren't present in the Uber data
    edges_to_drop = []
    for v1, v2, edge in data.graph.edges(data=True):
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
    data.graph.remove_edges_from(edges_to_drop)

    # Drop graph nodes with zero degree
    nodes_to_drop = [node for node, degree in data.graph.degree if degree==0]
    data.graph.remove_nodes_from(nodes_to_drop)
    print('Dropped %d vertices, %d edges from graph' % (len(nodes_to_drop), len(edges_to_drop)))

    # Drop Uber speeds rows that aren't in OSM data
    speeds_to_drop = set(data.speeds.index) - osm_way_ids_to_keep
    data.speeds.drop(speeds_to_drop, inplace=True)
    print('Dropped %d Uber speeds not in OSM map' % len(speeds_to_drop))

    # Add traffic as edge attribute in the graph
    for v1, v2, edge in data.graph.edges(data=True):
        edge['traffic'] = data.speeds.loc[edge['osmid'], 'traffic']

def create_dataframe_for_baseline(data):
    def getprop(edge, prop, default):
        if prop not in edge: return default
        x = edge[prop]
        if isinstance(x, list):
            x = x[0]
        return x

    rows = []
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
        rows.append({
            'osm_way_id': edge['osmid'],
            'lanes': lanes,
            'length': getprop(edge, 'length', 0), # Assume 0 by default
            'maxspeed': maxspeed,
            'traffic': edge['traffic'],
        })
    df = pd.DataFrame(rows)
    # Just take the average of all fields over a given OSM way ID
    df = df.groupby('osm_way_id').mean()
    return df
