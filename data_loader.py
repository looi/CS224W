import collections
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
pd.set_option('mode.chained_assignment','raise')

CityData = collections.namedtuple('CityData', 'junctions segments speeds graph')

def load_data_for_city(city):
    junctions_filename = 'movement-junctions-to-osm-nodes-%s-2019.csv.zip' % city
    segments_filename = 'movement-segments-to-osm-ways-%s-2019.csv.zip' % city
    speeds_filename = 'movement-speeds-quarterly-by-hod-%s-2019-Q2.csv.zip' % city
    graph_filename = '%s.gpickle.gz' % city
    data = CityData(
        junctions = pd.read_csv(junctions_filename),
        segments = pd.read_csv(segments_filename),
        speeds = pd.read_csv(speeds_filename),
        graph = nx.read_gpickle(graph_filename),
    )
    # Print basic stats on the data.
    junctions_num_rows = len(data.junctions)
    assert junctions_num_rows == data.junctions['junction_id'].nunique()
    segments_num_rows = len(data.segments)
    assert segments_num_rows == data.segments['segment_id'].nunique()
    print('Uber<->OSM mapping has %d junction IDs and %d segment IDs' % 
        (junctions_num_rows, segments_num_rows))
    speeds_num_rows = len(data.speeds)
    speeds_num_distinct_segment_ids = data.speeds['segment_id'].nunique()
    print('Speeds has %d rows, %d distinct segment IDs' %
        (speeds_num_rows, speeds_num_distinct_segment_ids))
    print('OSM graph has %d nodes, %d edges' %
        (nx.number_of_nodes(data.graph), nx.number_of_edges(data.graph)))
    return data

def plot_map_for_discrepancy(data):
    """Plots map showing discrepancy between Uber Movement and OSM data"""
    segment_ids_in_speeds_data = data.speeds['segment_id'].drop_duplicates()
    osm_way_ids_in_speeds_data = set(data.segments.merge(segment_ids_in_speeds_data,
        how='inner', on='segment_id')['osm_way_id'])

    edge_colors = []
    edges_in_uber = 0
    total_edges = 0
    for v1, v2, edge in data.graph.edges(data=True):
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
    ox.plot_graph(data.graph, edge_color=edge_colors, fig_height=12)

def plot_degree_distribution(data):
    in_degrees = collections.Counter(in_degree for node, in_degree in data.graph.in_degree)
    out_degrees = collections.Counter(out_degree for node, out_degree in data.graph.out_degree)

    plt.figure(figsize=(10,4))
    plt.subplot(121)
    deg, cnt = zip(*sorted(in_degrees.items()))
    plt.plot(deg, cnt)
    plt.title('In degree distribution')

    plt.subplot(122)
    deg, cnt = zip(*sorted(out_degrees.items()))
    plt.plot(deg, cnt)
    plt.title('Out degree distribution')

def merge_uber_osm_data(data):
    """Only keep data when edges are present in both Uber and OSM data"""
    segment_ids_in_speeds_data = data.speeds['segment_id'].drop_duplicates()
    osm_way_ids_in_speeds_data = set(data.segments.merge(segment_ids_in_speeds_data,
        how='inner', on='segment_id')['osm_way_id'])
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

    # Drop Uber segments that aren't in OSM data
    segments_to_drop = ~data.segments['osm_way_id'].isin(osm_way_ids_to_keep)
    data.segments.drop(data.segments[segments_to_drop].index, inplace=True)

    # Drop Uber junctions that aren't in OSM data
    osm_node_ids = set(node['osmid'] for n, node in data.graph.nodes(data=True))
    junctions_to_drop = ~data.junctions['osm_node_id'].isin(osm_node_ids)
    data.junctions.drop(data.junctions[junctions_to_drop].index, inplace=True)
    print('Dropped %d Uber junctions and %d Uber segments' %
        (junctions_to_drop.sum(), segments_to_drop.sum()))
