import networkx as nx
import osmnx as ox
import os.path

to_download = [
    ('san-francisco', 'San Francisco, California, USA'),
    ('new-york', 'New York, New York, USA'),
    ('seattle', 'Seattle, Washington, USA'),
]

for name, query in to_download:
    outfile = '%s.gpickle.gz' % name
    if os.path.exists(outfile):
        print(outfile, 'already exists')
        continue
    graph = ox.graph_from_place(query)
    nx.write_gpickle(graph, outfile)
