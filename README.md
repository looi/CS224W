# CS224W

## Possible Next Steps

1. Get Google Cloud set up and download Uber Movement data:
  * Quarterly Speeds Statistics by Hour of Day (Q2 2019)
  * Movement Segments to OSM Ways
  * Movement Junctions to OSM Nodes

2. Get OpenStreetMap nodes/ways as a networkx graph using osmnx (or some other library).
  * For example: https://github.com/gboeing/osmnx-examples/blob/master/notebooks/06-example-osmnx-networkx.ipynb

3. Use Uber Movement data to add traffic information to the graph. We have the OpenStreetMap mapping so it should not be hard.

4. Convert graph from networkx to pytorch_geometric using [torch_geometric.utils.from_networkx](https://pytorch-geometric.readthedocs.io/en/latest/modules/utils.html#torch_geometric.utils.from_networkx)

5. Run GCN, GraphSAGE or whatever using pytorch geometric.
