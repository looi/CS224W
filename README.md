# CS224W

## How to run code

* Download Uber Movement data and put in this folder:
  * [San Francisco](https://movement.uber.com/cities/san_francisco/downloads/speeds?lang=en-US&tp[y]=2019&tp[q]=2):
    * `movement-speeds-quarterly-by-hod-san-francisco-2019-Q2.csv.zip`
  * [New York](https://movement.uber.com/cities/new_york/downloads/speeds?lang=en-US&tp[y]=2019&tp[q]=2):
    * `movement-speeds-quarterly-by-hod-new-york-2019-Q2.csv.zip`
  * [Seattle](https://movement.uber.com/cities/seattle/downloads/speeds?lang=en-US&tp[y]=2019&tp[q]=2):
    * `movement-speeds-quarterly-by-hod-seattle-2019-Q2.csv.zip`
* Install necessary packages with `conda env create -f environment.yml`
* Run `download_osm.py` to download OSM maps for San Francisco, New York, Seattle.

## Possible Next Steps

1. Get Google Cloud set up.

2. Get OpenStreetMap nodes/ways as a networkx graph using osmnx (or some other library).
  * For example: https://github.com/gboeing/osmnx-examples/blob/master/notebooks/06-example-osmnx-networkx.ipynb

3. Use Uber Movement data to add traffic information to the graph. We have the OpenStreetMap mapping so it should not be hard.

4. Convert graph from networkx to pytorch_geometric using [torch_geometric.utils.from_networkx](https://pytorch-geometric.readthedocs.io/en/latest/modules/utils.html#torch_geometric.utils.from_networkx)

5. Run GCN, GraphSAGE or whatever using pytorch geometric.
