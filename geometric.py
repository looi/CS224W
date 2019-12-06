import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils
import networkx as nx
import data_loader

class TrafficSageNet(nn.Module):
    def __init__(self):
        super(TrafficSageNet, self).__init__()
        self.conv1 = GraphSage(3, 16)
        self.conv2 = GraphSage(16, 16)
        self.lin1 = nn.Linear(16, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        return self.lin1(x)

class GraphSage(pyg_nn.MessagePassing):
    """Non-minibatch version of GraphSage."""
    def __init__(self, in_channels, out_channels, reducer='mean', 
                 normalize_embedding=True):
        super(GraphSage, self).__init__(aggr=reducer)

        self.lin = nn.Linear(in_channels, out_channels)
        self.agg_lin = nn.Linear(in_channels + out_channels, out_channels)

        if normalize_embedding:
            self.normalize_emb = True

    def forward(self, x, edge_index):
        num_nodes = x.size(0)
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        return self.propagate(edge_index, size=(num_nodes, num_nodes), x=x)

    def message(self, x_j, edge_index, size):
        # x_j has shape [E, in_channels]
        # edge_index has shape [2, E]

        x_j = self.lin(x_j)
        x_j = F.relu(x_j)

        return x_j

    def update(self, aggr_out, x):
        # aggr_out has shape [N, out_channels]
        # x has shape [N, in_channels]
        
        aggr_out = self.agg_lin(torch.cat([x, aggr_out], dim=1))
        aggr_out = F.relu(aggr_out)
        if self.normalize_emb:
            aggr_out = F.normalize(aggr_out)

        return aggr_out

def create_line_graph_for_torch_geometric(data):
    df = data_loader.create_dataframe_for_baseline(data)
    line_graph = nx.line_graph(data.graph)
    # line_graph will return nodes like (a, b)
    # But we need integer node IDs for torch_geometric
    node_map = {}
    new_graph = nx.empty_graph(0, default=line_graph.__class__)
    for a, b in line_graph:
        idx = len(node_map)
        node_map[a,b] = idx
        new_graph.add_node(idx)
    for (a, b), (c, d) in line_graph.edges():
        new_graph.add_edge(node_map[a,b], node_map[c,d])
    traffic_values = [None]*new_graph.number_of_nodes()
    for v1, v2, edge in data.graph.edges(data=True):
        # line_graph will use vertices (a1,a2) where a1<a2
        a1, a2 = v1, v2
        if a1 > a2: a1, a2 = a2, a1
        new_dict = new_graph.nodes[node_map[a1,a2]]
        new_dict['lanes'] = df.loc[edge['osmid'], 'lanes']
        new_dict['length'] = df.loc[edge['osmid'], 'length']
        new_dict['maxspeed'] = df.loc[edge['osmid'], 'maxspeed']
        traffic_values[node_map[a1,a2]] = df.loc[edge['osmid'], 'traffic']
    tg_graph = torch_geometric.utils.from_networkx(new_graph)
    tg_graph['x'] = torch.stack((tg_graph['lanes'], tg_graph['length'], tg_graph['maxspeed'])).transpose(0,1)
    return tg_graph, traffic_values, node_map

def train_model(tg_graph, traffic_values, num_epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TrafficSageNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    traffic_values = torch.tensor(traffic_values, device=device).reshape((-1, 1))
    tg_graph = tg_graph.to(device)

    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        out = model(tg_graph)
        loss = F.mse_loss(out, traffic_values)
        print('Epoch %03d: MSE %.2f' % (epoch, loss))
        loss.backward()
        optimizer.step()
    # Return model in eval mode
    model.eval()
    return model

def plot_map_with_predicted_traffic(data, tg_graph, node_map, model):
    out = model(tg_graph).detach().cpu()
    for v1, v2, edge in data.graph.edges(data=True):
        # line_graph will use vertices (a1,a2) where a1<a2
        a1, a2 = v1, v2
        if a1 > a2: a1, a2 = a2, a1
        edge['traffic_predicted'] = float(out[node_map[a1,a2],0])
    data_loader.plot_map_with_traffic(data.graph, 'traffic_predicted')
