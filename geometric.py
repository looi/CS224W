import collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils
from sklearn.metrics import confusion_matrix
import networkx as nx
import data_loader

class TrafficSageNet(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim, reducer):
        super(TrafficSageNet, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GraphSage(input_dim, hidden_dim, reducer))
        assert (num_layers >= 1), 'Number of layers is not >=1'
        for l in range(num_layers-1):
            self.convs.append(GraphSage(hidden_dim, hidden_dim, reducer))
        self.lin1 = nn.Linear(hidden_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, output_dim)
        self.num_layers = num_layers

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)

        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=1)

class GraphSage(pyg_nn.MessagePassing):
    """Non-minibatch version of GraphSage."""
    def __init__(self, in_channels, out_channels, reducer, 
                 normalize_embedding=True):
        super(GraphSage, self).__init__(aggr=reducer)

        self.lin = nn.Linear(in_channels, out_channels)
        #self.lin1 = nn.Linear(in_channels, 128)
        #self.lin2 = nn.Linear(128, out_channels)
        self.agg_lin = nn.Linear(in_channels + out_channels, out_channels)
        #self.agg_lin1 = nn.Linear(in_channels + out_channels, 128)
        #self.agg_lin2 = nn.Linear(128, out_channels)

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
        #x_j = self.lin1(x_j)
        #x_j = F.relu(x_j)
        #x_j = self.lin2(x_j)
        #x_j = F.relu(x_j)

        return x_j

    def update(self, aggr_out, x):
        # aggr_out has shape [N, out_channels]
        # x has shape [N, in_channels]
        
        aggr_out = self.agg_lin(torch.cat([x, aggr_out], dim=1))
        aggr_out = F.relu(aggr_out)
        #aggr_out = self.agg_lin1(torch.cat([x, aggr_out], dim=1))
        #aggr_out = F.relu(aggr_out)
        #aggr_out = self.agg_lin2(aggr_out)
        #aggr_out = F.relu(aggr_out)
        if self.normalize_emb:
            aggr_out = F.normalize(aggr_out)

        return aggr_out

GNNData = collections.namedtuple('GNNData', 'tg_graph traffic_classes class_weights node_map')

def create_gnn_data_structures(data):
    df = data_loader.create_dataframe_for_baseline(data)
    # line_graph will return nodes like (a, b)
    # But we need integer node IDs for torch_geometric
    node_map = {}
    new_graph = nx.empty_graph(0, default=data.line_graph.__class__)
    for a, b in data.line_graph:
        idx = len(node_map)
        node_map[a,b] = idx
        new_graph.add_node(idx)
    for (a, b), (c, d) in data.line_graph.edges():
        new_graph.add_edge(node_map[a,b], node_map[c,d])
    traffic_classes = [None]*new_graph.number_of_nodes()
    attrs = ['lanes', 'length', 'diff_community', 'degree'] + ['road_type_%d'%i for i in range(13)]
    for v1, v2, edge in data.graph.edges(data=True):
        # line_graph will use vertices (a1,a2) where a1<a2
        a1, a2 = v1, v2
        if a1 > a2: a1, a2 = a2, a1
        new_dict = new_graph.nodes[node_map[a1,a2]]
        for y in attrs:
            new_dict[y] = df.loc[edge['osmid'], y]
        traffic_classes[node_map[a1,a2]] = df.loc[edge['osmid'], 'traffic_class']
    tg_graph = torch_geometric.utils.from_networkx(new_graph)
    tg_graph['x'] = torch.stack([tg_graph[y] for y in attrs]).transpose(0,1)

    vc = df['traffic_class'].value_counts()
    print(vc)
    tot = vc.sum()
    class_weights = {k:tot/v for k,v in vc.iteritems()}
    class_weights = [class_weights[i] for i in range(len(class_weights))]
    print(class_weights)

    return GNNData(tg_graph=tg_graph, traffic_classes=traffic_classes, class_weights=class_weights, node_map=node_map)

def train_gnn_model(gds, num_layers, num_epochs, hidden_dim, output_dim, reducer):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = gds[0].tg_graph['x'].shape[1]
    model = TrafficSageNet(num_layers, input_dim, hidden_dim, output_dim, reducer).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Weight each graph inverse proportional to its size
    graph_sizes = np.array([len(gd.traffic_classes) for gd in gds], dtype=np.int64)
    weights = graph_sizes.sum() / graph_sizes
    weights = np.repeat(weights, graph_sizes)
    weights = torch.tensor(weights, device=device, dtype=torch.float)

    traffic_classes = torch.cat([torch.tensor(gd.traffic_classes, device=device).to(torch.int64) for gd in gds], dim=0)
    tg_graphs = torch_geometric.data.Batch.from_data_list([gd.tg_graph for gd in gds]).to(device)
    class_weights = torch.stack([torch.tensor(gd.class_weights, device=device) for gd in gds]).sum(dim=0)

    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        out = model(tg_graphs)
        #loss = F.mse_loss(out, traffic_values)
        loss = (F.nll_loss(out, traffic_classes, weight=class_weights, reduction='none') * weights).mean()
        accuracy = (out.argmax(dim=1)==traffic_classes).float().mean()
        print('Epoch %03d: loss = %.5f, accuracy = %.5f' % (epoch, loss, accuracy))
        loss.backward()
        optimizer.step()
    # Return model in eval mode
    model.eval()
    return model

def save_gnn_models(gnn_models, filename_prefix):
    for city, model in gnn_models.items():
        torch.save(model.state_dict(), filename_prefix+city+'.pth')

def plot_map_with_gnn_predicted_traffic(data, gd, model):
    # max(dim=1) returns values, indices tuple; only need indices
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tg_graph = gd.tg_graph.to(device)
    traffic_classes = torch.tensor(gd.traffic_classes, device=device)
    out = model(tg_graph).detach().max(dim=1)[1]
    print('Accuracy =', float((out==traffic_classes).to(torch.float64).mean()))
    traffic_classes = traffic_classes.cpu()
    out = out.cpu()
    cm = confusion_matrix(traffic_classes, out)
    print('Accuracy confusion matrix\n', cm)
    per_class_accuracy = cm[np.arange(4), np.arange(4)]/np.sum(cm, axis=1)
    print('Per-class accuracy:', per_class_accuracy.mean(), per_class_accuracy)
    for v1, v2, edge in data.graph.edges(data=True):
        # line_graph will use vertices (a1,a2) where a1<a2
        a1, a2 = v1, v2
        if a1 > a2: a1, a2 = a2, a1
        edge['traffic_class_predicted'] = int(out[gd.node_map[a1,a2]])
    data_loader.plot_map_with_traffic(data.graph, 'traffic_class_predicted')
