import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import data_loader
import torch_geometric
import torch_geometric.nn as pyg_nn

class BaselineGCN(nn.Module):
    def __init__(self, input_dim, model):
        super(BaselineGCN, self).__init__()
        self.conv = MeanMP()
        self.post_model = model

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv(x, edge_index)
        
        return self.post_model(x)

class MeanMP(pyg_nn.MessagePassing):
    """Just compute mean"""
    def __init__(self, ):
        super(MeanMP, self).__init__(aggr = 'mean')


    def forward(self, x, edge_index):
        num_nodes = x.size(0)
        return self.propagate(edge_index, size=(num_nodes, num_nodes), x=x)

    # Message is just feature
    def message(self, x_j, edge_index, size):
        return x_j

    def update(self, aggr_out, x):
        return aggr_out

class SimpleNeuralNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, num_hidden_layers, hidden_dim):
        super(SimpleNeuralNetwork, self).__init__()
        self.layers = nn.ModuleList()
        for l in range(num_hidden_layers):
            self.layers.append(nn.Linear(input_dim if l==0 else hidden_dim, hidden_dim))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(p=0.5))
        self.layers.append(nn.Linear(input_dim if num_hidden_layers==0 else hidden_dim, output_dim))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return F.log_softmax(x, dim=1)

# Need to feed in same gnn data
def train_gcn_baseline(gds, output_dim, num_hidden_layers, num_epochs, hidden_dim = None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = gds[0].tg_graph['x'].shape[1]
    linear_model = SimpleNeuralNetwork(input_dim, output_dim, num_hidden_layers, hidden_dim).to(device)
    model = BaselineGCN(input_dim, linear_model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    traffic_classes = torch.cat([torch.tensor(gd.traffic_classes, device=device).to(torch.int64) for gd in gds], dim=0)
    tg_graphs = torch_geometric.data.Batch.from_data_list([gd.tg_graph for gd in gds]).to(device)
    class_weights = torch.stack([torch.tensor(gd.class_weights, device=device) for gd in gds]).sum(dim=0)

    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        out = model(tg_graphs)
        #loss = F.mse_loss(out, traffic_values)
        loss = F.nll_loss(out, traffic_classes, weight=class_weights)
        accuracy = (out.argmax(dim=1)==traffic_classes).float().mean()
        print('Epoch %03d: loss = %.5f, accuracy = %.5f' % (epoch, loss, accuracy))
        loss.backward()
        optimizer.step()
    # Return model in eval mode
    model.eval()
    return model

def train_neural_network(data, output_dim, num_hidden_layers, num_epochs, hidden_dim=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    linear_data = pd.concat(
        [data_loader.create_dataframe_for_baseline(df) for df in data],
        ignore_index=True)
    train_X = torch.tensor(linear_data[['lanes', 'length', 'maxspeed', 'diff_community', 'degree']].to_numpy(), device=device, dtype=torch.float)
    #scaler = StandardScaler()
    #scaler.fit(train_X)
    #train_X = scaler.transform(train_X)
    vc = linear_data['traffic_class'].value_counts()
    print(vc)
    tot = vc.sum()
    class_weights = {k:tot/v for k,v in vc.iteritems()}
    class_weights = [class_weights[i] for i in range(len(class_weights))]
    print(class_weights)

    model = SimpleNeuralNetwork(train_X.shape[1], output_dim, num_hidden_layers, hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    traffic_classes = torch.tensor(linear_data['traffic_class'].to_numpy(), device=device, dtype=torch.int64)
    class_weights = torch.tensor(class_weights, device=device)

    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        out = model(train_X)
        #loss = F.mse_loss(out, traffic_values)
        loss = F.nll_loss(out, traffic_classes, weight=class_weights)
        accuracy = (out.argmax(dim=1)==traffic_classes).float().mean()
        print('Epoch %03d: loss = %.5f, accuracy = %.5f' % (epoch, loss, accuracy))
        loss.backward()
        optimizer.step()
    # Return model in eval mode
    model.eval()
    return model

def plot_map_with_neural_network_predicted_traffic(data, model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    linear_data = data_loader.create_dataframe_for_baseline(data)
    x = torch.tensor(linear_data[['lanes', 'length', 'maxspeed', 'diff_community', 'degree']].to_numpy(), device=device, dtype=torch.float)
    traffic_classes = torch.tensor(linear_data['traffic_class'].to_numpy(), device=device, dtype=torch.int64)
    # max(dim=1) returns values, indices tuple; only need indices
    out = model(x).detach().max(dim=1)[1]
    print('Accuracy =', float((out==traffic_classes).to(torch.float64).mean()))
    traffic_classes = traffic_classes.cpu()
    out = out.cpu()
    cm = confusion_matrix(traffic_classes, out)
    print('Accuracy confusion matrix\n', cm)
    per_class_accuracy = cm[np.arange(4), np.arange(4)]/np.sum(cm, axis=1)
    print('Per-class accuracy:', per_class_accuracy.mean(), per_class_accuracy)
    osmid_to_idx = {x:i for i,x in enumerate(linear_data.index)}
    for v1, v2, edge in data.graph.edges(data=True):
        edge['traffic_class_predicted'] = int(out[osmid_to_idx[edge['osmid']]])
    data_loader.plot_map_with_traffic(data.graph, 'traffic_class_predicted')


def plot_map_with_gnn_predicted_traffic_gcn(data, gd, model):
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
