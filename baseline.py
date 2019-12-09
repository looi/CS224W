import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import data_loader

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
