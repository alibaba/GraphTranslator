import time
from ogb.nodeproppred import PygNodePropPredDataset
import torch_geometric.transforms as T
import torch
from torch_geometric.data import Data
import torch.nn.functional as F
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.nn import SAGEConv


bert_node_embeddings = torch.load("../../data/arxiv/bert_node_embeddings.pt")

dataset = PygNodePropPredDataset(name='ogbn-arxiv', root='./arxiv/', transform=T.ToSparseTensor())
row = dataset[0].adj_t.coo()[0]
col = dataset[0].adj_t.coo()[1]
src_node = torch.cat((row, col), dim=0)
dst_node = torch.cat((col, row), dim=0)
edge_index = torch.stack((src_node, dst_node), dim=0)
edge_index.shape

split_idx = dataset.get_idx_split()
train_idx, valid_idx, test_idx = split_idx['train'], split_idx['valid'], split_idx['test']

data = Data(x=bert_node_embeddings, edge_index=edge_index, train_idx=train_idx, valid_idx=valid_idx, test_idx=test_idx, y=dataset[0].y.squeeze())
train_loader = LinkNeighborLoader(
    data,
    batch_size=65536,
    shuffle=True,
    neg_sampling_ratio=1.0,
    num_neighbors=[10, 10],
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data = data.to(device)


class Net(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super(Net, self).__init__()
        self.conv1 = SAGEConv(in_dim, hid_dim)
        self.conv2 = SAGEConv(hid_dim, out_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return x


model = Net(768, 1024, 768).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


def train():
    model.train()

    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        h = model(batch.x, batch.edge_index)
        h_src = h[batch.edge_label_index[0]]
        h_dst = h[batch.edge_label_index[1]]
        pred = (h_src * h_dst).sum(dim=-1)
        loss = F.binary_cross_entropy_with_logits(pred, batch.edge_label)
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * pred.size(0)

    return total_loss / data.num_nodes


class LogisticRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear1 = nn.Linear(input_dim, 512)
        self.linear2 = nn.Linear(512, output_dim)
        self.act = nn.ReLU()

    def forward(self, x):
        out = self.linear1(x)
        out = self.act(out)
        out = self.linear2(out)
        return out


def compute_accuracy(outputs, labels):
    _, predicted = torch.max(outputs, 1)
    correct = (predicted == labels).sum().item()
    total = labels.size(0)
    accuracy = correct / total
    return accuracy


def test():
    with torch.no_grad():
        model.eval()
        out = model(data.x, data.edge_index)

    for epoch in range(1, 501):
        LR_model.train()
        optimizer.zero_grad()
        pred = LR_model(out[data.train_idx])

        label = F.one_hot(data.y[data.train_idx], 40).float()

        loss = criterion(pred, label)
        loss.backward()
        optimizer.step()

    LR_model.eval()
    val_outputs = LR_model(out[data.valid_idx])
    val_acc = compute_accuracy(val_outputs, data.y[data.valid_idx])

    test_outputs = LR_model(out[data.test_idx])
    test_acc = compute_accuracy(test_outputs, data.y[data.test_idx])

    return val_acc, test_acc


times = []
best_acc = 0
for epoch in range(10):
    start = time.time()
    input_dim = 768
    output_dim = torch.max(data.y) + 1
    LR_model = LogisticRegression(input_dim, output_dim).to(device)
    optimizer = torch.optim.Adam(LR_model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    loss = train()
    print("loss:", loss)

out = model(data.x, data.edge_index)
torch.save(out, "../../data/arxiv/graphsage_node_embeddings.pt")
