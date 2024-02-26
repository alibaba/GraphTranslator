import random
from ogb.nodeproppred import PygNodePropPredDataset
import torch_geometric.transforms as T
import torch
from torch_geometric.utils import degree
import pandas as pd
import numpy as np


dataset = PygNodePropPredDataset(name='ogbn-arxiv', root='./arxiv/', transform=T.ToSparseTensor())
row = dataset[0].adj_t.coo()[0]
col = dataset[0].adj_t.coo()[1]
src_node = torch.cat((row, col), dim=0)
dst_node = torch.cat((col, row), dim=0)
deg = degree(src_node, dtype=torch.long)
src_node = src_node.numpy()
dst_node = dst_node.numpy()
data = {'src_node': src_node, 'dst_node': dst_node}
arxiv_neighbor = pd.DataFrame(data)
print(deg,deg.shape)
print(arxiv_neighbor)
# 使用groupby优化
grouped_arxiv_neighbor = arxiv_neighbor.groupby('src_node')

# 定义一个空列表来存储每个组的采样结果
sample_neighbor_list = []

# 对每个组进行采样
for src, group in grouped_arxiv_neighbor:
    print(src)
    dst_list = group['dst_node'].to_list()
    deg_list = deg[dst_list].numpy()
    deg_sum = np.sum(deg_list)

    if len(dst_list) <= 5:
        sample = dst_list
    else:
        deg_prob = deg_list / deg_sum
        sample = random.choices(dst_list, weights=deg_prob, k=5)

    # 只保留采样的目标节点
    src_df = group[group['dst_node'].isin(sample)]
    sample_neighbor_list.append(src_df)

# 使用单次调用pd.concat来合并所有筛选后的DataFrame
sample_neighbor_df = pd.concat(sample_neighbor_list)

# 保存结果到CSV文件
# print(sample_neighbor_df)
print(sample_neighbor_list)
sample_neighbor_df.to_csv('sample_neighbor_df.csv')
