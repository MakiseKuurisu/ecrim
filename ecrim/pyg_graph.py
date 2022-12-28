import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from torch.nn import Linear
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
import pdb

class Attention(nn.Module):
    def __init__(self , h , d_model):
        super().__init__()

        assert d_model % h == 0

        self.d_model = d_model
        self.h = h
        self.dk = d_model // h

        self.WQ = nn.Linear(self.dk , self.dk)
        self.WK = nn.Linear(self.dk , self.dk)
        self.WV = nn.Linear(self.dk , self.dk)		

        self.reset_params()

    def reset_params(self):
        nn.init.xavier_normal_(self.WQ.weight.data)
        nn.init.xavier_normal_(self.WK.weight.data)
        nn.init.xavier_normal_(self.WV.weight.data)

        nn.init.constant_(self.WQ.bias.data , 0)
        nn.init.constant_(self.WK.bias.data , 0)
        nn.init.constant_(self.WV.bias.data , 0)

    def forward(self , S , R , S_mas , R_mas):
        '''
            S: (dps, seq_len, d)
            R: (dps, 1, d)

            S_mas: (dps, seq_len, 1)
            R_mas: (dps, 1, 1)
        '''

        h , dk = self.h , self.dk
        dps, seq_len, d = S.size()
        assert d == self.d_model or d == self.reduced_dim

        #pdb.set_trace()
        S = S.view(dps,seq_len,h,dk).permute(0,2,1,3).contiguous() #(dps, h, seq_len, dk)
        S_mas = S_mas.view(dps, 1, seq_len, 1)
        R = R.view(dps,1,h,dk).permute(0,2,1,3).contiguous()      #(dps , h , 1 , dk)
        R_mas = R_mas.view(dps,1,1,1)

        S_Q , S_K , S_V = self.WQ(S) , self.WK(S) , self.WV(S)
        R_Q , R_K , R_V = self.WQ(R) , self.WK(R) , self.WV(R)
        

        #from E to R
        beta = (S_Q * R_K.view(dps,h,1,dk)).sum(-1 , keepdim = True)

        S_Z = R_V.view(dps,h,1,dk) * beta.view(dps,h,seq_len,1) 


        S_Z = S_Z.masked_fill(~S_mas.expand(S_Z.size()).bool() , 0)

        S_Z = S_Z.view(dps,h,seq_len,dk).permute(0,2,1,3).contiguous().view(dps,seq_len,h*dk)

        return S_Z

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels, dim_node_features, num_classes):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(dim_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.clf = Linear(hidden_channels, num_classes)
        
    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        
        return x
def create_edges_sigle(r_node_lists, device):
    edge_list = []
    p_node_list = {}
    p_num = len(r_node_lists)
    r_node_start = p_num
    for p_idx, r_nodes in enumerate(r_node_lists):
        p_node_list[p_idx] = [i for i in range(r_node_start, r_node_start+len(r_nodes))]
        r_node_start += len(r_nodes)
    # P-R and R-P
    for p, r_list in p_node_list.items():
        source_node = [p] * len(r_list)
        destination_node = r_list
        pair = [[s, d] for s,d in zip(source_node, destination_node)]
        pair_reverse = [[d, s] for s,d in zip(source_node, destination_node)]
        edge_list.extend(pair+pair_reverse)
    # P-P
    p_p_edges = [[i, j] for i in range(p_num) for j in range(p_num) if i!=j]
    #print(p_p_edges)
    edge_list.extend(p_p_edges)
    edges = torch.tensor(np.array(edge_list)).t().to(device)
    #print(edges.shape)
    #print(edges)
    return edges
    

def load_features_single(r_embs, p_embs, dk):
    node_features = torch.zeros(len(r_embs)+len(p_embs), dk).to(p_embs.device)
    p_num = p_embs.size()[0]
    node_features[:p_num] = p_embs
    node_features[p_num:] = r_embs
    node_features.to(p_embs.device)
    return node_features


def create_graph_single(r_list, r_embs, p_embs):
    device = p_embs.device
    edge_index = create_edges_sigle(r_list, device)
    node_features = load_features_single(r_embs, p_embs, dk=p_embs.size()[-1])
    data = Data(x=node_features, edge_index=edge_index)
    return data


def create_edges(r_node_lists, device):
    def get_p_b_edges(n, b_start):
        edges = []
        for i in range(n):
            up = (i+1)*n -1 
            bottom = i*n
            for j in range(bottom, up+1):
                if j!= (i*n +i):
                    edges.append([i, j+b_start])
                    edges.append([j+b_start, i])
                else:
                    continue
        return edges
        
    edge_list = []
    p_node_list = {}
    p_num = len(r_node_lists)
    r_node_start = p_num
    for p_idx, r_nodes in enumerate(r_node_lists):
        p_node_list[p_idx] = [i for i in range(r_node_start, r_node_start+len(r_nodes))]
        r_node_start += len(r_nodes)
    # P-R and R-P
    for p, r_list in p_node_list.items():
        source_node = [p] * len(r_list)
        destination_node = r_list
        pair = [[s, d] for s,d in zip(source_node, destination_node)]
        pair_reverse = [[d, s] for s,d in zip(source_node, destination_node)]
        edge_list.extend(pair+pair_reverse)
    # P-P
    # p_p_edges = [[i, j] for i in range(p_num) for j in range(p_num) if i!=j]
    # edge_list.extend(p_p_edges)
    # print(p_p_edges)

    # P-B
    b_node_start = r_node_start
    #b_node_num = p_num*p_num 
    p_b_edges = get_p_b_edges(p_num, b_node_start)
    edge_list.extend(p_b_edges)

    edges = torch.tensor(np.array(edge_list)).t().to(device)

    return edges
    

def load_features(r_embs, p_embs, b_embs, dk):
    #P-idx    0      ~ |P|-1
    #R-idx   |P|     ~ |P|+|R|-1
    #B-idx   |P|+|R| ~ |P|+|R|+|B|-1
    #pdb.set_trace()
    node_features = torch.zeros(len(r_embs)+len(p_embs)+len(b_embs), dk).to(p_embs.device)
    p_num = p_embs.size()[0]
    r_num = r_embs.size()[0]
    b_num = b_embs.size()[0]
    node_features[:p_num] = p_embs
    node_features[p_num:p_num+r_num] = r_embs
    node_features[p_num+r_num:]
    node_features.to(p_embs.device)
    return node_features


def create_graph(r_list, r_embs, p_embs, b_embs):
    device = p_embs.device
    edge_index = create_edges(r_list, device)
    node_features = load_features(r_embs, p_embs, b_embs, dk=p_embs.size()[-1])
    data = Data(x=node_features, edge_index=edge_index)
    return data

if __name__ == "__main__":
    r_nodes = [[1,2,3,4]]
    r_embs = torch.zeros(4, 128)
    p_embs = torch.zeros(1, 128)
    b_embs = torch.zeros(3, 128)
    g = create_graph(r_nodes, r_embs, p_embs, b_embs)
    print(g)
    model = GCN(hidden_channels=64, dim_node_features=128, num_classes=10)
    print(model)