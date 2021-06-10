import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution


'''
class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)

'''


class GCN_4(nn.Module):
    def __init__(self, nfeat, nclass):
        super(GCN_4, self).__init__()

        self.gc1 = GraphConvolution(nfeat, 256)
        self.gc2 = GraphConvolution(256, 128)
        self.gc3 = GraphConvolution(128, 64)
        self.gc4 = GraphConvolution(64, nclass)

        #self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.relu(self.gc2(x, adj))
        x = F.relu(self.gc3(x, adj))
        x = F.relu(self.gc4(x, adj))

        return F.log_softmax(x, dim=1)



class GCN_9(nn.Module):
    def __init__(self, nfeat, nclass):
        super(GCN_9, self).__init__()

        self.gc1 = GraphConvolution(nfeat, 256)
        self.gc2 = GraphConvolution(256, 256)
        self.gc3 = GraphConvolution(256, 256)
        self.gc4 = GraphConvolution(256, 256)
        self.gc5 = GraphConvolution(256, 256)
        self.gc6 = GraphConvolution(256, 128)
        self.gc7 = GraphConvolution(128, 128)
        self.gc8 = GraphConvolution(128, 64)
        self.gc9 = GraphConvolution(64, nclass)
        #self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        #x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc2(x, adj))
        #x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc3(x, adj))
        #x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc4(x, adj))
        x = F.relu(self.gc5(x, adj))
        x = F.relu(self.gc6(x, adj))
        x = F.relu(self.gc7(x, adj))
        x = F.relu(self.gc8(x, adj))

        return F.log_softmax(x, dim=1)



class GCN_w(nn.Module):
    def __init__(self, nfeat, nclass, ):
        super(GCN_w, self).__init__()

        self.gc1 = GraphConvolution(nfeat, 512)
        self.gc2 = GraphConvolution(512, 512)
        self.gc3 = GraphConvolution(512, 512)
        self.gc4 = GraphConvolution(512, 512)
        self.gc5 = GraphConvolution(512, 256)
        self.gc6 = GraphConvolution(256, 128)
        self.gc7 = GraphConvolution(128, 128)
        self.gc8 = GraphConvolution(128, 64)
        self.gc9 = GraphConvolution(64, nclass)
        #self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.relu(self.gc2(x, adj))
        x = F.relu(self.gc3(x, adj))
        x = F.relu(self.gc4(x, adj))
        x = F.relu(self.gc5(x, adj))
        x = F.relu(self.gc6(x, adj))
        x = F.relu(self.gc7(x, adj))
        x = F.relu(self.gc8(x, adj))

        return F.log_softmax(x, dim=1)

class GCN_ww(nn.Module):
    def __init__(self, nfeat, nclass):
        super(GCN_ww, self).__init__()

        self.gc1 = GraphConvolution(nfeat, 1024)
        self.gc2 = GraphConvolution(1024, 1024)
        self.gc3 = GraphConvolution(1024, 1024)
        self.gc4 = GraphConvolution(1024, 512)
        self.gc5 = GraphConvolution(512, 256)
        self.gc6 = GraphConvolution(256, 128)
        self.gc7 = GraphConvolution(128, 128)
        self.gc8 = GraphConvolution(128, 64)
        self.gc9 = GraphConvolution(64, nclass)
        #self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.relu(self.gc2(x, adj))
        x = F.relu(self.gc3(x, adj))
        x = F.relu(self.gc4(x, adj))
        x = F.relu(self.gc5(x, adj))
        x = F.relu(self.gc6(x, adj))
        x = F.relu(self.gc7(x, adj))
        x = F.relu(self.gc8(x, adj))
        x = F.relu(self.gc9(x, adj))

        return F.log_softmax(x, dim=1)

class GCN_wwBN(nn.Module):
    def __init__(self, nfeat, nclass):
        super(GCN_wwBN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, 1024)
        self.bn1 = nn.BatchNorm1d(1024)
        self.gc2 = GraphConvolution(1024, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.gc3 = GraphConvolution(1024, 1024)
        self.bn3 = nn.BatchNorm1d(1024)
        self.gc4 = GraphConvolution(1024, 512)
        self.bn4 = nn.BatchNorm1d(512)
        self.gc5 = GraphConvolution(512, 256)
        self.bn5 = nn.BatchNorm1d(256)
        self.gc6 = GraphConvolution(256, 128)
        self.bn6 = nn.BatchNorm1d(128)
        self.gc7 = GraphConvolution(128, 128)
        self.bn7 = nn.BatchNorm1d(128)
        self.gc8 = GraphConvolution(128, 64)
        self.bn8 = nn.BatchNorm1d(64)
        self.gc9 = GraphConvolution(64, nclass)
        #self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.bn1(self.gc1(x, adj)))
        x = F.relu(self.bn2(self.gc2(x, adj)))
        x = F.relu(self.bn3(self.gc3(x, adj)))
        x = F.relu(self.bn4(self.gc4(x, adj)))
        x = F.relu(self.bn5(self.gc5(x, adj)))
        x = F.relu(self.bn6(self.gc6(x, adj)))
        x = F.relu(self.bn7(self.gc7(x, adj)))
        x = F.relu(self.bn8(self.gc8(x, adj)))
        x = F.relu(self.gc9(x, adj))

        return F.log_softmax(x, dim=1)
"""
CUDA out of memory
# params : 4511200
class GCN_res(nn.Module):
    def __init__(self, nfeat, nclass, dropout):
        super(GCN_res, self).__init__()

        self.gc1 = GraphConvolution(nfeat, 1024)
        self.gc2 = GraphConvolution(1024, 1024)
        self.gc3 = GraphConvolution(1024, 1024)
        self.gc4 = GraphConvolution(1024, 1024)
        self.gc5 = GraphConvolution(1024, 1024)
        self.gc6 = GraphConvolution(1024, 256)
        self.gc7 = GraphConvolution(256, 128)
        self.gc8 = GraphConvolution(128, 64)
        self.gc9 = GraphConvolution(64, nclass)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        identity1 = x.clone()

        x = F.relu(self.gc2(x, adj))
        identity2 = x.clone()
        x += identity1

        x = F.relu(self.gc3(x, adj))
        identity3 = x.clone()
        x += identity2

        x = F.relu(self.gc4(x, adj))
        identity4 = x.clone()
        x += identity3

        x = F.relu(self.gc5(x, adj))
        identity5 = x.clone()
        x += identity4

        x = F.relu(self.gc6(x, adj))
        x = F.relu(self.gc7(x, adj))
        x = F.relu(self.gc8(x, adj))
        x = F.relu(self.gc9(x, adj))

        return F.log_softmax(x, dim=1)
"""

class GCN_res(nn.Module):
    def __init__(self, nfeat, nclass):
        super(GCN_res, self).__init__()

        self.gc1 = GraphConvolution(nfeat, 1024)
        self.gc2 = GraphConvolution(1024, 1024)
        self.gc3 = GraphConvolution(1024, 1024)
        self.gc4 = GraphConvolution(1024, 64)
        self.gc5 = GraphConvolution(64, nclass)
        #self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        identity1 = x.clone()

        x = F.relu(self.gc2(x, adj))
        identity2 = x.clone()
        x += identity1

        x = F.relu(self.gc3(x, adj))
        x += identity2

        x = F.relu(self.gc4(x, adj))
        x = F.relu(self.gc5(x, adj))

        return F.log_softmax(x, dim=1)


class GCN_res2(nn.Module):
    def __init__(self, nfeat, nclass):
        super(GCN_res2, self).__init__()

        self.gc1 = GraphConvolution(nfeat, 1024)
        self.gc2 = GraphConvolution(1024, 1024)
        self.gc3 = GraphConvolution(1024, 1024)
        self.gc4 = GraphConvolution(1024, 64)
        self.gc5 = GraphConvolution(64, nclass)
        #self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        identity1 = x.clone()

        x = F.relu(self.gc2(x, adj))
        identity2 = x.clone()
        x += identity1

        x = F.relu(self.gc3(x, adj))
        x += identity1
        x += identity2

        x = F.relu(self.gc4(x, adj))
        x = F.relu(self.gc5(x, adj))

        return F.log_softmax(x, dim=1)
