import os
import glob
import random 

import numpy as np
import scipy.sparse as sp
import torch


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def load_data(data_path, subject):
    feat_name = ['curv', 'iH', 'sulc']
    mesh_path = os.path.join(data_path, "mesh")
    subj_path = os.path.join(mesh_path, subject)
    [v, f] = read_vtk(os.path.join(subj_path, 'lh.white.vtk'))    # Index for vertices starts from 0
    
    
    features = []
    for name in feat_name:
        for file in glob.glob(os.path.join(subj_path, "lh.{}.txt".format(name))):
            feat = np.genfromtxt(file, dtype=np.float32)
            features.append(feat)
    features = np.array(features)
    features = np.transpose(features)
    features = sp.csr_matrix(features)
    
    label_path = os.path.join(data_path, "label")
    subj_label_path = os.path.join(label_path, subject)
    labels_orig = np.genfromtxt(os.path.join(subj_label_path, "lh.label.txt"))
    labels = encode_onehot(labels_orig)
    
    '''
    features = labels
    features = sp.csr_matrix(features)
    print(features.shape)
    확인 결과 epoch 2만에 99퍼 나옴
    '''
    # assert labels.shape[0]==features.shape[0]

    # make the list of edges
    unordered_edges = np.concatenate(
        [np.stack((f[:, 0], f[:, 1]), axis=1),
            np.stack((f[:, 1], f[:, 2]), axis=1),
            np.stack((f[:, 2], f[:, 0]), axis=1)],
        axis=0
        )

    # 다시 만들어야됨
    adj = sp.coo_matrix(
        (np.ones(unordered_edges.shape[0]), (unordered_edges[:, 0], unordered_edges[:, 1])),    # row, col
        shape=(labels.shape[0], labels.shape[0]),
        dtype=np.float32
                )

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    #features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    return adj, features, labels


def normalize(mx):
    """"Col-normalize sparse matrix"""
    colsum = np.array(mx.sum(0))
    c_inv = np.power(colsum, -1).flatten()
    c_inv[np.isinf(c_inv)] = 0.
    c_mat_inv = sp.diags(c_inv)
    mx = mx.dot(c_mat_inv)
    return mx

'''
def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx
'''

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def read_vtk(filename):
    fid = open(filename, 'r')
    lines = fid.readlines()
    fid.close()

    v = []
    f = []
    b = ([lines.index(i) for i in lines if i.startswith("POINTS")])[0] + 1
    nVert = int(lines[b - 1].split()[1])
    for i in range(b, b + nVert):
        line = lines[i]
        row = [float(n) for n in line.split()] 
        v.append(row)

    b = ([lines.index(i) for i in lines if i.startswith("POLYGONS")])[0] + 1 
    nFaces = int(lines[b - 1].split()[1])
    for i in range(b, b + nFaces):
        line = lines[i]
        row = [int(n) for n in line.split()]
        row = row[1:]
        f.append(row)

    v = np.array(v)
    f = np.array(f)
    return v, f

class Logger(object):
    def __init__(self, path, int_form=':04d', float_form=':.6f'):
        self.path = path
        self.int_form = int_form
        self.float_form = float_form
        self.width = 0

    def __len__(self):
        try: return len(self.read())
        except: return 0

    def write(self, values):
        if not isinstance(values, Iterable):
            values = [values]
        if self.width == 0:
            self.width = len(values)
        assert self.width == len(values), 'Inconsistent number of items.'
        line = ''
        for v in values:
            if isinstance(v, int):
                line += '{{{}}} '.format(self.int_form).format(v)
            elif isinstance(v, float):
                line += '{{{}}} '.format(self.float_form).format(v)
            elif isinstance(v, str):
                line += '{} '.format(v)
            else:
                raise Exception('Not supported type.')
        with open(self.path, 'a') as f:
            f.write(line[:-1] + '\n')