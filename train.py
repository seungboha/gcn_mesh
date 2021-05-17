from __future__ import division
from __future__ import print_function

import os
import glob
import random
import time
import argparse
import numpy as np
from tqdm import tqdm 
import datetime

import torch
import torch.nn.functional as F
import torch.optim as optim

from utils import load_data, accuracy, Logger
from models import GCN

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--data_path', type=str, default="/data/human/Mindboggle/DL/")
parser.add_argument('--data_name', type=str, default='cora')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Split data into tv(train&val) / test set
subj_path = os.path.join(args.data_path, "mesh")
all_subjects = [subject for subject in os.listdir(subj_path) if os.path.isdir(os.path.join(subj_path, subject))]
n_subjects = len(all_subjects)
print(n_subjects)
subj_test = random.sample(all_subjects, int(n_subjects*0.2))
subj_tv = list(set(all_subjects) - set(subj_test))



def train(args, model, subject, epoch):
    adj, features, labels = load_data(args.data_path, subject)
    t = time.time()
    model.train()
    if args.cuda:
        model.cuda()
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()

    #print(adj.shape)
    #print(features.shape)
    #print(labels.shape)
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output, labels)
    acc_train = accuracy(output, labels)
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)

    #loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    #acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          #'loss_val: {:.4f}'.format(loss_val.item()),
          #'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))


def test(args, model, subject):
    model.eval()
    adj, features, labels = load_data(args.data_path, subject)
    output = model(features, adj)
    loss_test = F.nll_loss(output, labels)
    acc_test = accuracy(output, labels)
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))

# Model and optimizer
model = GCN(nfeat=32,
            nhid=args.hidden,
            nclass=32,
            dropout=args.dropout)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

now = datetime.datetime.now()
formatedDate = now.strftime('%Y%m%d_%H_%M_')
result_logger = Logger(os.path.join('./', formatedDate +'result.log'))

t_toal = time.time()
for epoch in range(args.epochs):
    t_epoch = time.time()
    # Split train data into trian/val
    subj_train = random.sample(subj_tv, int(n_subjects*0.7))
    subj_val = list(set(subj_tv) - set(subj_train))
    
    for subject in subj_train:
        print(subject)
        train(args, model=model, subject=subject, epoch=epoch)
    print("Time elapsed per epoch: {:.4f}s".format(time.time() - t_epoch))

print("Training finished")
print("Time elapsed in total: {:.4f}s".format(time.time() - t_total))


# Testing
#test()
