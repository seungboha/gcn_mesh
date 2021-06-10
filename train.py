from __future__ import division
from __future__ import print_function

import os
import glob
import random
import time
import pickle
import argparse
import numpy as np
from tqdm import tqdm 
import datetime
import copy

import torch
import torch.nn.functional as F
import torch.optim as optim


from utils import load_data, accuracy, Logger
from models import *

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='gcn_4',
                    help='Model to train')
parser.add_argument('--gpu_num', type=int, default=0,
                    help='GPU number to use')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--nfeat', type=int, default=6,
                    help='Number of infeatures.')
parser.add_argument('--nclass', type=int, default=32,
                    help='Number of classes.')
#parser.add_argument('--dropout', type=float, default=0.5,
#                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--data_path', type=str, default="/data/human/Mindboggle/DL/")
#parser.add_argument('--fastmode', action='store_true', default=False,
#                    help='Validate during training pass.')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device(f'cuda:{args.gpu_num}' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device) 

np.random.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Split data into tv(train&val) / test set
subj_path = os.path.join(args.data_path, "mesh")
all_subjects = [subject for subject in os.listdir(subj_path) if os.path.isdir(os.path.join(subj_path, subject))]

# read all meshes
print("Loading data...")
'''
아무것도 없을때 mesh 데이터 txt로 저장
meshes = dict()
for subject in tqdm(all_subjects):
    adj, features, labels = load_data(args.data_path, subject)
    meshes[subject] =  {'adj':adj, 'features':features, 'labels':labels}
with open('meshes.txt', 'wb') as f:
    pickle.dump(meshes, f)
'''
with open('meshes.txt', 'rb') as f:
    meshes = pickle.load(f)

n_subjects = len(all_subjects)
subj_test = random.sample(all_subjects, int(n_subjects*0.2))
subj_tv = sorted(list(set(all_subjects) - set(subj_test)))    # to reproduce

# Split train data into trian/val
subj_train = random.sample(subj_tv, int(n_subjects*0.7))
subj_val = list(set(subj_tv) - set(subj_train))

def train(args, model, meshes, subjects, epoch, logger):
    
    model.train()
    
    for subject in subjects:
        t = time.time()
        adj = meshes[subject]['adj']
        features = meshes[subject]['features']
        labels = meshes[subject]['labels']

        if args.cuda:
            model.cuda()
            features = features.cuda()
            adj = adj.cuda()
            labels = labels.cuda()

        optimizer.zero_grad()
        output = model(features, adj)
        loss_train = F.nll_loss(output, labels)
        acc_train = accuracy(output, labels)
        loss_train.backward()
        optimizer.step()

        #if not args.fastmode:
            # Evaluate validation set performance separately,
            # deactivates dropout during validation run.
        #    model.eval()
        #    output = model(features, adj)
    
        logger.write([epoch+1, '{:20}'.format(subject), 
                        loss_train.item(), acc_train.item()])

        print('Epoch: {:04d}'.format(epoch+1),
            'loss_train: {:.4f}'.format(loss_train.item()),
            'acc_train: {:.4f}'.format(acc_train.item()),
            'time: {:.4f}s'.format(time.time() - t))

def validate(args, model, meshes, subjects, epoch, logger, save_path):
    model.eval()

    best_result = {'subject': 'mmm', 'acc': 0, 'output': []}
    worst_result = {'subject': 'mmm', 'acc': 100, 'output': []}
    best_acc = 0
    worst_acc = 100

    best_loss = 10000000

    for subject in subjects:
        adj = meshes[subject]['adj']
        features = meshes[subject]['features']
        labels = meshes[subject]['labels']

        if args.cuda:
            model.cuda()
            features = features.cuda()
            adj = adj.cuda()
            labels = labels.cuda()

        output = model(features, adj)
        loss_val = F.nll_loss(output, labels)
        acc_val = accuracy(output, labels)
        
        if acc_val > best_acc:
            best_acc = acc_val
            best_result['subject'] = subject
            best_result['acc'] = acc_val
            best_result['output'] = output.data.cpu().tolist()
            
            # save the best model
            best_model = copy.deepcopy(model)

        if acc_val < worst_acc:
            worst_acc = acc_val
            worst_result['subject'] = subject
            worst_result['acc'] = acc_val
            worst_result['output'] = output.data.cpu().tolist()
        
        #if loss_val < best_loss:
        #    best_model = copy.deepcopy(model)
        
        logger.write([epoch+1, '{:20}'.format(subject), 
                        loss_val.item(), acc_val.item()])
        print("Val set results:",
            "loss= {:.4f}".format(loss_val.item()),
            "accuracy= {:.4f}".format(acc_val.item()))

    save_best = os.path.join(save_path, '{}_best.txt'.format(epoch+1))
    save_worst = os.path.join(save_path, '{}_worst.txt'.format(epoch+1))
    save_path_model = os.path.join(save_path, 'best_model.pth')

    with open(save_best, 'wb') as f:
        pickle.dump(best_result, f)
    with open(save_worst, 'wb') as f:
        pickle.dump(worst_result, f)           

    torch.save(best_model.state_dict(), save_path_model)



def test(args, model, subjects, logger, save_path):
    model.eval()
    output_dict = dict()
    for subject in subjects:
        adj = meshes[subject]['adj']
        features = meshes[subject]['features']
        labels = meshes[subject]['labels']
        
        if args.cuda:
            model.cuda()
            features = features.cuda()
            adj = adj.cuda()
            labels = labels.cuda()

        output = model(features, adj)
        loss_test = F.nll_loss(output, labels)
        acc_test = accuracy(output, labels)
        
        # save prediction
        output_dict[subject] = output.data.cpu().tolist()
        
        logger.write(['{:20}'.format(subject), loss_test.item(), acc_test.item()])
        print("Test set results:",
            "loss= {:.4f}".format(loss_test.item()),
            "accuracy= {:.4f}".format(acc_test.item()))

    print("Test finished...")
    
    output_path = os.path.join(save_path, 'output.txt')
    with open(output_path, 'wb') as f:
        pickle.dump(output_dict, f)



# Model and optimizer
if args.model == 'gcn_4': 
    model = GCN_4(nfeat=args.nfeat,
                    nclass=args.nclass)
elif args.model == 'gcn_9':
    model = GCN_9(nfeat=args.nfeat,
                    nclass=args.nclass)
elif args.model == 'gcn_w':
    model = GCN_w(nfeat=args.nfeat,
                    nclass=args.nclass)
elif args.model == 'gcn_ww':
    model = GCN_ww(nfeat=args.nfeat,
                    nclass=args.nclass)
elif args.model == 'gcn_res':
    model = GCN_res(nfeat=args.nfeat,
                    nclass=args.nclass)
elif args.model == 'gcn_res2':
    model = GCN_res(nfeat=args.nfeat,
                    nclass=args.nclass)
elif args.model == 'gcn_wwbn':
    model = GCN_wwBN(nfeat=args.nfeat,
                    nclass=args.nclass)

n_params = sum(p.numel() for p in model.parameters())
print("# of parameters: ", n_params)

optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

now = datetime.datetime.now()
formatedDate = now.strftime('%Y%m%d_%H_%M_')
result_dir = os.path.join('./', formatedDate + args.model)

if not os.path.exists(result_dir):
    os.mkdir(result_dir)

train_logger = Logger(os.path.join(result_dir, 'train.log'))
val_logger = Logger(os.path.join(result_dir, 'val.log'))
test_logger = Logger(os.path.join(result_dir, 'test.log'))

# if not exist 구현하자
arguments = args.__dict__
arguments['n_params'] = n_params
    
train_logger.write(arguments)
#val_logger.write(arguments)
#test_logger.write(arguments)


train_logger.write(['{:10}'.format('Epoch'),
                    '{:20}'.format('Subject'),
                    '{:15}'.format('Loss_train'),
                    '{:15}'.format('Acc_train')])
val_logger.write(['{:10}'.format('Epoch'),
                    '{:20}'.format('Subject'),
                    '{:15}'.format('Loss_val'),
                    '{:15}'.format('Acc_val')])
test_logger.write(['{:20}'.format('Subject'),
                    '{:15}'.format('Loss_test'),
                    '{:15}'.format('Acc_test')])

t_total = time.time()

for epoch in range(args.epochs):
    t_epoch = time.time()
    random.shuffle(subj_train)
    random.shuffle(subj_val)
    train(args, model=model, meshes=meshes, subjects=subj_train, 
            epoch=epoch, logger=train_logger)

    validate(args, model=model, meshes=meshes, subjects=subj_val, 
            epoch=epoch, logger=val_logger, save_path=result_dir)

    print("Time elapsed per epoch: {:.4f}s".format(time.time() - t_epoch))

print("Training finished")
print("Time elapsed in total: {:.4f}s".format(time.time() - t_total))


# Testing
model.load_state_dict(torch.load(os.path.join(result_dir, 'best_model.pth')))
test(args, model, subj_test, test_logger, save_path=result_dir)
