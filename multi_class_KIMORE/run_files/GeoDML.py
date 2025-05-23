import torch
import math
import numpy as np
import argparse
import os
import sys
import torch.nn as nn
import torch.nn.functional as F
import torchgeometry.core as tgmc
import copy
import numpy as np
from torch.optim.lr_scheduler import StepLR, ExponentialLR, ReduceLROnPlateau, CosineAnnealingLR
np.random.seed(42)
from fvcore.nn import FlopCountAnalysis, parameter_count
from thop import profile
import pickle
from sklearn.model_selection import train_test_split
from collections import Counter

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
from processing.CenteredScaled import CenteredScaled, ref_rot
from processing.inv_exp import inv_exp
from models.RigidNet_GeoDML import RigidNet
from models.NonRigidNet_GeoDML import NonRigidNet


#Parameters
parser = argparse.ArgumentParser()
parser.add_argument('--cast', default='log_0refseq', type=str,help = 'first frame of sequence (log_0refseq)')
parser.add_argument('--num_runs', default=10,type=int)
parser.add_argument('--learning_rate', default=0.001, type=float)
parser.add_argument('--num_epoch', default=40, type=int)
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--batch_size_test', default = 16, type=int)
parser.add_argument('--save_max', default=True, type=bool, help='if true, will only save the model with the best accuracy, else it will save every model')
parser.add_argument('--use_cuda', default=1, type=int)
opt = parser.parse_args()

use_cuda = opt.use_cuda
cuda = torch.cuda.is_available()
device = 'cuda' if (cuda == True and use_cuda == 1) else 'cpu'
if device == 'cuda':
    print('Using CUDA')
    torch.cuda.empty_cache()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
else:
    print('NOT using CUDA')

options=['RigidTransform','NonRigidTransform','RigidTransformInit','NonRigidTransformInit']

training_epochs= opt.num_epoch
batch_size= opt.batch_size
batch_size_test = opt.batch_size_test
learning_rate=opt.learning_rate
runs = opt.num_runs
save_max = opt.save_max
cast = opt.cast

data = np.load(r'C:\KIMORE_Dataset\data_and_features\action_classification_data\xyz\train_data.npy', allow_pickle=True)
labels = np.load(r'C:\KIMORE_Dataset\data_and_features\action_classification_data\xyz\train_label.pkl', allow_pickle=True)

# print(Counter(list(labels[1])))

data_2 = np.load(r'C:\KIMORE_Dataset\data_and_features\action_classification_data\ang\train_data.npy', allow_pickle=True)
labels_2 = np.load(r'C:\KIMORE_Dataset\data_and_features\action_classification_data\ang\train_label.pkl', allow_pickle=True)

labels = np.array(labels[1])
labels_2 = np.array(labels_2[1])

# *** Using only the pos features ***
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.15, random_state=42, stratify=labels)
test_data, val_data, test_labels, val_labels = train_test_split(test_data, test_labels, test_size=0.2, random_state=42, stratify=test_labels)

# train_data_2, test_data_2, train_labels_2, test_labels_2 = train_test_split(data_2, labels_2, test_size=0.2, random_state=42, stratify=labels_2)
# X_train = np.concatenate((train_data, train_data_2), axis=0)
# y_train = np.concatenate((train_labels, train_labels_2), axis=0)
# X_test = np.concatenate((test_data, test_data_2), axis=0)
# y_test = np.concatenate((test_labels, test_labels_2), axis=0)

X_train = train_data
y_train = train_labels
X_test = test_data
y_test = test_labels
X_val = val_data
y_val = val_labels

X_train = X_train.transpose(0, 2, 3, 1, 4).squeeze(-1)
X_test = X_test.transpose(0, 2, 3, 1, 4).squeeze(-1)
X_val = X_val.transpose(0, 2, 3, 1, 4).squeeze(-1)

print("--------------------------------------")
print('Train data size', X_train.shape)
print('Test data size', X_test.shape)
print('Val data size', X_val.shape)
print(y_train.shape)
print(y_test.shape)
print(y_val.shape)

# print(Counter(list(y_train)))
# print(Counter(list(y_test)))
# print(Counter(list(y_val)))

print("--------------------------------------")

y_train = np.array(y_train).astype('int32')
y_test = np.array(y_test).astype('int32')
y_val = np.array(y_val).astype('int32')

ref_skel = copy.deepcopy(X_train[0,0])
print('Preprocessing (going to preshape space)')

for i in range(X_train.shape[0]):
    for j in range(X_train[i].shape[0]):
        X_train[i, j] = CenteredScaled(X_train[i, j])

for i in range(X_test.shape[0]):
    for j in range(X_test[i].shape[0]):
        X_test[i, j] = CenteredScaled(X_test[i, j])

for i in range(X_val.shape[0]):
    for j in range(X_val[i].shape[0]):
        X_val[i, j] = CenteredScaled(X_val[i, j])

if cast == 'log_sref':
    for i in range(X_train.shape[0]):
        for j in range(X_train[i].shape[0]):
            try:
                X_train[i, j] = inv_exp(ref_skel, X_train[i, j])
            except:
                i = i + 1

    for i in range(X_test.shape[0]):
        for j in range(X_test[i].shape[0]):
            try:
                X_test[i, j] = inv_exp(ref_skel, X_test[i, j])
            except:
                i = i + 1    

    for i in range(X_val.shape[0]):
        for j in range(X_val[i].shape[0]):
            try:
                X_val[i, j] = inv_exp(ref_skel, X_val[i, j])
            except:
                i = i + 1    

elif cast == 'log_0refseq':
    for i in range(X_train.shape[0]):
        for j in range(X_train[i].shape[0]):
            try:
                X_train[i, j] = inv_exp(X_train[i, 0], X_train[i, j])
            except:
                i = i + 1

    for i in range(X_test.shape[0]):
        for j in range(X_test[i].shape[0]):
            try:
                X_test[i, j] = inv_exp(X_test[i, 0], X_test[i, j])
            except:
                i = i + 1

    for i in range(X_val.shape[0]):
        for j in range(X_val[i].shape[0]):
            try:
                X_val[i, j] = inv_exp(X_val[i, 0], X_val[i, j])
            except:
                i = i + 1   
else:
    print('Casting variant doesn\'t exist')
    quit()

num_frames = X_train.shape[1]
num_joints = X_train.shape[2]
dims = X_train.shape[3]
num_channels = num_joints * dims

print('Preshape space done !!!')
    
acc_ = []
for m in options:
    mod = m
    run_acc = []
    print('Running {}'.format(mod))
    for r in range(runs):
        print('run {} for {}'.format(r,mod))
        dml_run_acc = []
        for dml_run in range(runs+5): 

            if m == 'RigidTransform' or m == 'RigidTransformInit':
                rigid = True
            else:
                rigid = False
            
            if rigid == True:
                model = RigidNet(mod=mod, run=r, dml_run=dml_run).to(device)
            else:
                model = NonRigidNet(mod=mod, run=r, dml_run=dml_run).to(device)

            criterion = nn.CrossEntropyLoss()
            opt = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-3)
            scheduler = ReduceLROnPlateau(opt, 
                                mode='max',  
                                factor=0.1, 
                                patience=15,  
                                min_lr=1e-6)

            steps = len(X_train) // batch_size
            # print('model will be training for {} epochs with {} steps per epoch'.format(training_epochs,steps))

            for epoch in range(training_epochs):
                model.train()
                correct=0
                total=0
                epoch_loss = 0.0
                for i in range(steps + (1 if len(X_train) % batch_size != 0 else 0)):                       
                    start_idx = i * batch_size
                    end_idx = start_idx + batch_size
                    x, y = X_train[start_idx:end_idx], y_train[start_idx:end_idx]
                    # If the batch size exceeds the remaining samples, adjust the slicing
                    if end_idx > len(X_train):
                        x = X_train[start_idx:]
                        y = y_train[start_idx:]
                    inputs, labels = torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)
                    opt.zero_grad()   
                    output = model(inputs.float())
                    loss = criterion(output, labels.long())
                    loss.backward()
                    opt.step()
                    epoch_loss += loss.item()
                    y_pred_softmax = torch.log_softmax(output.data, dim = 1)
                    _, predicted = torch.max(y_pred_softmax, 1)
                    total += labels.size(0)
                    correct += (predicted == labels.long()).sum().item()

                accuracy = 100 * correct / total
                epoch_loss = epoch_loss / len(X_train)
                # print("Training Accuracy = {} : Training Loss {}".format(accuracy,epoch_loss)) 

                # ********** validation here **********
                steps_ = len(X_val) // batch_size
                model.eval() 
                val_loss = 0.0
                total_ = 0
                correct_ = 0
                with torch.no_grad():
                    for i in range(steps_ + (1 if len(X_val) % batch_size != 0 else 0)):
                        start_idx = i * batch_size
                        end_idx = start_idx + batch_size
                        x, y = X_val[start_idx:end_idx], y_val[start_idx:end_idx]

                        # If the batch size exceeds the remaining samples, adjust the slicing
                        if end_idx > len(X_val):
                            x = X_val[start_idx:]
                            y = y_val[start_idx:]

                        inputs_, labels_ = torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)
                        output_ = model(inputs_.float())
                        loss_ = criterion(output_, labels_.long())
                        val_loss += loss_.item()

                        y_pred_softmax_ = torch.log_softmax(output_.data, dim = 1)
                        _, predicted_ = torch.max(y_pred_softmax_, 1)
                        total_ += labels_.size(0)
                        correct_ += (predicted_ == labels_.long()).sum().item()

                accuracy_val = 100 * correct_ / total_
                val_loss = val_loss / len(X_val)
                # print("Validation Accuracy {} Validation loss {}".format(accuracy_val, val_loss))

                # current_lr = scheduler.optimizer.param_groups[0]['lr']
                # print('Current learning rate: {}'.format(current_lr))

                scheduler.step(accuracy_val)
                        
            correct_test = 0
            total_test = 0
            model.eval()
            with torch.no_grad():
                steps = len(X_test) // batch_size_test
                for i in range(steps + (1 if len(X_test) % batch_size_test != 0 else 0)):                 
                    start_idx = i * batch_size_test
                    end_idx = start_idx + batch_size_test
                    x, y = X_test[start_idx:end_idx], y_test[start_idx:end_idx]

                    # If the batch size exceeds the remaining samples, adjust the slicing
                    if end_idx > len(X_test):
                        x = X_test[start_idx:]
                        y = y_test[start_idx:]                
                    inputs, labels = torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)
                    outputs = model(inputs.float())
                    y_pred_softmax = torch.log_softmax(outputs.data, dim = 1)
                    _, predicted = torch.max(y_pred_softmax, 1)
                    total_test += labels.size(0)
                    correct_test += (predicted == labels.long()).sum().item()

            accuracy = 100*correct_test/total_test
            dml_run_acc.append(accuracy)
            
        run_acc.append(max(dml_run_acc))

    acc_.append(max(run_acc))
    
    print('Test accuracy for {} is {}'.format(mod, max(run_acc)))

print('--------------------------------')
print('Accuracies for 4 transfom layers (Rigid, NonRigid, RigidInit and NonRigidInit)'.format(), acc_)
print('--------------------------------')