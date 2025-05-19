import torch
import math
import numpy as np
import argparse
import os
import torch.nn as nn
import torch.nn.functional as F
import torchgeometry.core as tgmc
import copy
from processing.CenteredScaled import CenteredScaled
from processing.inv_exp import inv_exp
from processing.PT import *
from models.RigidNet120 import RigidNet120
from models.NonRigidNet120 import NonRigidNet120



#Parameters
parser = argparse.ArgumentParser()
parser.add_argument('--prot', default='xsub', type=str,help = 'xsub or xsetup')
parser.add_argument('--cast', default='log_sref', type=str,help = 'LogMap sameref (log_sref) or first frame of sequence (log_0refseq) or parallel transport (pt)')
parser.add_argument('--num_runs', default=10,type=int)
parser.add_argument('--learning_rate', default=0.0001, type=float)
parser.add_argument('--num_epoch', default=50, type=int)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--batch_size_test', default = 256, type=int)
parser.add_argument('--save_max', default=True, type=bool, help='if true, will only save the model with the best accuracy, else it will save every model')
parser.add_argument('--use_cuda', default=1, type=int)
parser.add_argument('--dir',default = '', type=str, help= 'The directory must contain the intended val_label.pkl and train_label.pkl')
parser.add_argument('--file_train', default='', type=str, help='This removes the use of the prot argument, both arguments must be specified')
parser.add_argument('--file_test', default='', type=str, help='This removes the use of the prot argument, both arguments must be specified')
opt = parser.parse_args()


prot = opt.prot
if prot not in ['xsub','xsetup']:
    print('Invalid protocol for NTU60 will be using the cross_subject protocol')
    prot = 'xsub'

dir = './data/nturgb_d120/{}/'.format(prot)
file_train = 'train_{}_interp100.npy'.format(prot)
file_test = 'test_{}_interp100.npy'.format(prot)


if opt.file_train != '' and opt.file_test != '':
    file_train = opt.file_train
    file_test = opt.file_test
else:
    print('Will be using the default generated files files')
    
use_cuda = opt.use_cuda
#argument block#

cuda = torch.cuda.is_available()
device = 'cuda' if (cuda == True and use_cuda == 1) else 'cpu'
if device == 'cuda':
    print('Using CUDA')
    torch.cuda.empty_cache()
else:
    print('NOT using CUDA')

rigid = True #will be changed based on model name
options=['RigidTransform','NonRigidTransform','RigidTransformInit','NonRigidTransformInit']


training_epochs= opt.num_epoch
batch_size= opt.batch_size
batch_size_test = opt.batch_size_test
learning_rate=opt.learning_rate
runs = opt.num_runs
save_max = opt.save_max
cast = opt.cast

#Loading data
print('loading data')
X_train = np.load(os.path.join(dir,file_train),  allow_pickle=True)
X_test = np.load(os.path.join(dir,file_test), allow_pickle=True)
y_train = np.load(os.path.join(dir,'train_label.pkl'), allow_pickle=True)
y_test = np.load(os.path.join(dir,'val_label.pkl'), allow_pickle=True)

print("--------------------------------------")
print (X_train.shape)
one_body = False
if ('body1' in file_train) or ('body2' in file_train):
    one_body = True


if one_body == False:
    print('reshaping (Destacking Joints and dims)')
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2]//3 , 3))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], X_test.shape[2]//3 , 3))

# get only labels no filenames
y_train = y_train[1]
y_test = y_test[1]

y_train = np.array(y_train).astype('int32')
y_test = np.array(y_test).astype('int32')

num_labels = 120
print(y_train.shape)

ref_skel = copy.deepcopy(X_train[0,0])

print('Preprocessing (going to preshape space)')
for i in range(X_train.shape[0]):
    for j in range(X_train[i].shape[0]):
        X_train[i, j] = CenteredScaled(X_train[i, j])

for i in range(X_test.shape[0]):
    for j in range(X_test[i].shape[0]):
        X_test[i, j] = CenteredScaled(X_test[i, j])


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

elif cast == 'PT':
    for i in range(X_train.shape[0]):
        for j in range(X_train[i].shape[0]):
            try:
                X_train[i, j] = inv_exp(ref_skel, X_train[i, j])
            except:
                i = i + 1
    for i in range(X_train.shape[0]):
        for j in range(X_train[i].shape[0]):
            V0= X_train[i, j]
            X0= X_train[i, 0] #source
            Y0= X_train[0, 0] #target
            X_train[i, j] = new_PTG2(X0, V0, Y0)      
  
    for i in range(X_test.shape[0]):
        for j in range(X_test[i].shape[0]):
            try:
                X_test[i, j] = inv_exp(ref_skel, X_test[i, j])
            except:
                i = i + 1
    for i in range(X_test.shape[0]):
        for j in range(X_test[i].shape[0]):
            V0= X_test[i, j]
            X0= X_test[i, 0] #source
            Y0= X_test[0, 0] #target
            X_test[i, j] = new_PTG2(X0, V0, Y0)    
else:
    print('Casting variant doesn\'t exist')
    quit()

print("Preshape Space Done !!!!!!!")

num_frames = X_train.shape[1]
num_joints = X_train.shape[2]
dims = X_train.shape[3]
num_channels = num_joints * dims

rigid = True

transform_acc = []
for m in options:
    acc = []
    loss = []
    mod = m
    save_weights_dir = os.path.join('./models', mod)
    os.makedirs(save_weights_dir, exist_ok=True)

    if m == 'RigidTransform' or m == 'RigidTransformInit':
        rigid = True
    else:
        rigid = False
    
    for r in range(runs):
        print('run {}'.format(r))

        if rigid == True:
            model = RigidNet120(mod,num_frames,num_joints, r).to(device)
        else:
            model = NonRigidNet120(mod,num_frames,num_joints, r).to(device)
        criterion = nn.CrossEntropyLoss() 
        opt = torch.optim.Adam(model.parameters(),lr=learning_rate)
        steps = len(X_train) // batch_size
        print('model will be training for {} epochs with {} steps per epoch'.format(training_epochs,steps))

        model.train()
        for epoch in range(training_epochs):
            correct=0
            total=0
            running_loss = 0.0
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
            epoch_loss = epoch_loss / steps 
            print("Training Accuracy = {} : Training Loss {}".format(accuracy,epoch_loss)) 

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

        if save_max == False:
            torch.save(model.state_dict(), os.path.join(save_weights_dir, 'model_weights_{prot}.pth'))    
        else:
            if len(acc) == 0:
                torch.save(model.state_dict(), os.path.join(save_weights_dir, 'model_weights_{}.pth'.format(prot)))
            elif accuracy > max(acc):
                torch.save(model.state_dict(), os.path.join(save_weights_dir, 'model_weights_{}.pth'.format(prot)))

        acc.append(accuracy)

        print('Accuracy of the network for run {} is {} '.format(r, 100*correct_test / total_test)) 

    print('Finsished Running for {}'.format(m))

    transform_acc.append(max(acc))

print('Results of the four transform layer is {}'.format(transform_acc))