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

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
from processing.CenteredScaled import CenteredScaled, ref_rot
from processing.inv_exp import inv_exp
from models.RigidNet_GeoDML import RigidNet
from models.NonRigidNet_GeoDML import NonRigidNet

# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["NUMEXPR_NUM_THREADS"] = "1"
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# torch.set_num_threads(1)

# ** KIMORE **
for ex in range(2,10):

    parser = argparse.ArgumentParser()
    parser.add_argument('--cast', default='log_0refseq', type=str,help = 'LogMap sameref (log_sref)')
    parser.add_argument('--num_runs', default=10,type=int)
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--num_epoch', default=50, type=int)
    parser.add_argument('--save_max', default=True, type=bool, help='if true, will only save the model with the best accuracy, else it will save every model')
    parser.add_argument('--use_cuda', default=1, type=int)
    opt = parser.parse_args()

    rigid = True
    options=['RigidTransform','NonRigidTransform','RigidTransformInit','NonRigidTransformInit']

    training_epochs= opt.num_epoch
    learning_rate=opt.learning_rate
    runs = opt.num_runs
    save_max = opt.save_max
    cast = opt.cast

    use_cuda = 1
    cuda = torch.cuda.is_available()
    device = 'cuda' if (cuda == True and use_cuda == 1) else 'cpu'
    if device == 'cuda':
        print('Using CUDA')
        torch.cuda.empty_cache()
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        print('NOT using CUDA')

    print("NOW RUNNING FOR EXERCISE", ex)


    folder_path = r'C:\KIMORE_Dataset\data_and_features\Extracted_Features\cv_cs\xyz\{}'.format(ex)
    folds_acc = {}

    save_proba = r'C:\KIMORE_Dataset\GeoDML\softmax\E{}'.format(ex)
    os.makedirs(save_proba, exist_ok=True)
    file_path_ = os.path.join(save_proba, 'prob.json')

    fold_test_predi_proba = {}

    for fold in range(len(os.listdir(folder_path))):

        batch_size = 16
        batch_size_test = 16

        print('Loading data for fold {}'.format(fold+1))
        X_train_pos = np.load(os.path.join(folder_path, str(fold+1), 'train_data.npy'),  allow_pickle=True)
        y_train = np.load(os.path.join(folder_path, str(fold+1), 'train_label.pkl'), allow_pickle=True)
        X_test_pos = np.load(os.path.join(folder_path, str(fold+1), 'eval_data.npy'),  allow_pickle=True)
        y_test = np.load(os.path.join(folder_path, str(fold+1), 'eval_label.pkl'), allow_pickle=True)

        X_train = X_train_pos.transpose(0, 2, 3, 1, 4).squeeze(-1)
        X_test = X_test_pos.transpose(0, 2, 3, 1, 4).squeeze(-1)
        print("--------------------------------------")

        # get only labels no filenames
        y_train = y_train[1]
        y_test = y_test[1]
        y_train = np.array(y_train).astype('int32')
        y_test = np.array(y_test).astype('int32')
            
        print(X_train.shape)
        print(X_test.shape)
        print(y_train.shape)
        print(y_test.shape)

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
        else:
            print('Casting variant doesn\'t exist')
            quit()

        num_frames = X_train.shape[1]
        num_joints = X_train.shape[2]
        dims = X_train.shape[3]
        num_channels = num_joints * dims

        print('----- Preshape space done ----- !!!')

        acc_ = []
        test_proba_ = []
        for m in options:
            loss = []
            mod = m
            run_acc = []
            run_test_proba = []
            print('Running {} for fold {}'.format(mod,fold+1))
            for r in range(runs):
                dml_run_acc = []
                dml_run_prob = []
                for dml_run in range(runs):                    
                    if m == 'RigidTransform' or m == 'RigidTransformInit':
                        rigid = True
                    else:
                        rigid = False
                    
                    if rigid:
                        model_class = RigidNet
                    else:
                        model_class = NonRigidNet

                    model = model_class(mod=mod, run=r, dml_run=dml_run, glob_h=True).to(device)
                    criterion = nn.CrossEntropyLoss()
                    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
                    steps = len(X_train) // batch_size
                    # print('model will be training for {} epochs with {} steps per epoch'.format(training_epochs,steps))

                    # ******* for PARAM and FLOPS ******
                    # flops_and_param_tensor = torch.tensor(X_test[0:1]).to(device)
                    # flops, params = profile(model, inputs=(flops_and_param_tensor,))
                    # print(f"Parameters: {params / 1e6:.2f}M, FLOPs: {flops / 1e6:.2f}M")

                    model.train()
                    for epoch in range(training_epochs):
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
                
                    correct_test = 0
                    total_test = 0
                    model.eval()
                    with torch.no_grad():
                        steps = len(X_test) // batch_size_test
                        test_proba = np.zeros(len(X_test))
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
                            y_pred_softmax_prob = torch.softmax(outputs.data, dim = 1).clone().detach().cpu().numpy()
                            predicted_proba_= y_pred_softmax_prob[:,0]
                            if end_idx > len(X_test):
                                test_proba[start_idx:] = predicted_proba_
                            else:
                                test_proba[start_idx:end_idx] = predicted_proba_
                            total_test += labels.size(0)
                            correct_test += (predicted == labels.long()).sum().item()
            
                    accuracy = 100*correct_test/total_test
                    dml_run_acc.append(accuracy)
                    dml_run_prob.append(test_proba)

                run_acc.append(max(dml_run_acc))
                max_index_fl_run = dml_run_acc.index(max(dml_run_acc))
                run_test_proba.append(dml_run_prob[max_index_fl_run])

            acc_.append(max(run_acc))
            max_index = run_acc.index(max(run_acc))
            test_proba_.append(run_test_proba[max_index])
            
        print('--------------------------------')
        print('fold {} accuracies for 4 transfom layers'.format(fold+1), acc_)
        print('--------------------------------')

        # Save the accuarcy and test prediction probabilities for each fold in a dictionary.
        folds_acc[fold+1] = acc_
        fold_test_predi_proba[fold+1] = test_proba_

    print('Accuracies for all folds {} :'.format(folds_acc))

    # *** Compute the mean of each transform for all the folds. The highest is our model for this exercise. 
    acc_values = list(folds_acc.values())
    mean = [sum(elements) / len(elements) for elements in zip(*acc_values)]
    max_of_mean = max(mean)
    max_of_mean_index = mean.index(max_of_mean)

    # Take the corresponding predicted probabilities
    pred_prob_of_highest_accu = {key: value[max_of_mean_index] for key, value in fold_test_predi_proba.items()}
    print('Test predict probabilities for all folds: {}'.format(pred_prob_of_highest_accu))

    # ******* Save the predicted probabuilities results *****
    with open(file_path_, "wb") as f:
        pickle.dump(pred_prob_of_highest_accu, f)

    if max_of_mean_index == 0 :
        print("The best transform layer for this exercise based on the 5 folds is {}, with mean acc of {}".format(options[max_of_mean_index], max_of_mean))
    elif max_of_mean_index == 1:
        print("The best transform layer for this exercise based on the 5 folds is {}, with mean acc of {}".format(options[max_of_mean_index], max_of_mean))
    elif max_of_mean_index == 2:
        print("The best transform layer for this exercise based on the 5 folds is {}, with mean acc of {}".format(options[max_of_mean_index], max_of_mean))
    else:
        print("The best transform layer for this exercise based on the 5 folds is {}, with mean acc of {}".format(options[max_of_mean_index], max_of_mean))

