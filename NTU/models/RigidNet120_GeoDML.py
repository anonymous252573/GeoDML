from layers.rigidtransform import RigidTransform
from layers.nonrigidtransform import NonRigidTransform
from layers.rigidtransforminit import RigidTransformInit
from layers.nonrigidtransforminit import NonRigidTransformInit
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchgeometry.core as tgmc


depth_1 = 128
kernel_size_1 = 3
stride_size = 2
depth_2 = 64
kernel_size_2 = 1
num_hidden = 512
num_labels = 120
dims = 3

class RigidNet120(nn.Module):
    def __init__(self,  mod = 'RigidTransform', num_frames = 100, num_joints = 25, r=0, dml_run=0):
        super(RigidNet120, self).__init__()
        self.num_channels = num_joints * dims
        self.mod = mod
        self.num_frames = num_frames
        self.num_joints = num_joints
        if mod == 'RigidTransform':
            self.rot = RigidTransform(num_frames,num_joints,r)
        elif mod == 'RigidTransformInit':
            self.rot = RigidTransformInit(num_frames,num_joints,r)
        self.conv1 = nn.Sequential(
            nn.Conv1d(self.num_channels, depth_1,kernel_size=kernel_size_1, stride=stride_size),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(depth_1, depth_2, kernel_size=kernel_size_2, stride=stride_size),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.LSTM = nn.LSTM(12, hidden_size=12, bidirectional =True)  
        self.pool=nn.MaxPool1d(kernel_size=2, stride=stride_size)
        self.fc1 = nn.Sequential(
            nn.Linear(depth_2*24, num_hidden),
            #nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.fc2 = nn.Sequential(
            nn.Linear(num_hidden, num_labels),
            #nn.ReLU(),
            #nn.Dropout(0.5)
        )

        torch.manual_seed(dml_run)
        self.mat_adap_glob_h = nn.Parameter(2 * torch.rand(1) -1)
        
    def forward(self, x):
        diag_mat_adap = torch.diag(torch.exp(self.mat_adap_glob_h.repeat(3)))
        diag_mat_adap = diag_mat_adap.unsqueeze(0).repeat(self.num_joints, 1, 1)
        x = torch.einsum('ijkl,kno->ijkl', x, diag_mat_adap)
        x = self.rot(x)
        x = x.view(x.size(0),self.num_joints*dims,self.num_frames)
        x = self.conv1(x)
        x = self.pool(self.conv2(x))
        x, _ = self.LSTM(x)
        x = x.view(x.size(0),-1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
