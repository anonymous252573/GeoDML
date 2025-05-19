import torch
import torch.nn as nn
import torch.nn.functional as F
import torchgeometry.core as tgmc

class NonRigidTransform(nn.Module):
    def __init__(self, num_frames, num_joints, r=0):
        super().__init__()
        self.num_frames, self.joints = num_frames, num_joints
        self.num_seq = 100 

        torch.manual_seed(r)

        angles = torch.Tensor(self.num_frames*self.joints,3)
        nn.init.uniform_(angles, a=-0.3,b=0.3)
        self.weights = nn.Parameter(angles)

    def forward(self, x):
        new_weights = tgmc.conversions.angle_axis_to_rotation_matrix(self.weights)[:,:3,:3]
        w_times_x= torch.matmul(x,new_weights)
        return w_times_x




