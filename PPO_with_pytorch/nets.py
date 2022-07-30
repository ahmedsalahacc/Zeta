import numpy as np
import torch
import torch.nn as nn

class Policy_Net (torch.nn.Module):
    def __init__(self,num_inputs,num_actions,continuous=False,Hidden_Layer=64):
        super().__init__()
        
        if len(num_inputs) > 1: # Using pixels as observations
            img_size = np.array(num_inputs[:2])
            size_after_conv1 = (img_size-4)//4 + 1
            size_after_conv2 = (size_after_conv1-4)//4 + 1
            size_after_conv3 = (size_after_conv2-4)//4 + 1
            num_inputs_to_linear = size_after_conv3[0]*size_after_conv3[1]*32
            self.policy = torch.nn.Sequential(
                nn.Conv2d(num_inputs[2],8,4,stride=4),
                nn.ReLU(),
                nn.Conv2d(8,16,4,stride=4),
                nn.ReLU(),
                nn.Conv2d(16,32,4,stride=4),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(num_inputs_to_linear,Hidden_Layer),
                nn.Tanh(),
                nn.Linear(Hidden_Layer,Hidden_Layer),
                nn.Tanh(),
                nn.Linear(Hidden_Layer,num_actions),
                nn.LogSoftmax(dim = 1)
            )  
        else:
            self.policy = torch.nn.Sequential(
                nn.Linear(num_inputs[0],Hidden_Layer),
                nn.Tanh(),
                nn.Linear(Hidden_Layer,Hidden_Layer),
                nn.Tanh(),
                nn.Linear(Hidden_Layer,num_actions),
                nn.LogSoftmax(dim = 1)
            )
        
        if continuous:
            self.policy[-1] = nn.Tanh()
        
    def forward(self,state):
        policy_pred = self.policy(state)
        return policy_pred
    
class Value_Net (torch.nn.Module):
    def __init__(self,num_inputs,Hidden_Layer=64):
        super().__init__()
        if len(num_inputs) > 1: # Using pixels as observations
            img_size = np.array(num_inputs[:2])
            size_after_conv1 = (img_size-4)//4 + 1
            size_after_conv2 = (size_after_conv1-4)//4 + 1
            size_after_conv3 = (size_after_conv2-4)//4 + 1
            num_inputs_to_linear = size_after_conv3[0]*size_after_conv3[1]*32
            self.actric = torch.nn.Sequential(
                nn.Conv2d(num_inputs[2],8,4,stride=4),
                nn.ReLU(),
                nn.Conv2d(8,16,4,stride=4),
                nn.ReLU(),
                nn.Conv2d(16,32,4,stride=4),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(num_inputs_to_linear,Hidden_Layer),
                nn.Tanh(),
                nn.Linear(Hidden_Layer,Hidden_Layer),
                nn.Tanh(),
                nn.Linear(Hidden_Layer,1)
            )  
        else:
            self.actric = torch.nn.Sequential(
                nn.Linear(num_inputs[0],Hidden_Layer),
                nn.Tanh(),
                nn.Linear(Hidden_Layer,Hidden_Layer),
                nn.Tanh(),
                nn.Linear(Hidden_Layer,1)
            )
    def forward(self,state):
        val = self.actric(state)
        return val
