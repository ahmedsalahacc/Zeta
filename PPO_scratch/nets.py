import numpy as np
from NN import *

def Policy_Net(num_inputs,num_actions,lr=1e-3,continuous=False,Hidden_Layer=64):
    layers = None
    if len(num_inputs) > 1: # Using pixels as observations
        img_size = np.array(num_inputs[:2])
        size_after_conv1 = (img_size-4)//4 + 1
        size_after_conv2 = (size_after_conv1-4)//4 + 1
        size_after_conv3 = (size_after_conv2-4)//4 + 1
        num_inputs_to_linear = size_after_conv3[0]*size_after_conv3[1]*32

    else:
        layers = [Linear(num_inputs[0],Hidden_Layer),
                  Tanh(),
                  Linear(Hidden_Layer,Hidden_Layer),
                  Tanh(),
                  Linear(Hidden_Layer,num_actions),
                  Softmax()]

    if continuous:
        layers[-1] = Tanh()


    return Sequential(layers,{'lr':lr,'type':'Adam','epsilon':1e-7,'beta1':0.9,'beta2':0.999})
        
    
def Value_Net(num_inputs,lr=1e-3,Hidden_Layer=64):
    layers = None
    if len(num_inputs) > 1: # Using pixels as observations
        img_size = np.array(num_inputs[:2])
        size_after_conv1 = (img_size-4)//4 + 1
        size_after_conv2 = (size_after_conv1-4)//4 + 1
        size_after_conv3 = (size_after_conv2-4)//4 + 1
        num_inputs_to_linear = size_after_conv3[0]*size_after_conv3[1]*32
    else:
        layers = [Linear(num_inputs[0],Hidden_Layer),
              Tanh(),
              Linear(Hidden_Layer,Hidden_Layer),
              Tanh(),
              Linear(Hidden_Layer,1)]
    return Sequential(layers,{'lr':lr,'type':'Adam','epsilon':1e-7,'beta1':0.9,'beta2':0.999})
