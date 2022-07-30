import numpy as np

class Linear:

    def __init__(self, in_no, out_no):
        self.shape = (out_no, in_no)
        self.W = np.random.randn(*self.shape)*np.sqrt(2/in_no)
        self.b = np.zeros((out_no,1))
        self.dW = np.zeros(self.shape)
        self.db = np.zeros((out_no,1))
        self.lastIn = None

    # ASSUMPTION: There's only one training example

    def forward(self, in1):
        # in1 shape=(dim_in, batch_size)
        in1 = in1.T
        self.lastIn = in1
        return (np.dot(self.W, in1) + self.b).T

    def backward(self, dOut):
        # dOut shape=(dim_out,batch_size)
        dOut = dOut.T
        self.dW += 1 / self.lastIn.shape[1] * np.dot(dOut, self.lastIn.T)
        self.db += 1 / self.lastIn.shape[1] * np.sum(dOut,axis = 1, keepdims = True)
        return np.dot(self.W.T, dOut).T
    
    def zero_grad(self):
        self.dW = np.zeros(self.shape)
        self.db = np.zeros((self.shape[0],1))
        self.lastIn = None


class ReLU:
    def __init__(self):
        self.lastIn = None

    def forward(self, in1):
        self.lastIn  = in1
        return np.maximum(in1, 0)

    def backward(self, dOut):
        return (self.lastIn > 0)*dOut
    
    def zero_grad(self):
        self.lastIn = None


class Tanh:
    def __init__(self):
        self.lastIn = None

    def forward(self,in1):
        self.lastIn  = in1
        return np.tanh(in1)

    def backward(self, dOut):
        return (1 - np.square(np.tanh(self.lastIn)))*dOut
    
    def zero_grad(self):
        self.lastIn = None

class MSE:
    def __init__(self):
        self.Y = None
        self.lastIn = None

    def forward(self, in1,in2):
        self.Y = in1
        self.lastIn = in2
        return np.linalg.norm(in1-in2)

    def backward(self, dOut):
        return 2* (self.lastIn-self.Y) * dOut
    
    def zero_grad(self):
        self.lastIn = None
        
class LogSoftmax:
    def __init__(self):
        self.lastIn = None
    def forward(self,in1):
        self.lastIn = in1
        exp = np.exp(in1)
        return exp - np.sum(exp,axis=1).reshape(-1,1)
    def backward(self,dOut):
        exp = np.exp(dOut)
        sum_exp = np.sum(exp,axis=1).reshape(-1,1)
        return 1 - exp/sum_exp
    def zero_grad(self):
        self.lastIn = None
        

        
class Softmax:
    def __init__(self):
        self.lastIn = None
    def forward(self,in1):
        self.lastIn = in1
        exp = np.exp(in1)
        return exp/(np.sum(exp,axis=1).reshape(-1,1)+1e-7)
    def backward(self,dOut):
        exp = np.exp(self.lastIn)
        sum_exp = np.sum(exp,axis=1).reshape(-1,1)
        return (exp*sum_exp-exp*exp)/((sum_exp*sum_exp)+1e-7)*dOut
    def zero_grad(self):
        self.lastIn = None
        

class Flatten:
    def __init__(self):
        self.lastIn = None
        
    def forward(self,in1):
        self.lastIn = in1
        return in1.reshape(in1.shape[0],-1)
    
    def backward(self,dOut):
        return dOut.reshape(*self.lastIn.shape)
    
    def zero_grad(self):
        self.lastIn = None

class Conv2D:

    def __init__(self, n_C_prev, n_C, f=3, stride=1, pad=0):
        self.stride = stride
        self.pad = pad
        self.f = f
        self.n_C_prev = n_C_prev
        self.n_C = n_C
        self.W = np.random.randn(f,f,n_C_prev,n_C)
        self.b = np.random.randn(1,1,1,n_C)

        self.dW = np.zeros((f,f,n_C_prev,n_C))
        self.db = np.zeros((1,1,1,n_C))
        self.lastIn = None


    def forward(self, In1):
        self.lastIn = In1
        In1_pad = np.pad(In1, ((0,0),(self.pad,self.pad),(self.pad,self.pad),(0,0)), mode='constant', constant_values = (0,0))

        (m, n_H_prev, n_W_prev, n_C_prev) = self.lastIn.shape
        n_H = int((n_H_prev + 2 *self.pad - self.f)/self.stride) +1;
        n_W = int((n_W_prev + 2 *self.pad -self.f)/self.stride) +1;
        Z = np.zeros((m,n_H,n_W,self.n_C))

        for i in range(m):
            in1_pad = In1_pad[i,:,:,:]
            for h in range(n_H):
                vert_start = self.stride*h
                vert_end = vert_start + self.f

                for w in range(n_W):
                    horiz_start = self.stride* w
                    horiz_end = horiz_start + self.f

                    for c in range(self.n_C):
                        in1_slice_prev = in1_pad[vert_start:vert_end, horiz_start:horiz_end, :];
                        weights = self.W[:,:,:,c]
                        biases = self.b[:,:,:,c]
                        Z[i, h, w, c] = float(np.sum(in1_slice_prev * weights)+biases);

        return Z

    def backward(self, dZ):
        (m, n_H_prev, n_W_prev, n_C_prev) = self.lastIn.shape
        (m, n_H, n_W, n_C) = dZ.shape

        dlastIn = np.zeros(self.lastIn.shape)
        lastIn_pad = np.pad(self.lastIn, ((0,0),(self.pad,self.pad),(self.pad,self.pad),(0,0)), mode='constant', constant_values = (0,0))
        dlastIn_pad = np.pad(dlastIn, ((0,0),(self.pad,self.pad),(self.pad,self.pad),(0,0)), mode='constant', constant_values = (0,0))

        for i in range(m):
            a_prev_pad = lastIn_pad[i,:,:,:]
            da_prev_pad = dlastIn_pad[i,:,:,:]

            for h in range(n_H):                   # loop over vertical axis of the output volume
                for w in range(n_W):               # loop over horizontal axis of the output volume
                    for c in range(n_C):           # loop over the channels of the output volume

                        vert_start = self.stride*h
                        vert_end = self.stride*h + self.f
                        horiz_start = self.stride*w
                        horiz_end = self.stride*w + self.f

                        # Use the corners to define the slice from a_prev_pad
                        a_slice = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end,:]

                        # Update gradients for the window and the filter's parameters using the code formulas given above
                        da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += self.W[:,:,:,c] * dZ[i,h,w,c];
                        self.dW[:,:,:,c] += a_slice * dZ[i,h,w,c]
                        self.db[:,:,:,c] += dZ[i,h,w,c]

        # Set the ith training example's dA_prev to the unpadded da_prev_pad (Hint: use X[pad:-pad, pad:-pad, :])
            dlastIn[i, :, :, :] = da_prev_pad[self.pad:-self.pad, self.pad:-self.pad,:]

        return dlastIn
    
    def zero_grad(self):
        self.dW = np.zeros((self.f,self.f,self.n_C_prev,self.n_C))
        self.db = np.zeros((1,1,1,self.n_C))
        self.lastIn = None

class Sequential:
    def __init__(self, layers,optim_params):
        self.layers = layers
        self.optim_params = optim_params
        self.output = None
        if self.optim_params['type'] == 'Adam':
            for layer in self.layers:
                if type(layer) in [Linear, Conv2D]:
                    layer.vdW = np.zeros(layer.dW.shape)
                    layer.vdb = np.zeros(layer.db.shape)
                    layer.sdW = np.ones(layer.dW.shape)
                    layer.sdb = np.ones(layer.db.shape)
        
    def forward(self,inp):
        for layer in self.layers:
            inp = layer.forward(inp)
        self.output = inp
        return inp
    
    def backward(self, loss_d):
        if self.output is None:
            print("Skipping backprop as no forward was done yet")
        else:
            out = loss_d
            for layer in self.layers[::-1]:
                out = layer.backward(out)
                if self.optim_params['type'] == 'Adam' and type(layer) in [Linear, Conv2D]: # Adam updates
                    layer.vdW = self.optim_params['beta1']*layer.vdW + (1-self.optim_params['beta1'])*layer.dW
                    layer.vdb = self.optim_params['beta1']*layer.vdb + (1-self.optim_params['beta1'])*layer.db
                    
                    layer.sdW = self.optim_params['beta2']*layer.sdW + (1-self.optim_params['beta2'])*np.square(layer.dW)
                    layer.sdb = self.optim_params['beta2']*layer.sdb + (1-self.optim_params['beta2'])*np.square(layer.db)
                
    def zero_grad(self):
        for layer in self.layers:
            layer.zero_grad()
    
    def optimize(self):
        for layer in self.layers:
            if type(layer) is Linear or type(layer) is Conv2D:
                if self.optim_params['type'] == 'SGD':
                    layer.W -= layer.dW*self.optim_params['lr']
                    layer.b -= layer.db*self.optim_params['lr']
                elif self.optim_params['type'] == 'Adam':
                    layer.W -= self.optim_params['lr']*layer.vdW/ np.sqrt(layer.sdW+self.optim_params['epsilon'])
                    layer.b -= self.optim_params['lr']*layer.vdb/ np.sqrt(layer.sdb+self.optim_params['epsilon'])