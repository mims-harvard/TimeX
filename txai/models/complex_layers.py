import torch
from torch.nn import Module, Parameter, init
from torch.nn import Conv2d, Linear, BatchNorm1d, BatchNorm2d
from torch.nn import ConvTranspose2d
from torch.nn.functional import relu, max_pool2d, avg_pool2d, dropout, dropout2d, interpolate, sigmoid, tanh

def complex_relu(input):
    return relu(input.real).type(torch.complex64)+1j*relu(input.imag).type(torch.complex64)

def complex_sigmoid(input):
    return sigmoid(input.real).type(torch.complex64)+1j*sigmoid(input.imag).type(torch.complex64)

def apply_complex(fr, fi, input, dtype = torch.complex64):
    return (fr(input.real)-fi(input.imag)).type(dtype) \
            + 1j*(fr(input.imag)+fi(input.real)).type(dtype)

class ComplexSigmoid(Module):

     def forward(self,input):
         return complex_sigmoid(input)

class ComplexReLU(Module):

     def forward(self,input):
         return complex_relu(input)

class ComplexLinear(Module):

    def __init__(self, in_features, out_features):
        super(ComplexLinear, self).__init__()
        self.fc_r = Linear(in_features, out_features)
        self.fc_i = Linear(in_features, out_features)

    def forward(self, input):
        return apply_complex(self.fc_r, self.fc_i, input)