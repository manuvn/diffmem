import numpy as np
import torch
# from torch.Tensor import masked_fill_ as masked_fill
import torch.nn as nn
import torch.nn.functional as F

def where(cond, x1, x2):
    return cond.float() * x1 + (1 - cond.float()) * x2

def ClipW(weight):
    weight = where(weight >= 1, 1, weight)
    weight = where(weight <= -1, -1, weight)
    return weight

def Rectify(weight):
    weight = where(weight <= 0.01, 0.01, weight)
    return weight

################################
def to_categorical(y, num_classes=None, dtype='float32'):
    """Converts a class vector (integers) to binary class matrix.
    E.g. for use with categorical_crossentropy.
    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
        dtype: The data type expected by the input, as a string
            (`float32`, `float64`, `int32`...)
    # Returns
        A binary matrix representation of the input. The classes axis
        is placed last.
    # Example
    ```python
    # Consider an array of 5 labels out of a set of 3 classes {0, 1, 2}:
    > labels
    array([0, 2, 1, 2, 0])
    # `to_categorical` converts this into a matrix with as many
    # columns as there are classes. The number of rows
    # stays the same.
    > to_categorical(labels)
    array([[ 1.,  0.,  0.],
           [ 0.,  0.,  1.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.],
           [ 1.,  0.,  0.]], dtype=float32)
    ```
    """

    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical

################################
class Binarize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight):
        # probs = torch.tanh(weight)
        # binarize the weight
        weight_b = where(weight >= 0, 1, -1)
        ctx.save_for_backward(weight)
        return weight_b

    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        weight = ctx.saved_tensors
        grad_weight = None

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        if ctx.needs_input_grad[0]:
            grad_weight = grad_output
        return grad_weight
binarize = Binarize.apply

class NoisyBinLinear(nn.Module):
    def __init__(self, num_ip, num_op, sigma=0.2):
        super(NoisyBinLinear, self).__init__()
        var = 2/(num_ip + num_op)
        init_w = torch.empty(num_ip, num_op).normal_(mean=0, std=var**0.5)
        self.weight = nn.Parameter(init_w, requires_grad=True)
        self.sigma = sigma
    def forward(self, input):
        with torch.no_grad():
            self.weight.data = ClipW(self.weight.data)
        weight_b  = binarize(self.weight)
        rand = torch.randn_like(weight_b) * self.sigma
        randweight = rand + weight_b
        with torch.no_grad():
            randweight.data = ClipW(randweight.data)
        return input.mm(randweight)

################################
tern_boundary=0.25
class Ternarize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight):
        # probs = torch.tanh(weight)
        # binarize the weight
        weight_t = where(weight >= tern_boundary, 1, weight)
        weight_t = where((weight_t >= -tern_boundary)&(weight_t <= tern_boundary), 0, weight_t)
        weight_t = where((weight_t <= -tern_boundary), -1, weight_t)
        ctx.save_for_backward(weight)
        return weight_t

    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        weight = ctx.saved_tensors
        grad_weight = None

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        if ctx.needs_input_grad[0]:
            grad_weight = grad_output
        return grad_weight
ternarize = Ternarize.apply

class NoisyTernLinear(nn.Module):
    def __init__(self, num_ip, num_op, sigma=0.2):
        super(NoisyTernLinear, self).__init__()
        var = 2/(num_ip + num_op)
        init_w = torch.empty(num_ip, num_op).normal_(mean=0, std=var**0.5)
        self.weight = nn.Parameter(init_w, requires_grad=True)
        self.sigma = sigma
    def forward(self, input):
        with torch.no_grad():
            self.weight.data = ClipW(self.weight.data)
        weight_t  = ternarize(self.weight)
        rand = torch.randn_like(weight_t) * self.sigma * tern_boundary
        randweight = rand + weight_t
        return input.mm(randweight)