import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tensorboardX import SummaryWriter


sns.set()
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    
print('Using PyTorch version:', torch.__version__, ' Device:', device)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def where(cond, x1, x2):
    return cond.float() * x1 + (1 - cond.float()) * x2


class Binarize(torch.autograd.Function):

    @staticmethod
    def forward(ctx, weight):
        probs = torch.tanh(weight)
        # binarize the weight
        weight_b = where(probs >= 0, 1, -1)
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


class BinLinear(nn.Module):
    def __init__(self, num_ip, num_op):
        super(BinLinear, self).__init__()
        self.weight = nn.Parameter(torch.empty(num_ip, num_op).uniform_(), requires_grad=True)

    def forward(self, input):
        weight_b = binarize(self.weight)
        return input.mm(weight_b)

class NoisyBinLinear(nn.Module):
    def __init__(self, num_ip, num_op, sigma=0.2):
        super(NoisyBinLinear, self).__init__()
        self.weight = nn.Parameter(torch.empty(num_ip, num_op).uniform_(), requires_grad=True)
        self.sigma = sigma
    def forward(self, input):
        weight_b  = binarize(self.weight)
        rand = torch.randn_like(weight_b) * self.sigma
        randweight = rand + weight_b
        return input.mm(randweight)


class Net(nn.Module):
    def __init__(self, nntype='Linear', nunits=50, nhidden=3):
        super(Net, self).__init__()

        if nntype == 'Linear':
            fc = nn.Linear
        elif nntype == 'Binary':
            fc = BinLinear
        elif nntype == 'NoisyBinary':
            fc = NoisyBinLinear

        # self.fc1 = nn.Linear(28*28, 50)
        # self.fc1_drop = nn.Dropout(0.2)
        # self.fc2 = nn.Linear(50, 50)
        # self.fc2_drop = nn.Dropout(0.2)
        # self.fc3 = nn.Linear(50, 10)

        ipdropout = nn.Dropout(0.2)
        iplayer = fc(28*28, nunits)
        # self.layers = [ipdropout, iplayer]
        self.layers = [iplayer]
        self.layers += [nn.Dropout(0.5)]
        for idx in range(nhidden):
            self.layers += [fc(nunits, nunits)]
            self.layers += [nn.ReLU(inplace=True)]
            self.layers += [nn.BatchNorm1d(nunits)]
            self.layers += [nn.Dropout(0.5)]

        oplayer = fc(nunits, 10)
        self.layers = self.layers+[oplayer]
        softmax = nn.Softmax(1)
        self.layers = self.layers+[softmax]
        self.net = nn.Sequential(*self.layers)

    def forward(self, x):
        # x = x.view(-1, 28*28)
        # x = F.relu(self.fc1(x))
        # x = self.fc1_drop(x)
        # x = F.relu(self.fc2(x))
        # x = self.fc2_drop(x)
        # return F.log_softmax(self.fc3(x), dim=1)

        x = x.view(-1, 28*28)
        return self.net(x)


def train(epoch, log_interval=200):
    # Set model to training mode
    model.train()
    
    # Loop over each batch from the training set
    for batch_idx, (data, target) in enumerate(train_loader):
        # Copy data to GPU if needed
        data = data.to(device)
        target = target.to(device)

        # Zero gradient buffers
        optimizer.zero_grad() 
        
        # Pass data through the network
        output = model(data)

        # Calculate loss
        loss = criterion(output, target)

        # Backpropagate
        loss.backward()
        
        # Update weights
        optimizer.step()
        
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.item()))
    return loss

def validate(loss_vector, accuracy_vector):
    model.eval()
    val_loss, correct = 0, 0
    for data, target in validation_loader:
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        val_loss += criterion(output, target).data.item()
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    val_loss /= len(validation_loader)
    loss_vector.append(val_loss)

    accuracy = 100. * correct.to(torch.float32) / len(validation_loader.dataset)
    accuracy_vector.append(accuracy)
    
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        val_loss, correct, len(validation_loader.dataset), accuracy))
    return accuracy

if __name__ == '__main__':

    # tensorboard --logdir logs
    parser = argparse.ArgumentParser(description='MNIST binary MLP')
    parser.add_argument('--nntype', default='Binary', type=str, help='Linear, Binary, NoisyBinary')
    parser.add_argument('--nunits', default=1024, type=int, help='Number of units in hidden layer')
    parser.add_argument('--nhidden', default=3, type=int, help='Number of hidden layers')
    parser.add_argument('--batch_size', default=200, type=int, help='Batch size')
    parser.add_argument('--nepochs', default=300, type=int, help='Number of training epochs')
    parser.add_argument('--lr', default=0.01, type=float, help='Initial learning rate')
    parser.add_argument('--lr_patience', default=20, type=int, help='Learning rate patience')
    parser.add_argument('--cuda', type=str, default='y',
                            help='y uses GPU. n uses CPU')
    args = parser.parse_args()

    tboard_dir = './logs/'+args.nntype+'_B'+str(args.batch_size)+'_H'+str(args.nhidden)+'_lr'+str(args.lr)
    # writer = SummaryWriter('./logs')
    writer = SummaryWriter(tboard_dir)

    # model
    model = Net(nntype=args.nntype, nunits=args.nunits, nhidden=args.nhidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)#, momentum=0.5)
    criterion = nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=args.lr_patience, verbose=True)

    print(model)

    # data
    batch_size = args.batch_size
    train_dataset = datasets.MNIST('./data', 
                                train=True, 
                                download=True, 
                                transform=transforms.ToTensor())
    validation_dataset = datasets.MNIST('./data', 
                                        train=False, 
                                        transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                            batch_size=batch_size, 
                                            shuffle=True)
    validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset, 
                                                    batch_size=batch_size, 
                                                    shuffle=False)
    for (X_train, y_train) in train_loader:
        print('X_train:', X_train.size(), 'type:', X_train.type())
        print('y_train:', y_train.size(), 'type:', y_train.type())
        break

    # Train
    epochs = args.nepochs
    lossv, accv = [], []
    for epoch in range(1, epochs + 1):
        l = train(epoch)
        writer.add_scalar('Loss', l, epoch)
        a = validate(lossv, accv)  
        writer.add_scalar('Accuracy', a, epoch)
        scheduler.step(l)
        lr = get_lr(optimizer)
        writer.add_scalar('Learning rate', lr, epoch)

    # Plot performance
    plt.figure(figsize=(5,3))
    plt.plot(np.arange(1,epochs+1), lossv)
    plt.title('validation loss')

    plt.figure(figsize=(5,3))
    plt.plot(np.arange(1,epochs+1), accv)
    plt.title('validation accuracy');

    writer.close()
