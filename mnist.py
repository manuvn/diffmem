import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import torch
import sys
# from torch.Tensor import masked_fill_ as masked_fill
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tensorboardX import SummaryWriter
from utils import *

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

sns.set()
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    
print('Using PyTorch version:', torch.__version__, ' Device:', device)

class Net(nn.Module):
    def __init__(self, nntype='Linear', nunits=50, nhidden=3, sigma=0.2):
        super(Net, self).__init__()
        kwargs = {}
        if nntype == 'Linear':
            fc = nn.Linear
        elif nntype == 'Binary':
            fc = BinLinear
        elif nntype == 'NoisyBinary':
            fc = NoisyBinLinear
            kwargs['sigma'] = sigma
        elif nntype == 'Ternary':
            fc = TernLinear
        elif nntype == 'NoisyTernary':
            fc = NoisyTernLinear
            kwargs['sigma'] = sigma
        elif nntype == 'DiffMem':
            fc = DiffMemLinear
            kwargs['sigma'] = sigma
        elif nntype == 'DiffMemN':
            fc = DiffMemNormLinear
            kwargs['sigma'] = sigma
        else:
            fc = nn.Linear

        drop_in = 0.2
        drop_h = 0.5

        ipdropout = nn.Dropout(drop_in)
        iplayer = fc(28*28, nunits, **kwargs)
        self.layers = [ipdropout, iplayer]
        # self.layers = [iplayer]
        self.layers += [nn.Dropout(drop_h)]
        for idx in range(nhidden):
            self.layers += [fc(nunits, nunits, **kwargs)]
            self.layers += [nn.ReLU(inplace=True)]
            self.layers += [nn.BatchNorm1d(nunits, eps=1e-6, momentum=0.9)]
            self.layers += [nn.Dropout(drop_h)]

        oplayer = fc(nunits, 10, **kwargs)
        self.layers = self.layers+[oplayer]
        softmax = nn.Softmax(1)
        self.layers = self.layers+[softmax]
        self.net = nn.Sequential(*self.layers)

    def forward(self, x):
        x = x.view(-1, 28*28)
        return self.net(x)


def train(epoch, log_interval=200):
    # Set model to training mode
    model.train()
    
    # Loop over each batch from the training set
    for batch_idx, (data, target) in enumerate(train_loader):
        # Copy data to GPU if needed
        if args.loss == 'Hinge':
            target = torch.Tensor(to_categorical(target,10))
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
        if args.loss == 'Hinge':
            tgt = torch.Tensor(to_categorical(target,10))
        else:
            tgt = target
        data = data.to(device)
        target = target.to(device)
        tgt = tgt.to(device)
        output = model(data)
        val_loss += criterion(output, tgt).data.item()
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()
        # correct += output.eq(target.data).cpu().sum()

    val_loss /= len(validation_loader)
    loss_vector.append(val_loss)

    accuracy = 100. * correct.to(torch.float32) / len(validation_loader.dataset)
    accuracy_vector.append(accuracy)
    
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        val_loss, correct, len(validation_loader.dataset), accuracy))
    return accuracy

if __name__ == '__main__':
    """
    Run script: 
    python mnist.py --nntype=NoisyBinary --nunits=1024 --nhidden=3 --batch_size=100 --nepochs=200
    """

    # tensorboard --logdir logs
    parser = argparse.ArgumentParser(description='MNIST binary MLP')
    parser.add_argument('--nntype', default='Binary', type=str, help='Linear, Binary, NoisyBinary, Ternary, NoisyTernary, DiffMem, DiffMemN')
    parser.add_argument('--nunits', default=1024, type=int, help='Number of units in hidden layer')
    parser.add_argument('--nhidden', default=3, type=int, help='Number of hidden layers')
    parser.add_argument('--batch_size', default=200, type=int, help='Batch size')
    parser.add_argument('--nepochs', default=300, type=int, help='Number of training epochs')
    parser.add_argument('--lr', default=0.001, type=float, help='Initial learning rate')
    parser.add_argument('--sigma', default=0.1, type=float, help='Diff mem spread')
    parser.add_argument('--lr_patience', default=100, type=int, help='Learning rate patience')
    parser.add_argument('--logpath', type=str, default='./logs/', 
                            help='Save path for the logs')
    parser.add_argument('--loss', type=str, default='Hinge', 
                            help='Hinge, CrossEntropy')
    parser.add_argument('--cuda', type=str, default='y',
                            help='y uses GPU. n uses CPU')
    args = parser.parse_args()

    # model
    model = Net(nntype=args.nntype, 
                nunits=args.nunits, 
                nhidden=args.nhidden, 
                sigma=args.sigma).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)#, momentum=0.5)
    if args.loss == 'Hinge':
        criterion = nn.MultiLabelSoftMarginLoss()
    elif args.loss == 'CrossEntropy':
        criterion = nn.CrossEntropyLoss()
    else:
        print('Unsupported loss')
        sys.exit()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', 
                            factor=0.1, 
                            patience=args.lr_patience, 
                            verbose=True,
                            min_lr = 1e-4)

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
    T = time.strftime("M%mD%dH%HM%M")
    tboard_dir = args.logpath+args.nntype+'_B'+str(args.batch_size)+'_H'+str(args.nhidden)+'_N'+str(args.nunits)+'_S'+str(args.sigma)+'_lr'+str(args.lr)+'-Time-'+T
    # writer = SummaryWriter('./logs')
    writer = SummaryWriter(tboard_dir)

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
