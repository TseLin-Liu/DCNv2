import sys
import time
sys.path.insert(0, '..')
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import  numpy as np
import torch,tqdm
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dcn_v2_mm import DCN as MyDCN

transformer1 = []
resize=224
if resize:
    transformer1 += [transforms.Resize(resize)]
transformer1 += [transforms.RandomVerticalFlip(p=0.5)]
transformer1 += [transforms.ToTensor()]
transformer1 = transforms.Compose(transformer1)

transformer2 = []
resize=224
if resize:
    transformer2 += [transforms.Resize(resize)]
transformer2 += [transforms.ToTensor()]
transformer2 = transforms.Compose(transformer2)
mnist_train = torchvision.datasets.CIFAR10(root="./", train=True, transform=transformer1, target_transform=None, download=True)
mnist_test = torchvision.datasets.CIFAR10(root="./", train=False, transform=transformer2, target_transform=None, download=True)

batch_size = 32                    

if sys.platform.startswith('win'):
    num_workers = 0
else:
    num_workers = 4       

train_iter = DataLoader(mnist_train, batch_size, shuffle=True, num_workers=num_workers)
test_iter = DataLoader(mnist_test, batch_size, shuffle=True, num_workers=num_workers)

def try_gpu():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    return device

device = try_gpu()


class Resblock(nn.Sequential):
    def __init__(self,in_channels,num_kernel,out_channels,stride):
        super(Resblock,self).__init__()
        self.in_channels,self.num_kernel,self.out_channels,self.stride=in_channels,num_kernel,out_channels,stride
        self.m1=nn.Sequential(
            MyDCN(self.in_channels, self.num_kernel, kernel_size=1 ,padding  = 0),
            nn.BatchNorm2d(self.num_kernel),
            nn.Conv2d(self.num_kernel, self.num_kernel, kernel_size=3, stride=self.stride, padding=1),
            nn.BatchNorm2d(self.num_kernel),
            MyDCN(self.num_kernel, self.out_channels, kernel_size=1,padding  = 0),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU(),
            )  
        if self.stride==2:
            self.m2=nn.Sequential(
                MyDCN(self.in_channels,self.out_channels,kernel_size=1,padding  = 0 ),
                nn.MaxPool2d(kernel_size=2, stride = self.stride)) 
        else:
            if self.in_channels==self.out_channels:
                self.m2=nn.Sequential()
            else:
                self.m2=nn.Sequential(
                    MyDCN(self.in_channels,self.out_channels,kernel_size=1,padding  = 0))
        
    def forward(self,x):
        # import pdb;pdb.set_trace()
        return self.m1(x)+self.m2(x)

class Add_module1(nn.Sequential):
    def __init__(self,n,in_channels,num_kernel,out_channels):
        super().__init__()
        self.n,self.in_channels,self.num_kernel,self.out_channels=n,in_channels,num_kernel,out_channels
        self.m3=nn.Sequential()
        for i in range(self.n):
            if i==0:
                self.m3.add_module('%d'%i,Resblock(self.in_channels,self.num_kernel,self.out_channels,stride=2))
            else:
                self.m3.add_module('%d'%i,Resblock(self.out_channels,self.num_kernel,self.out_channels,stride=1))

    def forward(self,x):
        return self.m3(x)


net=nn.Sequential(
    nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
    Resblock(64,64,256,1),
    Resblock(256,64,256,1),
    Resblock(256,64,256,1),
    Add_module1(4,256,128,512),
    Add_module1(23,512,256,1024),
    Add_module1(3,1024,512,2048),
    nn.AdaptiveAvgPool2d(output_size=(1, 1)),
    nn.Flatten(),
    nn.Linear(in_features=2048, out_features=10),
)

# X = torch.randn(size=(1, 1, 224, 224))
# for blk in net:
#     X = blk(X)
#     print(blk.__class__.__name__,'output shape:\t',X.shape)

# net.load_state_dict(torch.load("Rnet_params06"))
net.to(device)

lr=0.0001
num_epochs=2 
criterion = nn.CrossEntropyLoss()
'''----------------------------------------'''
def evaluate_accuracy(data_iter, net, device=torch.device('cpu')):
    net.eval()  #
    acc_sum, n = torch.tensor([0], dtype=torch.float32, device=device), 0
    for X, y in data_iter:
        X, y = X.to(device), y.to(device)
        with torch.no_grad():
            y = y.long()
            acc_sum += torch.sum((torch.argmax(net(X), dim=1) == y))
            n += y.shape[0]
    return acc_sum.item()/n

# test_acc = evaluate_accuracy(train_iter, net, device) 
# print('test acc %.3f'\
#           % (test_acc))
'''---------------------------------------'''
def train_ch5(net, train_iter, test_iter, criterion, num_epochs, batch_size, device, lr=None):
    """Train and evaluate a model with CPU or GPU."""
    print('training on', device)
    net.to(device)
    optimizer = optim.Adam(net.parameters(),betas=(0.9,0.999), eps =1e-08, lr=lr, amsgrad =False)
    for epoch in range(num_epochs):
        net.train() # Switch to training mode
        n, start = 0, time.time()
        train_l_sum = torch.tensor([0.0], dtype=torch.float32, device=device)
        train_acc_sum = torch.tensor([0.0], dtype=torch.float32, device=device)
        for X, y in tqdm.tqdm(train_iter):
            optimizer.zero_grad()
            X=X.to(device)
            y=y.to(device) 
            y_hat = net(X)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                y = y.long()
                train_l_sum += loss.float()
                train_acc_sum += (torch.sum((torch.argmax(y_hat, dim=1) == y))).float()
                n += y.shape[0]

        test_acc = evaluate_accuracy(test_iter, net, device) 
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'\
            % (epoch + 1, train_l_sum/n, train_acc_sum/n, test_acc, time.time() - start))

train_ch5(net, train_iter, test_iter, criterion, num_epochs, batch_size, device, lr)
torch.save(net.state_dict(),'Rnet_params06')