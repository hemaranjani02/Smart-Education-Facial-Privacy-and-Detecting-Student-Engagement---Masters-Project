import numpy as np
from torch import nn, optim
import torch
from torch.nn.functional import one_hot

class MLP(nn.Module): #Downgraded
    def __init__(self, network_shape,
                 n_classes,
                 loss_function='CrossEntropyLoss', loss_function_params=dict(),
                 optimizer='Adam', optimizer_params=dict(lr=1e-3, weight_decay=1e-5),
                 af='ReLU', af_params=dict(),
                 af_fin='Sigmoid', af_fin_params=dict(),
                 attach_softmax:bool=False,
                 **kwargs):
        
        super(MLP,self).__init__() #Legacy mode
        self.shape=np.array(network_shape)
        # o o o o o o : [0, shape[0]] > Linear, batchnorm, af (regular)
        #   o o o o   : [1, shape[1]] > Linear, batchnorm, af (regular)
        #     o o     : [2, shape[2]] > Linear, batchnorm, af_fin
        #      o      : [3, n_classes] > Linear
        assert len(self.shape.shape)==1 #supports 1-dimensional input
        n_layers=len(self.shape)
        self.n_classes=n_classes
        
        estimator=[]
        for i, n in enumerate(self.shape):
            if i<n_layers-2:
                estimator+=[
                    nn.Linear(self.shape[i], self.shape[i+1]),
                    nn.BatchNorm1d(num_features=self.shape[i+1]),
                    getattr(nn, af)(**af_params)
                    ]
            elif i==n_layers-2: #The last layer
                estimator+=[
                    nn.Linear(self.shape[i], self.shape[i+1]),
                    nn.BatchNorm1d(num_features=self.shape[i+1]),
                    getattr(nn, af_fin)(**af_fin_params)
                ]
            elif i==n_layers-1: #Additional linear layer
                estimator+=[
                    nn.Linear(self.shape[i], self.n_classes)
                ]
                
        
        self.estimator=nn.Sequential(*estimator)
        self.softmax=nn.Softmax(dim=1)
        self.loss_function=loss_function #name of loss function
        self.criterion=getattr(nn, loss_function)(**loss_function_params)
        # self.scheduler=None if ...
        self.learner_type=2 #leagcy compatible
        self.optimizer=getattr(optim, optimizer)(self.parameters(), **optimizer_params) #leagcy compatible
        # self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu') #gpu, legacy compatible
        self.train_losses=None #legacy compatible
        self.test_losses=None #legacy compatible
        # self.to(self.device)
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device) #legacy compatible
        self.attach_softmax=attach_softmax
        
    def forward(self, x):
        if self.attach_softmax:
            return self.softmax(self.estimator(x))
        else:
            return self.estimator(x)
        
    def train(self, dataLoader, epochs, attach_label_onehot=False, attach_label_binarize=True):
        assert attach_label_onehot!=attach_label_binarize #you can use only one option
        for epoch in range(epochs):
            for x, label in dataLoader:
                if attach_label_onehot:
                    label=one_hot(label, num_classes=self.n_classes)
                elif attach_label_binarize:
                    label=(label>=2.0).int().long()
                self.optimizer.zero_grad()
                output=self(x)
                if self.n_classes>2:
                    loss=0
                    target=label.reshape(-1, self.n_classes).float()
                    for i in range(self.n_classes):
                        if False:
                            loss+=self.criterion(output, target[:, i])
                        else:
                            loss+=self.criterion(output[:, i], target[:, i])
                else:
                    loss=self.criterion(output, label.reshape(-1).long())
                self.train_losses=loss if self.train_losses==None else torch.vstack((self.train_losses, loss))
                loss.backward()
                self.optimizer.step()
                
    def test(self, dataLoader, attach_label_onehot=False, attach_label_binarize=True):
        assert attach_label_onehot!=attach_label_binarize #you can use only one option
        with torch.no_grad():
            result=None
            # for x, label in dataLoader:
            for i, (x, label) in enumerate(dataLoader):
                if attach_label_onehot:
                    label=one_hot(label, num_classes=self.n_classes)
                elif attach_label_binarize:
                    label=(label>=2.0).int().long()
                self.optimizer.zero_grad()
                output=self(x)
                if self.n_classes>2:
                    loss=0
                    target=label.reshape(-1, self.n_classes).float()
                    for i in range(self.n_classes):
                        if False:
                            loss+=self.criterion(output, target[:, i])
                        else:
                            loss+=self.criterion(output[:, i], target[:, i])
                else:
                    loss=self.criterion(output, label.reshape(-1).long())
                self.test_losses=loss if self.test_losses==None else torch.vstack((self.test_losses, loss))
                result=output if result==None else torch.vstack((result, output))
            if self.learner_type==2:
                pred=result
            return pred
        
        
class MLPRegressor(nn.Module): #Downgraded
    def __init__(self, network_shape,
                 n_classes,
                 loss_function='MSELoss', loss_function_params=dict(),
                 optimizer='Adam', optimizer_params=dict(lr=1e-3, weight_decay=1e-5),
                 af='ReLU', af_params=dict(),
                 af_fin='Tanh', af_fin_params=dict(),
                 **kwargs):
        
        super(MLPRegressor,self).__init__() #Legacy mode
        self.shape=np.array(network_shape)
        # o o o o o o : [0, shape[0]] > Linear, batchnorm, af (regular)
        #   o o o o   : [1, shape[1]] > Linear, batchnorm, af (regular)
        #     o o     : [2, shape[2]] > Linear, batchnorm, af_fin
        #      o      : [3, n_classes] > Linear
        assert len(self.shape.shape)==1 #supports 1-dimensional input
        n_layers=len(self.shape)
        
        estimator=[]
        for i, n in enumerate(self.shape):
            if i<n_layers-2:
                estimator+=[
                    nn.Linear(self.shape[i], self.shape[i+1]),
                    nn.BatchNorm1d(num_features=self.shape[i+1]),
                    getattr(nn, af)(**af_params)
                    ]
            elif i==n_layers-2: #The last layer
                estimator+=[
                    nn.Linear(self.shape[i], self.shape[i+1]),
                    nn.BatchNorm1d(num_features=self.shape[i+1]),
                    getattr(nn, af_fin)(**af_fin_params)
                ]
            elif i==n_layers-1: #Additional linear layer
                estimator+=[
                    nn.Linear(self.shape[i], n_classes)
                ]
                
        
        self.estimator=nn.Sequential(*estimator)
        # self.softmax=nn.Softmax(dim=1)
        self.loss_function=loss_function #name of loss function
        self.criterion=getattr(nn, loss_function)(**loss_function_params)
        # self.scheduler=None if ...
        self.learner_type=2 #leagcy compatible
        self.optimizer=getattr(optim, optimizer)(self.parameters(), **optimizer_params) #leagcy compatible
        # self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu') #gpu, legacy compatible
        self.train_losses=None #legacy compatible
        self.test_losses=None #legacy compatible
        # self.to(self.device)
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device) #legacy compatible
        
    def forward(self, x):
        # if self.attach_softmax:
            # return self.softmax(self.estimator(x))
        # else:
        return self.estimator(x)
        
    def train(self, dataLoader, epochs):
        for epoch in range(epochs):
            for x, label in dataLoader:
                self.optimizer.zero_grad()
                output=self(x)
                loss=self.criterion(output, label.float())
                self.train_losses=loss if self.train_losses==None else torch.vstack((self.train_losses, loss))
                loss.backward()
                self.optimizer.step()
                
    def test(self, dataLoader):
        with torch.no_grad():
            result=None
            # for x, label in dataLoader:
            for i, (x, label) in enumerate(dataLoader):
                output=self(x)
                loss=self.criterion(output, label.float())
                self.test_losses=loss if self.test_losses==None else torch.vstack((self.test_losses, loss))
                result=output if result==None else torch.vstack((result, output))
            if self.learner_type==2:
                pred=result
            return pred