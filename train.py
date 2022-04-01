# -*- coding: utf-8 -*-
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
#import xlwt
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

       

class HydrogelNet(nn.Module):
    
    def __init__(self, n_features, n_hidden, n_sequence, n_layers, n_classes):
        super(HydrogelNet, self).__init__()
        
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.n_sequence = n_sequence
        self.n_layers = n_layers
        self.n_classes = n_classes
        self.conv1d = nn.Conv1d(in_channels = 2,out_channels = 2,kernel_size = 1)
        self.linear_1 = nn.Linear(in_features=n_hidden, out_features=128)
        self.dropout_1 = nn.Dropout(p=0.2)        
        self.linear_2 = nn.Linear(in_features=128, out_features=n_classes)        
        
    
    def forward(self, x):
        out = self.conv1d(x)       
        out = out.view(-1,358)
        out = self.linear_1(out)
        out = self.dropout_1(out)
        out = F.relu(out)
        out = self.linear_2(out)
        out = F.relu(out)
        
        return out
    
    
def train_model(model, train_dataloader, n_epochs):
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)#, weight_decay = 0.0001)
      
    for epoch in range(n_epochs):
        for i, (X_train, y_train) in enumerate(train_dataloader):
            
            y_hat = model(X_train)            
            loss = loss_fn(y_hat.float(), y_train)
 
            if i == 0 and (epoch+1)%20==0:
               print(f'Epoch {epoch+1} train loss: {loss.item()}')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
    return model



class SensorDataset(Dataset):
    
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]
    
    def __len__(self):
        return self.X.shape[0]

def evaluateModel(prediction, y):
    prediction = torch.argmax(prediction, dim=1)

    good = 0
    for i in range(len(y)):
        if (prediction[i] == y[i]):
            good = good +1
    return (good/len(y)) * 100.0    
    
n_features=179 
n_sequence=2 
n_hidden=358 
n_layers=1 
n_classes=5
n_epochs = 2400
n_batch_size = 179

df = pd.read_csv("./dataset.csv", header=None)
data = df.values
np.random.seed(1234)
np.random.shuffle(data)
sc = MinMaxScaler(feature_range = (0, 1))
data[:,:-1] = sc.fit_transform(data[:,:-1])
kf = KFold(n_splits=10)
train_acc_sum ,val_acc_sum = 0,0

for train_index, test_index in kf.split(data): 
 
    train_dataset = data[train_index]   
    test_dataset = data[test_index]

    X_train, y_train = train_dataset[:,:-1].reshape(-1,2,179), train_dataset[:,-1]
    X_test, y_test = test_dataset[:,:-1].reshape(-1,2,179), test_dataset[:,-1] 
    X_train = torch.from_numpy(X_train).float().to(device)
    y_train = torch.from_numpy(y_train).long().to(device)
    X_test = torch.from_numpy(X_test).float().to(device)
    y_test = torch.from_numpy(y_test).long().to(device)

    train_dataset = SensorDataset(X_train, y_train)
    model = HydrogelNet(n_features, n_hidden, n_sequence, n_layers, n_classes).to(device)
    train_dataloader = DataLoader(dataset = train_dataset, batch_size=n_batch_size, shuffle=False)
    model = train_model(model, train_dataloader, n_epochs = n_epochs)
    
    with torch.no_grad():
        y_hat_train = model(X_train) 
        train_acc = evaluateModel(y_hat_train, y_train)
        print("Train Accuracy in the fold", train_acc)
    
        y_hat_test = model(X_test)
        val_acc = evaluateModel(y_hat_test, y_test)
        print("Validation Accuracy in the fold ", val_acc)
        train_acc_sum += train_acc
        val_acc_sum += val_acc

print('train_acc_sum:%.4f\n'%(train_acc_sum/10),'valid_acc_sum:%.4f'%(val_acc_sum/10))





