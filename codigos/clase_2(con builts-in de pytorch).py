import numpy as np
import torch
import torch.nn as nn


# Input (temp, rainfall, humidity)
inputs = np.array([[73, 67, 43], [91, 88, 64], [87, 134, 58], 
                   [102, 43, 37], [69, 96, 70], [73, 67, 43], 
                   [91, 88, 64], [87, 134, 58], [102, 43, 37], 
                   [69, 96, 70], [73, 67, 43], [91, 88, 64], 
                   [87, 134, 58], [102, 43, 37], [69, 96, 70]], 
                  dtype='float32')

# Targets (apples, oranges)
targets = np.array([[56, 70], [81, 101], [119, 133], 
                    [22, 37], [103, 119], [56, 70], 
                    [81, 101], [119, 133], [22, 37], 
                    [103, 119], [56, 70], [81, 101], 
                    [119, 133], [22, 37], [103, 119]], 
                   dtype='float32')


# pylint: disable=E1101
inputs= torch.from_numpy(inputs)
targets = torch.from_numpy(targets)
# pylint: enable=E1101

from torch.utils.data import TensorDataset

train_ds= TensorDataset(inputs, targets)

from torch.utils.data import DataLoader

batch_size = 5
train_dl= DataLoader(train_ds, batch_size, shuffle=True)



model= nn.Linear(3,2) # 3 values for inputs and 2 for targets
#print(model.weight)
#print (model.bias)

preds = model (inputs)

import torch.nn.functional as F

#Definicion de la funcion loss
loss_fn= F.mse_loss
loss= loss_fn(model(inputs),targets)

#Definicion de optimizador

opt=torch.optim.SGD(model.parameters(), lr=1e-5)

def fit (num_epochs, model, loss_fn, opt, train_dl):
    for epoch in range (num_epochs):
        for xb,yb in train_dl:
            pred = model (xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            opt.zero_grad()
        if (epoch+1) % 10 == 0:
             print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))


fit(100, model, loss_fn, opt,train_dl)




