import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
import os


dataset = CIFAR10(root='data/', transform = ToTensor())
test_dataset = CIFAR10 (root='data/', train= False, transform = ToTensor())
dataset_size = 50000
test_dataset_size = 10000
classes = dataset.classes
num_classes = 10

#img, label = dataset [0]
#plt.imshow(img.permute(1,2,0))
#print('Label (numeric):', label)
#print('Label (textual):', classes[label])
#plt.show()

torch.manual_seed(43)
val_size = 5000
train_size = len(dataset) - val_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])
batch_size=128
train_loader = DataLoader(train_ds, batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size*2)
test_loader = DataLoader(test_dataset, batch_size*2)

for images, _ in train_loader:
    #print('images.shape:', images.shape)
    #plt.figure(figsize=(16,8))
    #plt.axis('off')
    #plt.imshow(make_grid(images, nrow=16).permute((1, 2, 0)))
    #plt.show()
    img_shape= images.shape
    break

def accuracy (outputs, labels):
    _, preds = torch.max (outputs, dim=1)
    return torch.tensor(torch.sum (preds==labels).item()/len (preds))

class ImageClassificationBase (nn.Module):
    def training_step (self, batch):
        images, labels = batch
        out = self (images)
        loss = F.cross_entropy(out, labels)
        return loss
    def validation_step (self, batch):
        images, labels = batch
        out = self (images)
        loss = F.cross_entropy(out, labels)
        acc = accuracy (out, labels)
        return {'val_loss': loss.detach(), 'val_acc': acc}
    def validation_epoch_end (self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))


def evaluate (model, val_loader):
    outputs = [model.validation_step (batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)
def fit (epochs, lr, model, train_loader, val_loader, opt_func = torch.optim.SGD):
    history = []
    optimizer = opt_func (model.parameters(), lr)
    for epoch in range (epochs):
        #Training phase
        for batch in train_loader:
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        #Fase de validacion
        result= evaluate (model, val_loader)
        model.epoch_end(epoch, result)
        history.append (result)
    return history

def plot_losses(history):
    losses = [x['val_loss'] for x in history]
    plt.plot(losses, '-x')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Loss vs. No. of epochs')
    plt.show()
def plot_accuracies(history):
    accuracies = [x['val_acc'] for x in history]
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs. No. of epochs')
    plt.show()



class CIFAR10Model (ImageClassificationBase):
    def __init__(self, in_size, hid_size1, hid_size2, hid_size3, out_size):
        super().__init__()
        self.linear1 = nn.Linear (in_size, hid_size1)
        self.linear2 = nn.Linear (hid_size1, hid_size2)
        self.linear3 = nn.Linear (hid_size2, hid_size3)
        self.linear4 = nn.Linear (hid_size3, out_size)
    def forward (self, xb):
        # Flatten images into vectors
        xb = xb.view(xb.size(0), -1)
        # Apply layers & activation functions
        out = self.linear1(xb)
        out = F.relu(out)
        out = self.linear2(out)
        out = F.relu(out)
        out = self.linear3(out)
        out = F.relu(out)
        #Obtener predicciones usando la layer de salida
        out = self.linear4(out)
        return out

input_size = 3*32*32
hidden_size1 = 64
hidden_size2 = 32
hidden_size3 = 16
output_size = 10
model = CIFAR10Model (input_size, hidden_size1, hidden_size2, hidden_size3, output_size)

history = [evaluate (model, val_loader)]
print (history)

e1, l1 = 5, 0.01
history += fit(e1, l1, model, train_loader, val_loader)
e2, l2 = 2, 0.003
history += fit(e2, l2, model, train_loader, val_loader)
e3, l3 = 4, 0.0001
history += fit(e3, l3, model, train_loader, val_loader)
e4, l4 = 3, 0.000001
history += fit(e4, l4, model, train_loader, val_loader)

#plot_losses (history)
#plot_accuracies (history)

print (evaluate (model, test_loader))
torch.save(model.state_dict(), 'cifar10-feedforward.pth')

archivo_de_texto = open ('C:\\Users\\usuario\\Documents\\PROGRMACIÓN\\python\\deep_learning_con_pytorch\\codigos\\tareas\\cifar 10\\datos.txt', 'a')
archivo_de_texto.write ('\n TERCER ARQUITECTURA \n')
archivo_de_texto.write ('cantidad de hidden layers: 3 \n tamaños: ' + str(hidden_size1) + ' y ' + str(hidden_size2) + ' y ' + str(hidden_size3) + '\n')
archivo_de_texto.write ('fits de history: ' + '\n 1) epochs = ' + str(e1) + ' ; learning rate = ' + str(l1))
archivo_de_texto.write ('\n 2) epochs = ' + str(e2) + ' ; learning rate = ' + str(l2) +'\n 3) epochs = ' + str(e3) + ' ; learning rate = ' + str(l3))
archivo_de_texto.write ('\n 4) epochs = ' + str(e4) + ' ; learning rate = ' + str(l4))
archivo_de_texto.write('\n resultados: ' + str(evaluate(model, test_loader)) + '\n')
archivo_de_texto.close()


