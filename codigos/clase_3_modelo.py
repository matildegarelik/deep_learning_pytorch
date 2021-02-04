import torch
import torchvision
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

#Definición de sets de datos
dataset= MNIST (root='data/')
dataset = MNIST (root='data/', train = True, transform=transforms.ToTensor())
test_dataset = MNIST (root='data/', train=False, transform=transforms.ToTensor())
train_ds,val_ds=random_split (dataset, [50000,10000])



#Separacion en batchs para los sets de datos de entrenamiento y validacion
batch_size= 128
train_loader = DataLoader (train_ds, batch_size, shuffle=True)
val_loader = DataLoader (val_ds, batch_size, shuffle=True)

#Modelo de regresión logistica
input_size=28*28
num_classes=10

def accuracy(outputs,labels):
    # pylint: disable=E1101
    _, preds= torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds==labels).item()/len(preds))
    # pylint: enable=E1101


class MnistModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear=nn.Linear(input_size, num_classes)
    def forward (self, xb):
        xb= xb.reshape(-1,784)
        out = self.linear(xb)
        return out
    def training_step(self,batch):
        images, labels = batch
        out=self(images)  #Generate predictions
        loss=F.cross_entropy(out, labels)  #Calculate loss
        return loss
    def validation_step(self, batch):
        images, labels = batch
        out=self(images)  #Generate predictions
        loss=F.cross_entropy(out, labels)  #Calculate loss
        acc=accuracy(out, labels) # Calculate accuracy
        return {'val_loss': loss , 'val_acc': acc}
    def validation_epoch_end(self, outputs):
        batch_losses=[x['val_loss'] for x in outputs]
        # pylint: disable=E1101
        epoch_loss=torch.stack(batch_losses).mean() #Combine losses
        batch_accs= [x['val_acc']for x in outputs]
        epoch_acc=torch.stack(batch_accs).mean()
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
        # pylint: enable=E1101
    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss:{:.4f}, val_acc:{:.4f}".format(epoch, result['val_loss'], result['val_acc']))
    

model= MnistModel()

def evaluate (model, val_loader):
    outputs=[model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        # Training Phase 
        for batch in train_loader:
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(model, val_loader)
        model.epoch_end(epoch, result)
        history.append(result)
    return history

result0 = evaluate(model, val_loader)

#history1 = fit(5, 0.001, model, train_loader, val_loader)
#history2 = fit(5, 0.001, model, train_loader, val_loader)
#history3 = fit(5, 0.001, model, train_loader, val_loader)
#history4 = fit(5, 0.001, model, train_loader, val_loader)
#history5 = fit(5, 0.001, model, train_loader, val_loader)

#history = [result0] + history1 + history2 + history3 + history4
#accuracies = [result['val_acc'] for result in history]
#plt.plot(accuracies, '-x')
#plt.xlabel('epoch')
#plt.ylabel('accuracy')
#plt.title('Accuracy vs. N° of epochs')
#plt.show()
 

def predict_image (img, model):
    xb = img.unsqueeze(0)
    yb = model (xb)
    _, preds = torch.max(yb, dim=1)
    return preds[0].item()

#img, label = test_dataset[0]
#plt.imshow(img[0], cmap='gray')
#print('Label:', label, 'Predicted: ', predict_image(img, model))
#plt.show()


test_loader = DataLoader(test_dataset, batch_size=256)
result=evaluate(model, test_loader)
print(result)

torch.save(model.state_dict(), 'mnist-logistic.pth')


#Esta parte es para comprobar que se guardaron mis  matrices de weights and biases

model2=MnistModel()
model2.load_state_dict(torch.load('mnist-logistic.pth'))
test_loader = DataLoader(test_dataset, batch_size=256)
result2=evaluate(model2, test_loader)
print(result2)
print(model.parameters())