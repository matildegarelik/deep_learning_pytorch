import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split

def accuracy(outputs, labels):
    # pylint: disable=E1101
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))
    # pylint: enable=E1101

class MnistModel(nn.Module):
    """Feedfoward neural network with 1 hidden layer"""
    def __init__(self, in_size, hidden_size, out_size):
        super().__init__()
        # hidden layer
        self.linear1 = nn.Linear(in_size, hidden_size)
        # output layer
        self.linear2 = nn.Linear(hidden_size, out_size)
        
    def forward(self, xb):
        # Flatten the image tensors
        xb = xb.view(xb.size(0), -1)
        # Get intermediate outputs using hidden layer
        out = self.linear1(xb)
        # Apply activation function
        out = F.relu(out)
        # Get predictions using output layer
        out = self.linear2(out)
        return out
    
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        # pylint: disable=E1101
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
        # pylint: enable=E1101
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))


test_dataset = MNIST (root="C:\\Users\\usuario\\Documents\\PROGRMACIÓN\\python\\deep_learning_con_pytorch\\data", train=False, transform=ToTensor())
input_size= 784
hidden_size =32
num_classes = 10

model = MnistModel (input_size,hidden_size, out_size=num_classes)

def evaluate (model, val_loader):
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

model.load_state_dict(torch.load("C:\\Users\\usuario\\Documents\\PROGRMACIÓN\\python\\deep_learning_con_pytorch\\mnist-logistic-clase4.pth"))
#test_loader = DataLoader(train_ds, batch_size=256)
#result=evaluate(model, test_loader)
#print(result)


def predict_image (img, model):
    xb = img.unsqueeze(0)
    yb = model (xb)
    _, preds = torch.max(yb, dim=1)
    return preds[0].item()

print("Ponga un número al azar")
num = int(input())


img, label = test_dataset[num]
plt.imshow(img[0], cmap='gray')
#print('Label:', label)
print('Para mi (yo el programa) ese numero es un ', predict_image(img, model))
plt.show()