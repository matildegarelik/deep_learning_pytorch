import torch
import torchvision
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import random_split
from torch.utils.data import DataLoader

dataset= MNIST (root='data/')
test_dataset = MNIST (root='data/', train=False)
train_ds,val_ds=random_split (dataset, [50000,10000])


#image, label = dataset[0]
#plt.imshow(image, cmap='gray')
#plt.show()
#print('Label: ', label)


dataset = MNIST (root='data/', train = True, transform=transforms.ToTensor())

#img_tensor, label = dataset[0]
#print(img_tensor.shape, label)
#print (img_tensor[:,10:15,10:15])
# pylint: disable=E1101
print (torch.max(img_tensor), torch.min(img_tensor))
# pylint: enable=E1101
#plt.imshow(img_tensor[0,10:15,10:15], cmap='gray')
#plt.show()

batch_size= 128
train_loader = DataLoader (train_ds, batch_size, shuffle=True)
val_loader = DataLoader (val_ds, batch_size, shuffle=True)




