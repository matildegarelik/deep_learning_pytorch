import torch
import torchvision
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision.datasets.utils import download_url
from torch.utils.data import DataLoader, TensorDataset, random_split
from math import sqrt

dataset_url = "https://hub.jovian.ml/wp-content/uploads/2020/05/insurance.csv"
data_filename = r"C:\\Users\\usuario\Documents\\PROGRMACIÓN\\archivosCSV\\insurance.csv"
#download_url(dataset_url,'.')
dataframe = pd.read_csv(data_filename,sep=',')

my_name = "matilde"


def customize_dataset(dataframe_raw, rand_str): #Me pone lindo para ver los datos
    dataframe = dataframe_raw.copy(deep=True)
    # drop some rows
    dataframe = dataframe.sample(int(0.95*len(dataframe)), random_state=int(ord(rand_str[0])))
    # scale input
    dataframe.bmi = dataframe.bmi * ord(rand_str[1])/100.
    # scale target
    dataframe.charges = dataframe.charges * ord(rand_str[2])/100.
    # drop column
    if ord(rand_str[3]) % 2 == 1:
        dataframe = dataframe.drop(['region'], axis=1)
    return dataframe


#dataframe=customize_dataset(dataframe_raw, my_name)
num_rows=1338
num_columns = 7
input_cols = ['age', 'sex', 'bmi', 'children', 'smoker','region','charges']
categorical_cols=['sex','smoker','region']
output_cols = ['charges']

def dataframe_to_arrays(dataframe):
    # Make a copy of the original dataframe
    dataframe1 = dataframe.copy(deep=True)
    # Convert non-numeric categorical columns to numbers
    for col in categorical_cols:
        dataframe1[col] = dataframe1[col].astype('category').cat.codes
    # Extract input & outupts as numpy arrays
    inputs_array = dataframe1[input_cols].to_numpy()
    targets_array = dataframe1[output_cols].to_numpy()
    return inputs_array, targets_array

inputs_array, targets_arrays = dataframe_to_arrays(dataframe)
# pylint: disable=E1101
inputs = torch.from_numpy(inputs_array)
targets = torch.from_numpy(targets_arrays)
inputs, targets =inputs.float(), targets.float()
# pylint: enable=E1101
targets = targets.squeeze (dim=1)

dataset = TensorDataset (inputs, targets)
val_percent = 0.15
val_size = int(num_rows * val_percent)
train_size = int(num_rows-val_size)
train_ds, val_ds = random_split (dataset, [train_size, val_size])


batch_size = 32
train_loader =DataLoader(train_ds,batch_size, shuffle = True)
val_loader = DataLoader(val_ds, batch_size, shuffle =True)


input_size = len(input_cols)
output_size = len (output_cols)

def accuracy(out,targets):
  return sqrt((out.item()-targets.item())**2)/len(target.item())


class InsuranceModel (nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
    def forward(self ,xb):
        out = self.linear(xb)
        return out
    def training_step (self, batch):
        inputs, targets = batch
        out = self (inputs)
        loss=F.kl_div(out, targets)
        return loss
    def validation_step (self, batch):
        inputs, targets = batch
        out = self (inputs)
        loss= F.kl_div(out, targets)
        acc = accuracy (out, targets)
        return {'val_loss': loss.detach(), 'val_acc': acc.detach()}
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        # pylint: disable=E1101
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        # pylint: enable=E1101
        batch_accs = [x['val_acc'] for x in outputs]
        # pylint: disable=E1101
        epoch_acc = torch.stack(batch_accs).mean()   # Combine losses
        # pylint: enable=E1101
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    def epoch_end(self, epoch, result, num_epochs):
        # Print result every 20th epoch
        if (epoch+1) % 20 == 0 or epoch == num_epochs-1:
            print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch+1, result['val_loss'], result ['val_acc']))

model = InsuranceModel()

def evaluate (model, val_loader):
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)
def fit (epochs, lr, model, train_loader, val_loader, opt_func = torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range (epochs):
        for batch in train_loader:
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        result = evaluate (model, val_loader)
        model.epoch_end(epoch, result, epochs)
        history.append (result)
    return history

result0 = evaluate(model, val_loader)
print('result0: ', result0)

epochs= 5
lr = 0.001
#history1 = fit (epochs, lr, model, train_loader, val_loader)

epochs= 35
lr = 0.00001
#history2 = fit (epochs, lr, model, train_loader, val_loader)

model = InsuranceModel()

epochs= 500
lr = 0.0001
history3 = fit (epochs, lr, model, train_loader, val_loader)

epochs= 7000
lr = 0.001
#history4 = fit (epochs, lr, model, train_loader, val_loader)

epochs= 800
lr = 1e-6
#history5 = fit (epochs, lr, model, train_loader, val_loader)

def predict_single(input, target, model):
    inputs = input.unsqueeze(0)
    predictions = model(inputs)
    prediction = predictions[0].detach()
    print("Input:", input)
    print("Target:", target)
    print("Prediction:", prediction)

input, target = val_ds[10]
predict_single(input, target, model)




