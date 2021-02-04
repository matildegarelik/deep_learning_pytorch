import numpy as np
import torch

# las entradas son: (temperatura, lluvia, humedad)
inputs =np.array(([73,67,43],
                [91,88,64],
                [87,134,58],
                [102,43,37],
                [69,96,70]), dtype='float32')

# los targets, es decir, lo que queremos predecir, son: (manzanas, naranjas)
targets =np.array(([56,70],
                [81,101],
                [119,133],
                [22,37],
                [103,119]), dtype='float32')

# conversion de inputs y targets a tensores de torch
# pylint: disable=E1101
inputs= torch.from_numpy(inputs)
targets = torch.from_numpy(targets)
# pylint: enable=E1101

# weights (matrices de 2x3) and biases (vectores)
# pylint: disable=E1101
w = torch.randn(2,3, requires_grad=True)
b = torch.randn(2, requires_grad=True)
# pylint: enable=E1101
print('Winicial: ', w)
print('Binicial: ', b)

# funcion modelo
def model(x) :
    return x @ w.t() + b # usamos la matriz transpuesta de w para que se pueda multiplicar correctamente


# Funcion de costo: calcula el MSE (error minimo cuadrado)

def mse(t1,t2):
    diff = t1- t2
    # pylint: disable=E1101
    return torch.sum((diff*diff)/ diff.numel()) 
    # pylint: enable=E1101
    # este valor es el cuadrado de cuánto difiere cada elemento de la prediccion del valor real

# Generar predicciones
preds = model (inputs)


# Computar el error y gradientes
loss = mse (preds, targets)
print ('MSE inicial: ', loss)

loss.backward()
#print(w.grad)
#print(b.grad)

#Ajustar weights y resetear a cero los gradientes
with torch.no_grad():
    w-= w.grad * 1e-5
    b-= b.grad * 1e-5
    w.grad.zero_()
    b.grad.zero_()


#Calculo del nuevo costo
preds=model(inputs)
loss = mse (preds, targets)


# Entrenamiento de 100 epochs
for i in range (10000):
    preds = model (inputs)
    loss = mse (preds, targets)
    loss.backward()
    with torch.no_grad():
        w-= w.grad * 1e-5
        b-= b.grad * 1e-5
        w.grad.zero_()
        b.grad.zero_()

#Calculo del nuevo costo
preds=model(inputs)
loss = mse (preds, targets)
print ('MSE final: ', loss)