import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import AdamOptimizer

from sklearn.model_selection import train_test_split
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch import nn
import torch
import math
import time



num_qubits = 4
num_layers = 8

#dev = qml.device("default.qubit", wires=num_qubits)
dev = qml.device("lightning.gpu", wires=num_qubits)

# quantum circuit functions
def statepreparation(x):
    #qml.BasisEmbedding(x, wires=range(0, num_qubits))
    qml.AngleEmbedding(x, wires=range(num_qubits))

# update to dynamic layer
def layer(W):
    for i in range(num_qubits):
        qml.Rot(W[i,0], W[i,1], W[i,2], wires=i)

    for i in range(num_qubits):
        qml.CNOT(wires=[i, (i+1)%num_qubits])


# Config interface with Torch
#@qml.qnode(dev, interface="autograd")
@qml.qnode(dev, interface="torch", diff_method="adjoint")

def circuit(weights, x):

    statepreparation(x)

    for W in weights:
        layer(W)

    return qml.expval(qml.PauliZ(0))


# Define class of VQC
class VQC(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(0.01*torch.randn((num_layers, num_qubits, 3), dtype=torch.float64))

    def forward(self, x):
        preds = [circuit(self.weights, xi.to(torch.float64)) for xi in x]
        return torch.stack(preds)




# preparaing data
df_train = pd.read_csv('/home/hpcstudent/lan_workspace/VQC_quantum/VQC/dataset/train.csv')

df_train['Pclass'] = df_train['Pclass'].astype(str)

df_train = pd.concat([df_train, pd.get_dummies(df_train[['Pclass', 'Sex', 'Embarked']])], axis=1)

# I will fill missings with the median
df_train['Age'] = df_train['Age'].fillna(df_train['Age'].median())

df_train['is_child'] = df_train['Age'].map(lambda x: 1 if x < 12 else 0)
cols_model = ['is_child', 'Pclass_1', 'Pclass_2', 'Sex_female']

X_train, X_test, y_train, y_test = train_test_split(df_train[cols_model], df_train['Survived'], test_size=0.10, random_state=42, stratify=df_train['Survived'])

X_train = np.array(X_train.values, requires_grad=False)
Y_train = np.array(y_train.values * 2 - np.ones(len(y_train)), requires_grad=False)
# Create tensor dataset
#X_train = torch.Tensor(X_train)
X_train = X_train.astype(float) #La: X_train is an object
X_train = torch.from_numpy(X_train).float()
Y_train = torch.from_numpy(Y_train).long()
dataset = TensorDataset(X_train, Y_train)



# Setting model and optimizer
device = torch.device("cuda")
model = VQC().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# prepare minibacth and setup
file_path = "/home/hpcstudent/lan_workspace/VQC_quantum/VQC/results/qubit4/training_result_vqc_bs1024.csv"
batch_size = 1024
epochs = 20
def iterate_minibatch(x, y, batch_size, shuffle=True):
    n = x.shape[0]
    idx = np.arange(n)
    if shuffle:
        np.random.shuffle(idx)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        sel = idx[start:end]
        yield x[sel], y[sel]

start_time = time.time()
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    correct = 0
    total = 0

    for xb_cpu, yb_gpu in iterate_minibatch(X_train, Y_train, batch_size, shuffle=True):
        optimizer.zero_grad()
        xb = xb_cpu.to(device)
        yb = yb_gpu.to(device)
        preds_list = []

        for i in range(xb.shape[0]):
            xi = xb[i]
            pred = model(xi.unsqueeze(0))
            preds_list.append(pred)

        preds = torch.cat(preds_list)
        loss = criterion(preds, yb.double())
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()*yb.shape[0]
        pred_labels = torch.sign(preds.detach())
        correct += (pred_labels == yb.double()).sum().item()
        total += yb.shape[0]



    avg_loss = epoch_loss/total
    acc = correct/total

    print(f"Epoch {epoch+1} | Loss: {avg_loss:.6f} | Accuracy: {acc:.4f}")
    with open(file_path, 'a') as f:
        f.write(f"{epoch}, {avg_loss}, {acc}\n");
    print("Done writing to file\n")

end_time = time.time()
elapsed = end_time - start_time
training_speed = elapsed/epochs
print(f"Training speed is {training_speed}")

'''
X_test = np.array(X_test.values, requires_grad=False)
Y_test = np.array(y_test.values * 2 - np.ones(len(y_test)), requires_grad=False)

predictions = [np.sign(variational_classifier(weights, bias, x)) for x in X_test]

accuracy_score(Y_test, predictions)
precision_score(Y_test, predictions)
recall_score(Y_test, predictions)
f1_score(Y_test, predictions, average='macro')
'''
