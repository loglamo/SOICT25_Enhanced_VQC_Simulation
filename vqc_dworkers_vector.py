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
import pynvml


num_qubits = 4
num_layers = 4
num_vlanes = 64

try:
    dev = qml.device("lightning.gpu", wires=num_qubits, shots=None)
    print("Using Pennylane device: GPU\n")
except Exception as e:
    dev = qml.device("default.qubit", wires=num_qubits, shots=None)
    print("Falling back to default device: CPU\n")

# quantum circuit functions for VEC: Angleenbedding supports batched inputs
def statepreparation(x):
    #qml.AngleEmbedding(x, wires=range(num_qubits))
    print("Batched lanes execution\n")
    qml.AngleEmbedding(x, wires=range(num_qubits), rotation='X')

# dynamic layers
def layer(W):
    for i in range(num_qubits):
        qml.Rot(W[i,0], W[i,1], W[i,2], wires=i)
    for i in range(num_qubits):
        qml.CNOT(wires=[i, (i+1)%num_qubits])


# Config interface with Torch
#@qml.qnode(dev, interface="autograd")
@qml.qnode(dev, interface="torch", diff_method="adjoint")

def circuit(weights, x_batch_lanes): #Vectorized QNode

    statepreparation(x_batch_lanes)

    for W in weights:
        layer(W)

    return qml.expval(qml.PauliZ(0))


# Define class of VQC with vectorized lanes
class VQC(nn.Module):
    def __init__(self, num_layers, num_qubits, vec_lanes):
        super().__init__()
        self.num_layers = num_layers
        self.num_qubits = num_qubits
        self.vec_lanes = vec_lanes
        self.weights = nn.Parameter(0.01*torch.randn((num_layers, num_qubits, 3), dtype=torch.float64))


    def forward(self, x_batch):
        B = x_batch.shape[0]
        x_vec_full = x_batch.unsqueeze(1).repeat(1, self.vec_lanes, 1).reshape(-1, self.num_qubits)
        y_vec_full = circuit(self.weights, x_vec_full)
        preds = y_vec_full.reshape(B, self.vec_lanes).mean(dim = 1, keepdim=True)
        return preds




# preparaing data
file_path = '/home/hpcstudent/lan_workspace/VQC_quantum/VQC/results/qubit4/training_result_vqc_dw8_vec32.csv'
df_train = pd.read_csv('/home/hpcstudent/lan_workspace/VQC_quantum/VQC/dataset/train.csv')

df_train['Pclass'] = df_train['Pclass'].astype(str)

df_train = pd.concat([df_train, pd.get_dummies(df_train[['Pclass', 'Sex', 'Embarked']])], axis=1)

# I will fill missings with the median
df_train['Age'] = df_train['Age'].fillna(df_train['Age'].median())

df_train['is_child'] = df_train['Age'].map(lambda x: 1 if x < 12 else 0)
cols_model = ['is_child', 'Pclass_1', 'Pclass_2', 'Sex_female']

X_train, X_test, Y_train, Y_test = train_test_split(df_train[cols_model], df_train['Survived'], test_size=0.10, random_state=42, stratify=df_train['Survived'])

"""
X = torch.tensor(df_train[cols_model].to_numpy(dtype=float), dtype=torch.float64)
Y = torch.tensor(df_train['Survived'].to_numpy()*2-1, dtype=torch.float64)
dataset = TensorDataset(X, Y)
"""
#X_train = np.array(X_train.values, requires_grad=False)
#Y_train = np.array(y_train.values * 2 - np.ones(len(y_train)), requires_grad=False)
# Create tensor dataset

X_train = X_train.to_numpy().astype(float) #La: X_train is an object
X_train = torch.from_numpy(X_train).float()
Y_train = Y_train.to_numpy().astype(float)
Y_train = torch.from_numpy(Y_train).double().unsqueeze(1)
dataset = TensorDataset(X_train, Y_train)

"""
print(f"The number of samples is: {len(dataset)}")
print(f"Shape of X_train is: {dataset[0][0].shape}")
print(f"Shape of Y_train is: {dataset[0][1].shape}")
"""

# Setting model and optimizer
device = torch.device("cuda")
model = VQC(num_layers=num_layers, num_qubits=num_qubits, vec_lanes=num_vlanes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()


# Create DataLoader
train_loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True
        )

epochs = 20


pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)

start_time = time.time()
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    correct = 0
    total = 0
    print(f"Start training in {epoch+1}")
    for xb, yb in train_loader:
        xb_cpu, yb_gpu = xb.cpu(), yb.to(device)
        optimizer.zero_grad()
        preds = model(xb_cpu)
        loss = criterion(preds, yb_gpu)
        loss.backward()
        optimizer.step()

        
        epoch_loss += loss.item()*xb_cpu.size(0)
        pred_signs = torch.sign(preds.detach())
        correct += (pred_signs == yb_gpu).sum().item()
        total += len(yb_gpu)

    avg_loss = epoch_loss/total
    acc = correct/total

    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
    gpu_util = util.gpu

    print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Accuracy: {acc:.4f} | GPU_utilization: {gpu_util}")
    with open(file_path, 'a') as f:
        f.write(f"{epoch}, {avg_loss}, {acc}, {gpu_util}\n");
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
