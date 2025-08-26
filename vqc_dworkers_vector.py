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
num_vlanes = 16

try:
    dev = qml.device("lightning.gpu", wires=num_qubits, shots=None)
    print("Using Pennylane device: GPU\n")
except Exception as e:
    dev = qml.device("default.qubit", wires=num_qubits, shots=None)
    print("Falling back to default device: CPU\n")

# quantum circuit functions for VEC
def statepreparation(x):
    #qml.AngleEmbedding(x, wires=range(num_qubits))
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

def circuit(weights, x): #Vectorized QNode

    statepreparation(x)

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
        if x_batch.dtype != torch.float64:
            x_batch = x_batch.double()

        x_vec_full = x_batch.unsqueeze(1).repeat(1, self.vec_lanes, 1).reshape(-1, self.num_qubits)
        weights_vec = self.weights
        y_vec_full = circuit(weights_vec, x_vec_full)
        y_vec_full = y_vec_full.reshape(B, self.vec_lanes)
        preds = y_vec_full.mean(dim = 1).unsqueeze(1)
        #preds = [circuit(self.weights, xi) for xi in x]
        return preds




# preparaing data
file_path = '/home/hpcstudent/lan_workspace/VQC_quantum/VQC/results/qubit4/training_result_vqc_dw32_vec16.csv'
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
Y_train = torch.from_numpy(Y_train).double().unsqueeze(1)
dataset = TensorDataset(X_train, Y_train)



# Setting model and optimizer
device = torch.device("cuda")
model = VQC(num_layers=num_layers, num_qubits=num_qubits, vec_lanes=num_vlanes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.MSELoss()


# Create DataLoader
train_loader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=True,
        num_workers=32,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True
        )

epochs = 20

start_time = time.time()
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    correct = 0
    total = 0

    for xb, yb in train_loader:
        xb_cpu, yb_gpu = xb.cpu(), yb.to(device)
        optimizer.zero_grad()
        preds = model(xb_cpu)
        loss = criterion(preds, yb_gpu)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            epoch_loss += loss.item()*xb_cpu.size(0)
            pred_labels = torch.sign(preds).squeeze(1)
            true_labels = yb_gpu.squeeze(1)
            correct += (pred_labels == true_labels).sum().item()
            total += xb_cpu.size(0)

    avg_loss = epoch_loss/total
    acc = correct/total

    print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Accuracy: {acc:.4f}")
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
