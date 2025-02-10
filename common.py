import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from ase.formula import Formula
from tqdm import tqdm, trange

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score, mean_absolute_error

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch import optim
from torch.optim.lr_scheduler import ExponentialLR, StepLR

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
    )

elements = {'Fe': 0, 'Ni': 1, 'Co': 2, 'Cr': 3, 'Mn': 4, 'Pd': 5}
electronegs = [1.83, 1.91, 1.88, 1.66, 1.55, 2.20]
radii = [1.277, 1.246, 1.250, 1.285, 1.292, 1.376]
valences = [8, 10, 9, 6, 7, 10]

def composition(formula) -> list:
    count = Formula(formula).count()
    tot = sum(count.values())
    composition = [0. for i in range(6)]
    for element in count:
        composition[elements[element]] = count[element] / tot
    
    avg_electroneg = np.inner(composition, electronegs)
    avg_radius = np.inner(composition, radii)
    avg_valence = np.inner(composition, valences)
    delta_radius = np.sqrt(np.inner(composition, (1 - radii / avg_radius) ** 2))
    delta_electroneg = np.sqrt(np.inner(composition, (electronegs - avg_electroneg) ** 2))
    S = 0
    R = 8.3145
    for c in composition:
        if c != 0:
            S += -R * c * np.log(c)
    return composition + [S, delta_radius, delta_electroneg, avg_valence]

def features(df, phase):
    formulas_phase = [structure.split('_')[0] for structure in df['structures'].to_numpy() if structure.split('_')[1][-3:] == phase]
    X_phase = []
    for formula in formulas_phase:
        X_s = composition(formula)
        X_phase.append(X_s)
    return np.array(X_phase)

def values(df, property, phase):
    df_structure, df_property = df['structures'].to_numpy(), df[property].to_numpy()
    return np.array([df_property[i] for (i, structure) in enumerate(df_structure) if structure.split('_')[1][-3:] == phase]) 

def preprocessing(X_train, X_test, y):
    scaler = StandardScaler()
    if scaler:
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        if X_test.any():
            X_test = scaler.transform(X_test)
      
    y_scaler = StandardScaler()
    if y_scaler:
        y_scaler.fit(y.reshape(-1, 1))
        y_trans = y_scaler.transform(y.reshape(-1, 1)).flatten()
    else:
        y_trans = y
    return (torch.tensor(X_train, dtype = torch.float, device = device), torch.tensor(X_test, dtype = torch.float, device = device), 
            torch.tensor(y_trans, dtype = torch.float, device = device), scaler, y_scaler)


class HEA_Dataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(y, pred)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def test(X, y, model, loss_fn, y_scaler, error = False, plot = False):
    model.eval()
    with torch.no_grad():
        pred = model(X)
        y_trans = y_scaler.transform(y.reshape(-1, 1)).flatten() if y_scaler else y
        y_trans = torch.tensor(y_trans, dtype = torch.float, device = device)
        test_loss = loss_fn(y_trans, pred).item()
    
    if error:
        pred = pred.numpy(force = True)
        if y_scaler:
            pred = y_scaler.inverse_transform(pred.reshape(-1, 1)).flatten()
        correct = mean_absolute_error(y, pred), r2_score(y, pred)
        print(f"R2: {correct[1]}, loss: {test_loss:>8f}")

        if plot:
            y_range = [y.min(), y.max()]
            fig, ax = plt.subplots(figsize = (6,6))
            ax.set_aspect(1)
            plt.plot(y_range, y_range, 'k--')
            plt.scatter(y, pred)
            plt.show()        
        return correct
    
    return test_loss

# Define models
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(6, 58)
        self.conv = nn.Sequential(
            nn.Conv2d(1, 1, 5, padding = 'same'),
        )
        self.linear_relu_stack = nn.Sequential(
            
            nn.Linear(64, 64),            
            nn.ReLU(),

            nn.Linear(64, 64),
            nn.ReLU(),      

            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        embd = self.embedding(torch.arange(6, device = device))
        x = torch.cat((x[:, :6] @ embd, x[:, 6:10], x[:, -2:]), dim = 1)
        y = self.linear_relu_stack(x).flatten()
        return y

class MLP_source(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(6, 58)
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            
            nn.Linear(64, 64),
            nn.ReLU(),

            nn.Linear(64, 1)
        )

    def forward(self, x):
        embd = self.embedding(torch.arange(6, device = device))
        x = torch.cat((x[:, :6] @ embd, x[:, 6:10], x[:, -2:]), dim = 1)
        y = self.linear_relu_stack(x).flatten()
        return y
