import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ase import Atoms
from ase.formula import Formula
from ase.io import read

from sklearn.linear_model import Ridge, Lasso

from common import composition

from tqdm import tqdm


elements = {'Fe': 0, 'Ni': 1, 'Co': 2, 'Cr': 3, 'Mn': 4, 'Pd': 5}
ele = ['Fe', 'Ni', 'Co', 'Cr', 'Mn', 'Pd']
X_indices_2 = dict()
X_indices_3 = dict()

t = 6
for i in range(6):
    for j in range(i, 6):
        if j == 5 and i == 4: continue
        X_indices_2[(ele[i], ele[j])] = t
        X_indices_2[(ele[j], ele[i])] = t
        t += 1

for i in range(6):
    for j in range(i, 6):
        if j == 5 and i == 4: continue
        for k in range(j, 6):
            if k == 5 and j == 4: continue
            X_indices_3[(ele[i], ele[j], ele[k])] = t
            X_indices_3[(ele[i], ele[k], ele[j])] = t
            X_indices_3[(ele[j], ele[i], ele[k])] = t
            X_indices_3[(ele[j], ele[k], ele[i])] = t
            X_indices_3[(ele[k], ele[i], ele[j])] = t
            X_indices_3[(ele[k], ele[j], ele[i])] = t   
            t += 1         
X_len = t

X2 = {"fcc": [], "bcc": []}
X3 = {"fcc": [], "bcc": []}


s = {"fcc": "FeNiCoCrMn_sqsfcc", "bcc": "FeNiCoCrMn_sqsbcc"}

for phase in ["fcc", "bcc"]:
    atoms = read("HEA_data/POSCARS/" + s[phase], format = "vasp")
    N = len(atoms)
    NN_num, Ntri_num = (12, 24) if phase == "fcc" else (8, 36)
    atoms = atoms.repeat(3)
    d = atoms.get_all_distances(mic = False)    
    NNs = np.argsort(d)        
        
    for i in range(13 * N, 14 * N):
        triplets = list()
        for j1 in range(1, NN_num + 1):
            k1 = NNs[i, j1]
            # if k1 > i: pass
            X2[phase].append((i, k1))
            
            for j2 in range(j1 + 1, NN_num + 1):
                k2 = NNs[i, j2]
                triplets.append(((k1, k2), d[i, k1] + d[i, k2] + d[k1, k2]))
                
            if phase == "bcc":
                NNN_num = 6
                for j2 in range(NN_num + 1, NN_num + NNN_num + 1):
                    k2 = NNs[i, j2]
                    triplets.append(((k1, k2), d[i, k1] + d[i, k2] + d[k1, k2]))
            
        triplets.sort(key = lambda x: x[1])
        for t in range(Ntri_num):
            k1, k2 = triplets[t][0]
            X3[phase].append((i, k1, k2))

def fit(property, phase, train_index, alpha):
    data = pd.read_csv("HEA_data/" + phase + "_train.csv")
    X = data.iloc[:, 1:-2].to_numpy()
    X = X[train_index]
    y = data[property].to_numpy()
    y = y[train_index]
    X_train = X
    y_train = y
    param = dict(alpha = alpha, max_iter = 10 ** 4)
    regr = Ridge(**param)
    regr.fit(X_train, y_train)
    return regr.coef_, regr.intercept_

def generate(atom_num, coef, intercept, phase):
    symbols = []
    for i in range(6):
        symbol = ele[i]
        for j in range(atom_num[i]):
            symbols.append(symbol)
    s = Formula.from_list(symbols).format("reduce")
    X_s = composition(s)

    X = [0. for i in range(X_len)]

    np.random.seed(0)
    np.random.shuffle(symbols)
    symbols_new = np.tile(symbols, 27)
    for i in range(13 * N, 14 * N):
        X[elements[symbols_new[i]]] += 1
    for (i, j) in X2[phase]:
        X[X_indices_2[(symbols_new[i], symbols_new[j])]] += 1
    for (i, j, k) in X3[phase]:
        X[X_indices_3[(symbols_new[i], symbols_new[j], symbols_new[k])]] += 1

    for i in range(6):
        X[i] /= N
    for i in range(6, 26):
        X[i] /= len(X2[phase])
    for i in range(26, X_len):
        X[i] /= len(X3[phase])
    if len(coef) == 76:
        y_s = np.inner(coef, X) + intercept
    return X_s, y_s

def Gen(train_index, atom_nums, property, phase, alpha):
    coef, intercept = fit(property, phase, train_index, alpha)
    X_new = []
    y_new = []
    for atom_num in tqdm(atom_nums):
        X_s, y_s = generate(atom_num, coef, intercept, phase)
        X_new.append(X_s)
        y_new.append(y_s)
    return X_new, y_new

