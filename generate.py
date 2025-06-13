import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ase import Atoms
from ase.formula import Formula
from ase.io import read

from sklearn.linear_model import Ridge, Lasso

from common import composition

from itertools import combinations_with_replacement, permutations
from tqdm import tqdm

elements = {'Fe': 0, 'Ni': 1, 'Co': 2, 'Cr': 3, 'Mn': 4, 'Pd': 5}
ele = ['Fe', 'Ni', 'Co', 'Cr', 'Mn', 'Pd']
X_indices_2 = dict()
X_indices_3 = dict()
X_indices_2n = dict()
t = 6

for i in combinations_with_replacement(ele, 2):
    if {'Mn', 'Pd'}.issubset(set(i)): continue
    for j in permutations(i):
        if j not in X_indices_2:
            X_indices_2[j] = t
    t += 1
for i in combinations_with_replacement(ele, 3):
    if {'Mn', 'Pd'}.issubset(set(i)): continue
    for j in permutations(i):
        if j not in X_indices_3:
            X_indices_3[j] = t
    t += 1
for i in combinations_with_replacement(ele, 2):
    if {'Mn', 'Pd'}.issubset(set(i)): continue
    for j in permutations(i):
        if j not in X_indices_2n:
            X_indices_2n[j] = t
    t += 1
X_len = t

X2 = {"fcc": [], "bcc": []}
X3 = {"fcc": [], "bcc": []}
X2n = {"fcc": [], "bcc": []}

s = {"fcc": "FeNiCoCrMn_sqsfcc", "bcc": "FeNiCoCrMn_sqsbcc"}

for phase in ["fcc", "bcc"]:
    atoms = read("HEA_data/POSCARS/" + s[phase], format = "vasp")
    N = len(atoms)
    NN_num, Ntri_num, NNN_num = (12, 24, 6) if phase == "fcc" else (8, 36, 6)
    atoms = atoms.repeat(3)
    d = atoms.get_all_distances(mic = False)    
    NNs = np.argsort(d)
        
    for i in range(13 * N, 14 * N):
        triplets = list()
        for j1 in range(1, NN_num + 1):
            k1 = NNs[i, j1]
            X2[phase].append((i, k1))
            
            for j2 in range(j1 + 1, NN_num + 1):
                k2 = NNs[i, j2]
                triplets.append(((k1, k2), d[i, k1] + d[i, k2] + d[k1, k2]))
                
            if phase == "bcc":
                for j2 in range(NN_num + 1, NN_num + NNN_num + 1):
                    k2 = NNs[i, j2]
                    triplets.append(((k1, k2), d[i, k1] + d[i, k2] + d[k1, k2]))
            
        triplets.sort(key = lambda x: x[1])
        for t in range(Ntri_num):
            k1, k2 = triplets[t][0]
            X3[phase].append((i, k1, k2))
        
        for j1 in range(NN_num + 1, NN_num + NNN_num + 1):
            k1 = NNs[i, j1]
            X2n[phase].append((i, k1))

def fit(property, phase, train_index, alpha, r):
    data = pd.read_csv("HEA_data/" + phase + "_train.csv")
    X = np.concatenate((data.iloc[:, 1:77].to_numpy(), data.iloc[:, 182:-2].to_numpy()), axis = 1)
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
    for (i, j) in X2n[phase]:
        X[X_indices_2n[(symbols_new[i], symbols_new[j])]] += 1

    for i in range(6):
        X[i] /= N
    for i in range(6, 26):
        X[i] /= len(X2[phase])
    for i in range(26, 76):
        X[i] /= len(X3[phase])
    for i in range(76, X_len):
        X[i] /= len(X2n[phase])
    y_s = np.inner(coef, X) + intercept
    return X_s, y_s

def Gen(train_index, atom_nums, property, phase, alpha):
    coef, intercept = fit(property, phase, train_index, alpha, None)
    X_new = []
    y_new = []
    for atom_num in tqdm(atom_nums):
        X_s, y_s = generate(atom_num, coef, intercept, phase)
        X_new.append(X_s)
        y_new.append(y_s)
    return X_new, y_new