import pandas as pd
import numpy as np
from ase import Atoms
from ase.io import read

from tqdm import tqdm

elements = {'Fe': 0, 'Ni': 1, 'Co': 2, 'Cr': 3, 'Mn': 4, 'Pd': 5}
ele = ['Fe', 'Ni', 'Co', 'Cr', 'Mn', 'Pd']
X_indices = dict()

t = 6
for i in range(6):
    for j in range(i, 6):
        if j == 5 and i == 4: continue
        X_indices[(ele[i], ele[j])] = t
        X_indices[(ele[j], ele[i])] = t
        t += 1

for i in range(6):
    for j in range(i, 6):
        if j == 5 and i == 4: continue
        for k in range(j, 6):
            if k == 5 and j == 4: continue
            X_indices[(ele[i], ele[j], ele[k])] = t
            X_indices[(ele[i], ele[k], ele[j])] = t
            X_indices[(ele[j], ele[i], ele[k])] = t
            X_indices[(ele[j], ele[k], ele[i])] = t
            X_indices[(ele[k], ele[i], ele[j])] = t
            X_indices[(ele[k], ele[j], ele[i])] = t   
            t += 1         
X_len = t

df = pd.read_csv("HEA_data/Database.csv")
for phase in ["fcc", "bcc"]:
    data = df[df["phase"] == phase]
    structures = data["structures"]
    data_ef = data['Eform (eV/atom)'].reset_index(drop = True)
    data_ms = data['Ms (mub/atom)'].reset_index(drop = True)
    X = []

    pbar = tqdm(structures)
    for s in pbar:
        pbar.set_description(s)

        atoms = read("HEA_data/POSCARS/" + s, format = "vasp")
        N = len(atoms)
        NN_num, Ntri_num = (12, 24) if phase == "fcc" else (8, 36)
        
        atoms = atoms.repeat(3)
        d = atoms.get_all_distances(mic = False)
        NNs = np.argsort(d)
        
        symbols = atoms.get_chemical_symbols()
        X_s = [0. for i in range(X_len)]
        
        for i in range(13 * N, 14 * N):
            X_s[elements[symbols[i]]] += 1
            
            triplets = list()
            for j1 in range(1, NN_num + 1):
                k1 = NNs[i, j1]
                X_s[X_indices[(symbols[i], symbols[k1])]] += 1
                
                for j2 in range(j1 + 1, NN_num + 1):
                    k2 = NNs[i, j2]
                    # print(N, k1, k2)
                    triplets.append(((k1, k2), d[i, k1] + d[i, k2] + d[k1, k2]))
                                
                if phase == "bcc":
                    NNN_num = 6
                    for j2 in range(NN_num + 1, NN_num + NNN_num + 1):
                        k2 = NNs[i, j2]
                        triplets.append(((k1, k2), d[i, k1] + d[i, k2] + d[k1, k2]))
                
            triplets.sort(key = lambda x: x[1])
            for t in range(Ntri_num):
                k1, k2 = triplets[t][0]
                X_s[X_indices[(symbols[i], symbols[k1], symbols[k2])]] += 1
        
        for i in range(6):
            X_s[i] /= N
        for i in range(6, 26):
            X_s[i] /= NN_num * N
        for i in range(26, X_len):
            X_s[i] /= Ntri_num * N
        X.append(X_s)
    
    df_p = pd.concat([pd.DataFrame(X).round(9), data_ef, data_ms], axis = 1)
    df_p.to_csv("HEA_data/" + phase + "_train.csv")

