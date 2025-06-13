import pandas as pd
import numpy as np
from ase import Atoms
from ase.io import read

from itertools import combinations_with_replacement, permutations
from parfor import pmap
from tqdm import tqdm

def stat(s, phase):
    atoms = read("HEA_data/POSCARS/" + s, format = "vasp")
    N = len(atoms)
    NN_num, Ntri_num, NNN_num, N4_num = (12, 24, 6, 24) if phase == "fcc" else (8, 36, 6, 24)
    
    atoms = atoms.repeat(3)
    
    d = []
    for i in range(0, 27 * N):
        d.append(atoms.get_distances(i, range(27 * N)))
    d = np.array(d)

    NNs = np.argsort(d)
    
    symbols = atoms.get_chemical_symbols()
    X_s = [0. for i in range(X_len)]
    
    for i in range(13 * N, 14 * N):
        X_s[elements[symbols[i]]] += 1
         
        triplets = list()
        triplets2 = list()
        for j1 in range(1, NN_num + 1):
            k1 = NNs[i, j1]
            X_s[X_indices_2[(symbols[i], symbols[k1])]] += 1
            
            for j2 in range(j1 + 1, NN_num + 1):
                k2 = NNs[i, j2]
                # print(N, k1, k2)
                triplets.append(((k1, k2), d[i, k1] + d[i, k2] + d[k1, k2]))                
                            
                if phase == "bcc":
                    triplets2.append(((k1, k2), d[i, k1] + d[i, k2] + d[k1, k2]))
            if phase == "bcc":
                for j2 in range(NN_num + 1, NN_num + NNN_num + 1):
                    k2 = NNs[i, j2]
                    triplets.append(((k1, k2), d[i, k1] + d[i, k2] + d[k1, k2]))
        
        triplets.sort(key = lambda x: x[1])
        triplets2.sort(key = lambda x: x[1])

        for t in range(Ntri_num):
            k1, k2 = triplets[t][0]
            X_s[X_indices_3[(symbols[i], symbols[k1], symbols[k2])]] += 1
            if phase == "fcc":
                quadruplets = list()
                for j3 in range(1, NN_num + 1):
                    k3 = NNs[i, j3]
                    if k3 == k1 or k3 == k2: continue
                    quadruplets.append((k3, triplets[t][1] + d[i, k3] + d[k1, k3] + d[k2, k3]))
                quadruplets.sort(key = lambda x: x[1])
                k3 = quadruplets[0][0]
                X_s[X_indices_4[(symbols[i], symbols[k1], symbols[k2], symbols[k3])]] += 1
        
        if phase == "bcc":
            for t in range(12):
                k1, k2 = triplets2[t][0]
                quadruplets = list()
                for j3 in range(NN_num + 1, NN_num + NNN_num + 1):
                    k3 = NNs[i, j3]
                    quadruplets.append((k3, triplets2[t][1] + d[i, k3] + d[k1, k3] + d[k2, k3]))
                quadruplets.sort(key = lambda x: x[1])
                for t2 in range(2):
                    k3 = quadruplets[t2][0]
                    X_s[X_indices_4[(symbols[i], symbols[k1], symbols[k2], symbols[k3])]] += 1

        for j1 in range(NN_num + 1, NN_num + NNN_num + 1):
            k1 = NNs[i, j1]
            X_s[X_indices_2n[(symbols[i], symbols[k1])]] += 1
    
    for i in range(6):
        X_s[i] /= N
    for i in range(6, 26):
        X_s[i] /= NN_num * N
    for i in range(26, 76):
        X_s[i] /= Ntri_num * N
    for i in range(76, 180):
        X_s[i] /= N4_num * N
    for i in range(180, X_len):
        X_s[i] /= NNN_num * N
    return X_s

if __name__ == "__main__":
    elements = {'Fe': 0, 'Ni': 1, 'Co': 2, 'Cr': 3, 'Mn': 4, 'Pd': 5}
    ele = ['Fe', 'Ni', 'Co', 'Cr', 'Mn', 'Pd']
    t = 6

    X_indices_2 = dict() # 2-NN
    X_indices_3 = dict() # 3-NN
    X_indices_4 = dict() # 4-NN
    X_indices_2n = dict() # 2-NNN
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
    for i in combinations_with_replacement(ele, 4):
        if {'Mn', 'Pd'}.issubset(set(i)): continue
        for j in permutations(i):
            if j not in X_indices_4:
                X_indices_4[j] = t
        t += 1
    for i in combinations_with_replacement(ele, 2):
        if {'Mn', 'Pd'}.issubset(set(i)): continue
        for j in permutations(i):
            if j not in X_indices_2n:
                X_indices_2n[j] = t
        t += 1

    X_len = t

    df = pd.read_csv("HEA_data/Database.csv")
    for phase in ['fcc', 'bcc']:
        data = df[df["phase"] == phase]
        structures = data["structures"]
        data_ef = data['Eform (eV/atom)'].reset_index(drop = True)
        data_ms = data['Ms (mub/atom)'].reset_index(drop = True)
        X = pmap(stat, structures, (phase,))
        df_p = pd.concat([pd.DataFrame(X).round(9), data_ef, data_ms], axis = 1)
        df_p.to_csv("HEA_data/" + phase + "_train.csv")

