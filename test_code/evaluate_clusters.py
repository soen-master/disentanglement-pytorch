import numpy as np
from kmodes.kmodes import KModes
import pickle
import torch

mask_attr = np.zeros(shape=(40,3))
with open('which_attributes.txt', 'r') as file:
    i = 0
    for line in file:
        h = (line.split(','))
        mask_attr[i, 0] = i
        mask_attr[i,1] = int(h[1])
        mask_attr[i,2] = int(h[2][:-1])
        i += 1

Y = []
with open('list_attr_celeba.csv', 'r') as fp:
    lines = fp.readlines()
    columns = ['filename'] + lines[0].split(',')
    columns = columns[2:]
    for line in lines[1:]:
        values = [value.strip() for value in line.split(',')]
        #print(line)
        #print(values)
        assert len(values[1:]) == len(columns), str(len(values))+' '+str(len(columns))

        img_filename = values[0]
        attributes = (np.array(list(map(int, values[1:]))) > 0).astype(int)
        Y.append(attributes)
        
columns = [i for i in range(40) if mask_attr[i,1] == 1]
Y = np.array(Y)
Y = Y[:, columns] 
Z = Y
np.random.shuffle(Y)
Y = Y[:25000]

print('columns:', columns)
print('len:',len(columns))

with open('new_km.pickle', 'rb') as f:
    km = pickle.load(f)


labels = km.predict(Z)
labels = torch.tensor(np.asarray(labels, dtype=int))

l = len(labels)
total = []
for i in range(10):
    mask = (labels == i)
    total.append(len(labels[mask]))
    

print('Perc', np.array(total)/l)


# It is important to use binary access

from sklearn.linear_model import LogisticRegression     
log_reg = LogisticRegression(max_iter=10000)
log_reg.fit(Z, labels)
print('Score', log_reg.score(Z, labels))
print('completed')
