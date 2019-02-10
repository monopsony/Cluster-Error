import numpy as np
import os as os
import shutil as shutil
import hashlib as hashlib
from sgdml.predict import GDMLPredict
from sgdml.utils import io
from sklearn.metrics import mean_squared_error


path = os.path.dirname(os.path.realpath(__file__))

def saveData(data, indices, name):
  
  if isinstance(data,basestring):
    database = np.load(data)
  else:
    database=data
  
  energy, coord, forces, elem = database['E'], database['R'], database['F'], database['z']

  base_vars = {'type': 'd',
               'R': coord[indices],
               'z': elem,
               'E': energy[indices],
               'F': forces[indices],
               'name': name,
               'theory': 'unknown'}
               
  md5_hash = hashlib.md5()
  for key in ['z', 'R', 'E', 'F']:
    md5_hash.update(hashlib.md5(base_vars[key].ravel()).digest())
          
  base_vars['md5'] = md5_hash.hexdigest() 
  
  np.savez_compressed(name, **base_vars)

def remove_all_2D(a,indices):
  for ind in indices:
    for suba in a:
      try:
        suba.remove(ind)
      except:
        pass

def remove_all_1D(a,indices):
  for ind in indices:
    try:
      a.remove(ind)
    except:
      pass

n_test,n_validate=1000,10000
sgdml_input=np.loadtxt(path+"/Info/sgdml_input")
v1=str(int(sgdml_input[0]))

model=np.load(path+"/Info/model_train"+v1+".npz")
training_indices=model["train_idxs"]
test_indices=model["test_idxs"]
print(training_indices)
print(len(training_indices), len(test_indices))

indices=np.load(path+"/Info/indices_all.npy")
print("********indices**************")
print("before/after")
l1,l2,s1,s2=[],[],0,0
for c in indices:
  l1.append(len(c))
  s1=s1+len(c)
remove_all_2D(indices,training_indices)
for c in indices:
  l2.append(len(c))
  s2=s2+len(c)

print(l1)
print(l2)
print(s1,s2)

if os.path.exists(path+"/Info/training_set.npz"):
  os.remove(path+"/Info/training_set.npz")
saveData(path+"/dataset.npz",training_indices,path+"/Info/training_set.npz")
saveData(path+"/dataset.npz",training_indices,path+"/Info/training_set"+v1+".npz")

if os.path.exists(path+"/Info/training_indices.npz"):
  os.remove(path+"/Info/training_indices.npz")
np.save(path+"/Info/training_indices.npy",training_indices)
np.save(path+"/Info/training_indices"+v1+".npy",training_indices)

if os.path.exists(path+"/Info/test_set.npz"):
  os.remove(path+"/Info/test_set.npz")
saveData(path+"/dataset.npz",test_indices,path+"/Info/test_set.npz")

if os.path.exists(path+"/Info/test_indices.npz"):
  os.remove(path+"/Info/test_indices.npz")
np.save(path+"/Info/test_indices.npy",test_indices)

if os.path.exists(path+"/Info/indices_rem"):
  os.remove(path+"/Info/indices_rem")
np.save(path+"/Info/indices_rem.npy",indices)  
np.save(path+"/Info/indices_rem"+v1+".npy",indices)
   
  
  
  
  
  
  
