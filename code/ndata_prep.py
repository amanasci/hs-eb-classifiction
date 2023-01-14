import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# save np.load
np_load_old = np.load

# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

X = np.load("X_lstm.npy")
Y = np.load("Y_lstm.npy")

# Finding minimum length LightCurve
min = 10000000000
for e in X:
  if  (min > len(e)):
    min = len(e)

print(min)

#Making every LightCurve same length
xt = []
for e in X:
  xt.append(e[:12901])
X = np.array(xt)
print(X.shape,Y.shape)

mean = np.mean(X,axis=1)
std = np.std(X, axis = 1)
var = np.var(X,axis =1)
kurtosis = kurtosis(X,axis = 1)
skew = skew(X, axis=1)

df = pd.DataFrame()
print(df)
df['lc'] = X.tolist()
df['mean'] = mean
df['std'] = std
df['var'] = var
df['kurtosis'] = kurtosis
df['skew'] = skew
df['class'] = Y
print(df)

df.to_csv('ndata.csv') #Saving ndata.csv
