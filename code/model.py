import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.stats import kurtosis , skew
import pywt

X = np.load("X_lstm.npy")

xt = []
for e in X:
  xt.append(e[:12901])
X = np.array(xt)

df = pd.read_csv("ndata.csv")


y = df['class']
x = df.drop(columns=['class','lc','Unnamed: 0'])

#Wavelet Transform

xl = X
xl = xl/np.max([np.max(i) for i in xl]) #xl.shape = (1591, 12901)
xl,q = pywt.dwt(xl,'sym5')
xl = np.expand_dims(xl, -1) #xl.shape = (1591, 6455, 1)
#New LightCurves have 6455 data points only.


#Normalizing x 

x = x/x.max()

#Trani-Test Split 
from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.3,shuffle=True,random_state=42)
xl_train, xl_test,Y1_train,Y1_test = train_test_split(xl,y,test_size=0.3,shuffle=True,random_state=42)

#Getting shape
N ,D =x.shape #D = 5 in our case.


#Defining Model


dot_img_file = 'model_1.png' #Saving model image


#LSTM input
net_input = keras.layers.Input(shape=(xl.shape[1],xl.shape[2]), name='net_input')
zin = keras.layers.LSTM(512, return_sequences=True)(net_input)
z = keras.layers.LSTM(512)(zin)
z = keras.layers.Dense(1000, activation="relu")(z)


#Normal input
inputs = tf.keras.Input(shape=(D,))
o = keras.layers.Dense(500, activation="relu")(inputs)
o = keras.layers.Dense(500, activation="relu")(o)
o = keras.layers.Dense(1000, activation="relu")(tf.concat([z,o], axis=1)) #Joining both layers
o = keras.layers.Dense(500, activation="relu")(o)
o = keras.layers.GaussianDropout(0.1)(o)
outputs=tf.keras.layers.Dense(1,activation='sigmoid')(o)
model= tf.keras.Model(inputs=[inputs,net_input], outputs=outputs)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001,decay=1e-4),
             loss='binary_crossentropy',
             metrics=['accuracy'])
tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)
print(model.summary())

r = model.fit([X_train,xl_train],Y_train,validation_data=([X_test,xl_test],Y_test),epochs=60,batch_size = 8)

model.save('/Models/model1.h5')

plt.figure(figsize=(12, 8))
plt.plot(r.history['loss'],label='Training Loss')
plt.plot(r.history['val_loss'],label='Validation Loss')
plt.legend()
plt.show()

plt.figure(figsize=(12,8))
plt.plot(r.history['accuracy'],label='Training Accuracy')
plt.plot(r.history['val_accuracy'],label='Validation Accuracy')
plt.legend()
plt.show()