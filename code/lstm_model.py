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


## LSTM Implementation

dot_img_file = 'model_2.png'



#LSTM input
net_input = keras.layers.Input(shape=(xl.shape[1],xl.shape[2]), name='net_input')
zin = keras.layers.LSTM(512, return_sequences=True)(net_input)
z = keras.layers.LSTM(512)(zin)
z = keras.layers.Dense(1000, activation="relu")(z)


o = keras.layers.Dense(1000, activation="relu")(z)
o = keras.layers.Dense(500, activation="relu")(o)
o = keras.layers.GaussianDropout(0.2)(o)
outputs=tf.keras.layers.Dense(1,activation='sigmoid')(o)
model= tf.keras.Model(inputs=net_input, outputs=outputs)
model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.0001),
             loss='binary_crossentropy',
             metrics=['accuracy'])
tf.keras.utils.plot_model(model, to_file=dot_img_file,dpi=196, show_shapes=True,show_layer_names=False)
print(model.summary())

r = model.fit(xl_train,Y_train,validation_split=0.1, epochs=100, batch_size = 12)

model.save('model_lstm.h5')

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


print("Model Evaluation on Test Data")
print(model.evaluate(xl_train,Y_test))

from sklearn.metrics import confusion_matrix
import seaborn as sn

# Make predictions on your test data
predictions = model.predict(X_test)

# Convert predictions to binary class labels (0 or 1)
predictions = np.round(predictions)

# Compute the confusion matrix
confusion_matrix = confusion_matrix(Y_test, predictions)

# Print the confusion matrix
print(confusion_matrix)


# Create a dataframe from the confusion matrix
df_cm = pd.DataFrame(confusion_matrix, index = ["Class 0","Class 1"],
                  columns = ["Class 0","Class 1"])
plt.figure(figsize=(12,8))
# Create a heatmap from the dataframe
sn.heatmap(df_cm, annot=True,fmt='g',cmap="Blues")


plt.xlabel("Predicted")
plt.ylabel("True")
plt.title('Confusion Matrix')
plt.show()

from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import classification_report


# Convert predictions to binary class labels (0 or 1)
predictions1 = np.round(predictions)

# Compute the precision score
precision = precision_score(Y_test, predictions1)

# Compute the recall score
recall = recall_score(Y_test, predictions1)

# Compute the F1 score
f1 = f1_score(Y_test, predictions1)

# Print the precision, recall, and F1 scores
print("Precision: {:.3f}".format(precision))
print("Recall: {:.3f}".format(recall))
print("F1: {:.3f}".format(f1))

# Get the classification report
class_report = classification_report(Y_test, predictions1)

print(class_report)


from sklearn.metrics import roc_curve, auc

# Compute false positive rate, true positive rate, and thresholds
fpr, tpr, thresholds = roc_curve(Y_test, predictions)

# Compute area under the curve (AUC)
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure()
plt.figure(figsize=(12,8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.3f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()