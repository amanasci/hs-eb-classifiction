import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os


def load_data():
    """
        This function loads the csv files for each object in the each class and cleans them using the procedure 
        in Exploring short-term optical variability of blazars using TESS, Monthly Notices of the Royal Astronomical Society,
        https://doi.org/10.1093/mnras/stac3125 

        The data was predownloaded using a different script.
    """
  n=0
  dataX=[]
  for file in os.listdir("/Datasets/Star Classification/ECB_data/"):
    try : 
      df = pd.read_csv("/Datasets/Star Classification/ECB_data/"+file)
      x=pd.to_numeric(df['sap_flux'],errors='coerce')
      mean = x.mean()
      std = x.std()
      df= df[np.logical_and(x<(mean+3*std),x>(mean-3*std))]
      df=pd.to_numeric(df['sap_flux'],errors='coerce')
      dataX.append(np.array(df,dtype=np.float))
      n+=1
    except:
      print("Error in "  + file)
  
  dataY = [0.0 for y in range(n)]
  print("Done ECBs")
  
  n=0
  for file in os.listdir("/Datasets/Star Classification/HS_data/"):
    df = pd.read_csv("/Datasets/Star Classification/HS_data/"+file)
    x=pd.to_numeric(df['sap_flux'],errors='coerce')
    mean = x.mean()
    std = x.std()
    df= df[np.logical_and(x<(mean+3*std),x>(mean-3*std))]
    df=pd.to_numeric(df['sap_flux'],errors='coerce')
    dataX.append(np.array(df,dtype=np.float))
    n+=1

  dataY2 = [1.0 for y in range(n)]
  dataY = np.array(dataY)
  dataY2 = np.array(dataY2)
  dataY= np.append(dataY,dataY2)
  dataX =np.array(dataX)
  return dataX,dataY

X,Y = load_data()
np.save("X_lstm.npy",X)
np.save("Y_lstm.npy",Y)