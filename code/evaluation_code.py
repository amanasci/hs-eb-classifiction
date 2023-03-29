from sklearn.model_selection import train_test_split
import numpy as np


def eval(model,x,xl, y, n): 
    seeds = np.random.randint(1,1000,n)

    acc_score = 0
    loss_score = 0 
    for s in seeds:
        X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.3,shuffle=True,random_state=s)
        xl_train, xl_test,Y1_train,Y1_test = train_test_split(xl,y,test_size=0.3,shuffle=True,random_state=s)
        score = model.evaluate([X_test,xl_test],Y_test,batch_size=8)
        loss_score += score[0]
        acc_score += score[1]

    print(acc_score/len(seeds))