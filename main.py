import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix


def main():
    parkinsons = fetch_ucirepo(id=174) 
    X = parkinsons.data.features 
    y = parkinsons.data.targets 

    # metadata 
    print(parkinsons.metadata) 
    
    # variable information 
    print(parkinsons.variables)

    dataset=pd.read_csv('./telemonitoring/parkinsons_updrs.data')

    x=dataset.iloc[:,3:22].values
    y=dataset.iloc[:, 21].values

    x_train , x_test, y_train , y_test = train_test_split(x,y,test_size=0.2 , random_state=0)

    sc=StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test=sc.transform(x_test)

    classifier = Sequential()

    classifier.add(Dense(
        units=6,
        kernel_initializer="uniform",
        activation="relu",
        input_dim=x_train.shape[1]
    ))

    classifier.add(Dense(
        units=1,
        kernel_initializer="uniform",
        activation="sigmoid"
    ))

    classifier.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=['accuracy']
    )

    y_predict= classifier.predict(x_test).round()
    y_predict=(y_predict > 0.5)

    cm=confusion_matrix(y_test.round() , y_predict)
    print( cm)

    TP = cm[1, 1]  # True Positives
    TN = cm[0, 0]  # True Negatives
    FP = cm[0, 1]  # False Positives
    FN = cm[1, 0]  # False Negatives

    accuracy = (TP + TN) / (TP + TN + FP + FN)

    print("Accuracy:", accuracy)

if __name__ == "__main__":
    main()
