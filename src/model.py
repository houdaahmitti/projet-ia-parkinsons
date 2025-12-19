from keras.models import Sequential
from keras.layers import Dense

def build_model(input_dim):
    
    model = Sequential()
    model.add(Dense(12, activation="relu", kernel_initializer="he_uniform", input_dim=input_dim))
    model.add(Dense(6, activation="relu", kernel_initializer="he_uniform"))
    model.add(Dense(1, activation="linear"))
    model.compile(optimizer="adam", loss="huber", metrics=["mae"])
    return model
