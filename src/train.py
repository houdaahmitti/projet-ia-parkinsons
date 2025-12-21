from keras.callbacks import EarlyStopping
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    confusion_matrix
)
import matplotlib.pyplot as plt
import numpy as np


def train_model(model, x_train, y_train, x_test, y_test, epochs=50, batch_size=32, threshold=20):

    # -------- Training --------
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True
    )

    history = model.fit(
        x_train,
        y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=1
    )

    # -------- Prediction --------
    y_pred = model.predict(x_test).ravel()

    # -------- Regression Metrics --------
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # -------- Classification Metrics --------
    y_test_class = (y_test > threshold).astype(int)
    y_pred_class = (y_pred > threshold).astype(int)

    cm = confusion_matrix(y_test_class, y_pred_class)

    TP = cm[1, 1]
    TN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]

    accuracy = (TP + TN) / (TP + TN + FP + FN)

    # -------- Plots --------
    plt.figure(figsize=(8,5))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Learning Curves')
    plt.legend()
    plt.grid(True)
    plt.show()

    # -------- Results --------
    results = {
        "MAE": mae,
        "MSE": mse,
        "R2": r2,
        "Accuracy": accuracy,
        "Confusion Matrix": cm,
        "TP": TP,
        "TN": TN,
        "FP": FP,
        "FN": FN
    }
    model.save("model.h5")

    return model, results
