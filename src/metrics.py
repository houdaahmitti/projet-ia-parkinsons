from sklearn.metrics import mean_absolute_error, mean_squared_error, confusion_matrix

def regression_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    return mae, mse

def classification_metrics(y_true, y_pred, threshold=20):
    y_true_class = (y_true > threshold).astype(int)
    y_pred_class = (y_pred > threshold).astype(int)
    
    cm = confusion_matrix(y_true_class, y_pred_class)
    TP = cm[1, 1]
    TN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]
    
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    return accuracy, cm
