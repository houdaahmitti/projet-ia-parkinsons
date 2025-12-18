from src.load_data import load_parkinsons_data
from src.preprocess import preprocess_data
from src.model import build_model
from src.train import train_model
from src.metrics import regression_metrics, classification_metrics

def main():
    X, y = load_parkinsons_data()
    x_train, x_test, y_train, y_test, scaler = preprocess_data(X, y)
    
    model = build_model(input_dim=x_train.shape[1])
    model, history = train_model(model, x_train, y_train)
    
    y_pred = model.predict(x_test)
    
    mae, mse = regression_metrics(y_test, y_pred)
    accuracy, cm = classification_metrics(y_test, y_pred)
    
    print("MAE:", mae)
    print("MSE:", mse)
    print("Accuracy:", accuracy)
    print("Confusion Matrix:\n", cm)

if __name__ == "__main__":
    main()
