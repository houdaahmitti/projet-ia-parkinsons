from src.load_data import load_parkinsons_data
from src.preprocess import preprocess_data
from src.model import build_model
from src.train import train_model

def main():
    X, y = load_parkinsons_data()

    x_train, x_test, y_train, y_test, scaler = preprocess_data(X, y)

    model = build_model(input_dim=x_train.shape[1])

    model, results = train_model(
        model,
        x_train,
        y_train,
        x_test,
        y_test
    )

    print("\n--- Evaluation Results ---")
    for k, v in results.items():
        print(k, ":", v)

if __name__ == "__main__":
    main()
