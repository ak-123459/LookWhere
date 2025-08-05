from src.data.data_loader import load_data
from src.models.train_model import train_model
from src.models.evaluate_model import evaluate_model
from app.utils import tune_hyperparameters
from app.src.data.preprocess import  preprocess_data


def main():

    data = load_data()
    X_train, X_test, y_train, y_test, label_encoder = preprocess_data(data)

    model = train_model(X_train, y_train, label_encoder)

    evaluate_model(model, X_test, y_test, label_encoder)

    # Optional tuning
    print("\nTuning hyperparameters...")
    tune_hyperparameters(X_train, y_train)


if __name__ == "__main__":
    main()
