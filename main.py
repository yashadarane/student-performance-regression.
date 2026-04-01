from sklearn.model_selection import train_test_split
from src.config import *
from src.data_loader import load_data
from src.preprocessing import handle_missing_values, encode_categorical, select_features
from src.model import train_model, predict
from src.evaluate import evaluate
from src.visualise import (
    plot_correlation,
    plot_predictions,
    plot_feature_relationships,
    plot_residuals,
    plot_target_distribution
)


def main():
    # Load data
    df = load_data(path)

    # Preprocessing
    df = handle_missing_values(df)
    df = encode_categorical(df)

    # Visualization
    plot_correlation(df)

    # Feature selection
    X, y = select_features(df, features, target)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Train model
    model = train_model(X_train, y_train)

    # Predict
    y_pred = predict(model, X_test)

    # Evaluate
    results = evaluate(y_test, y_pred)

    print("\nModel Performance:")
    for key, value in results.items():
        print(f"{key}: {value:.4f}")

    # Visualization
    plot_predictions(y_test, y_pred)
    plot_feature_relationships(df, target)
    plot_residuals(y_test, y_pred)

if __name__ == "__main__":
    main()