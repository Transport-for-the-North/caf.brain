# -*- coding: utf-8 -*-
"""
Created on: 10/8/2024
Original author: Adil Zaheer
"""
# Built-Ins
import os

# Third Party
import numpy as np
import pandas as pd

# pylint: disable=import-error,wrong-import-position
# pylint: enable=import-error,wrong-import-position
import tensorflow as tf
from keras.saving.save import load_model
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


#### simple model functions_to_be_processed (tensorflow logistic regression) ####
def create_tf_logistic_regression(input_dim):
    model = tf.keras.Sequential(
        [tf.keras.layers.Dense(1, input_shape=(input_dim,), activation="sigmoid")]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def train_model(
    X_train,
    y_train,
    X_predict,
    y_validate,
    output_folder,
    index_col,
    target_column,
    predict_data,
):
    model_path = os.path.join(output_folder, "tf_model")

    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
        print("Saved model loaded")
    else:
        model = create_tf_logistic_regression(X_train.shape[1])
        history = model.fit(
            X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=1
        )
        model.save(model_path)

    predict_predictions = model.predict(X_predict)

    evaluation_results = evaluate_model(
        model, X_train, y_train, X_predict, y_validate, output_folder
    )
    print(f"Train Accuracy: {evaluation_results['Train Accuracy']:.4f}")
    print(f"Validation Accuracy: {evaluation_results['Validation Accuracy']:.4f}")
    print(f"Validation AUC: {evaluation_results['Validation AUC']:.4f}")
    print("\nTop 5 Important Features:")

    prediction_df = pd.DataFrame(
        {target_column: predict_predictions.flatten()}, index=predict_data.index
    )
    prediction_file_path = os.path.join(output_folder, "predictions.csv")
    prediction_df.to_csv(prediction_file_path, index_label=index_col, index=True)
    print(f"Predictions saved to: {prediction_file_path}")

    return model, predict_predictions


#### complex model functions_to_be_processed ####
def create_improved_model(input_dim):
    model = Sequential(
        [
            Dense(64, activation="relu", input_shape=(input_dim,)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(32, activation="relu"),
            BatchNormalization(),
            Dropout(0.3),
            Dense(16, activation="relu"),
            BatchNormalization(),
            Dropout(0.3),
            Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC()],
    )

    return model


def train_improved_model(
    X_train,
    y_train,
    X_predict,
    y_validate,
    output_folder,
    index_col,
    target_column,
    predict_data,
    epochs=100,
    batch_size=32,
):
    model_path = os.path.join(output_folder, "improved_tf_model")

    if os.path.exists(model_path):
        model = load_model(model_path)
        print("Saved model loaded")
    else:
        model = create_improved_model(X_train.shape[1])

        validation_split = 0.2
        X_train_split, X_val_split = np.split(
            X_train, [int((1 - validation_split) * len(X_train))]
        )
        y_train_split, y_val_split = np.split(
            y_train, [int((1 - validation_split) * len(y_train))]
        )

        early_stopping = EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        )
        reduce_lr = ReduceLROnPlateau(
            monitor="val_loss", factor=0.2, patience=5, min_lr=0.00001
        )

        history = model.fit(
            X_train_split,
            y_train_split,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val_split, y_val_split),
            callbacks=[early_stopping, reduce_lr],
            verbose=1,
        )

        model.save(model_path)
        print(f"Model saved to {model_path}")

    predict_predictions = model.predict(X_predict)

    evaluation_results = evaluate_model(
        model, X_train, y_train, X_predict, y_validate, output_folder
    )

    print(f"Train Accuracy: {evaluation_results['Train Accuracy']:.4f}")
    print(f"Validation Accuracy: {evaluation_results['Validation Accuracy']:.4f}")
    print(f"Validation AUC: {evaluation_results['Validation AUC']:.4f}")
    print("\nTop 5 Important Features:")

    prediction_df = pd.DataFrame(
        {target_column: predict_predictions.flatten()}, index=predict_data.index
    )
    prediction_file_path = os.path.join(output_folder, "improved_predictions.csv")
    prediction_df.to_csv(prediction_file_path, index_label=index_col, index=True)
    print(f"Predictions saved to: {prediction_file_path}")

    return model, predict_predictions


#### coefficient stats ####
def calculate_coefficient_stats(model, X_train, y_train, output_folder):
    if len(model.layers) > 1:
        return calculate_complex_model_stats(model, X_train, y_train, output_folder)

    # Extract coefficients and intercept
    coefficients = model.layers[0].get_weights()[0].flatten()
    intercept = model.layers[0].get_weights()[1][0]

    # Calculate standard errors
    y_pred = model.predict(X_train).flatten()
    residuals = y_train - y_pred
    mse = np.mean(residuals**2)
    std_errors = np.sqrt(mse * (1 / np.sum((X_train - X_train.mean()) ** 2, axis=0)))

    # Calculate z-scores and p-values
    z_scores = coefficients / std_errors
    p_values = 2 * (1 - stats.norm.cdf(np.abs(z_scores)))

    feature_names = X_train.columns.tolist()
    results = pd.DataFrame(
        {
            "Feature": feature_names,
            "Coefficient": coefficients,
            "Std_Error": std_errors,
            "Z_Score": z_scores,
            "P_Value": p_values,
            "Odds_Ratio": np.exp(coefficients),
        }
    )

    # Add intercept to results
    intercept_row = pd.DataFrame(
        {
            "Feature": ["Intercept"],
            "Coefficient": [intercept],
            "Std_Error": [np.nan],
            "Z_Score": [np.nan],
            "P_Value": [np.nan],
            "Odds_Ratio": [np.exp(intercept)],
        }
    )
    results = pd.concat([intercept_row, results]).reset_index(drop=True)
    results.to_csv(os.path.join(output_folder, "coefficients_results.csv"))
    return results


def calculate_complex_model_stats(model, X_train, y_train, output_folder):
    feature_names = X_train.columns.tolist()

    # Calculate baseline score
    y_pred = model.predict(X_train)
    baseline_score = accuracy_score(y_train, (y_pred > 0.5).astype(int))

    # Calculate feature importance
    importances = []
    for col in X_train.columns:
        X_permuted = X_train.copy()
        X_permuted[col] = np.random.permutation(X_permuted[col])
        y_pred_permuted = model.predict(X_permuted)
        permuted_score = accuracy_score(y_train, (y_pred_permuted > 0.5).astype(int))
        importance = baseline_score - permuted_score
        importances.append(importance)

    results = pd.DataFrame({"Feature": feature_names, "Importance": importances})

    # Sort by importance
    results = results.sort_values("Importance", ascending=False).reset_index(drop=True)

    results.to_csv(f"{output_folder}/feature_importance_results.csv", index=False)
    return results


def calculate_feature_importance(model, X, y, output_folder):
    n_repeats = 5
    baseline_score = roc_auc_score(y, model.predict(X))
    importances = []
    for col in X.columns:
        scores = []
        for _ in range(n_repeats):
            X_permuted = X.copy()
            X_permuted[col] = np.random.permutation(X_permuted[col])
            y_pred_permuted = model.predict(X_permuted)
            permuted_score = roc_auc_score(y, y_pred_permuted)
            scores.append(baseline_score - permuted_score)
        importances.append(np.mean(scores))

    dat = pd.DataFrame({"Feature": X.columns, "Importance": importances})
    dat.to_csv(os.path.join(output_folder, "feature_importance_results.csv"))
    return


def evaluate_model(model, X_train, y_train, X_predict, y_validate, output_folder):
    # Train set performance
    y_train_pred = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, (y_train_pred > 0.5).astype(int))
    train_auc = roc_auc_score(y_train, y_train_pred)

    # Validation set performance
    y_val_pred = model.predict(X_predict)
    val_accuracy = accuracy_score(y_validate, (y_val_pred > 0.5).astype(int))
    val_auc = roc_auc_score(y_validate, y_val_pred)

    # Classification report and confusion matrix
    val_predictions = (y_val_pred > 0.5).astype(int)
    class_report = classification_report(y_validate, val_predictions, output_dict=True)
    conf_matrix = confusion_matrix(y_validate, val_predictions)

    # Feature importance
    if len(model.layers) > 1:
        calculate_feature_importance(model, X_predict, y_validate, output_folder=output_folder)

    results = {
        "Train Accuracy": train_accuracy,
        "Train AUC": train_auc,
        "Validation Accuracy": val_accuracy,
        "Validation AUC": val_auc,
        "Confusion Matrix": conf_matrix.tolist(),
        "Classification Report": class_report,
    }

    dat = pd.DataFrame([results])
    dat.to_csv(os.path.join(output_folder, "model_evaluation_results.csv"), index=False)

    return results
