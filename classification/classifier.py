import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


class Classifier:
    """
    A classifier that wraps around the RandomForest model for training, predicting,
    evaluating, and saving/loading models.

    Attributes:
        model (RandomForestClassifier): The Random Forest model used for classification.

    Methods:
        fit: Trains the model on the provided training data.
        predict: Predicts the class labels for the given test data.
        evaluate: Evaluates the model's accuracy based on the true and predicted labels.
        download: Saves the trained model to a specified directory.
        load: Loads a previously saved model from a specified path.
        predict_proba: Returns the predicted probability distribution for each class.
    """

    def __init__(self):
        """
        Initializes the classifier with a Random Forest model with default parameters.

        Attributes:
            model (RandomForestClassifier): A RandomForest model with 100 trees.
        """
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)

    def fit(self, x_train: pd.DataFrame, y_train: pd.DataFrame):
        """
        Trains the Random Forest model on the provided training data.

        Args:
            x_train (pd.DataFrame): The feature matrix for training (rows: samples, columns: features).
            y_train (pd.DataFrame): The target labels for training (rows: samples, columns: labels).
        """
        self.model.fit(x_train, y_train)

    def predict(self, x_test: pd.DataFrame) -> pd.DataFrame:
        """
        Predicts the class labels for the given test data.

        Args:
            x_test (pd.DataFrame): The feature matrix for testing (rows: samples, columns: features).

        Returns:
            pd.DataFrame: The predicted class labels for the test data.
        """
        predictions = self.model.predict(x_test)
        return predictions

    def evaluate(self, y_test: pd.DataFrame, y_pred: pd.DataFrame) -> float:
        """
        Evaluates the model's performance using accuracy score.

        Args:
            y_test (pd.DataFrame): The true class labels for the test data.
            y_pred (pd.DataFrame): The predicted class labels for the test data.

        Returns:
            float: The accuracy of the model, calculated as the percentage of correct predictions.
        """
        return accuracy_score(y_test, y_pred)

    def download(self, directory: str, model_name: str):
        """
        Saves the trained model to the specified directory.

        Args:
            directory (str): The path to the directory where the model will be saved.
            model_name (str): The name of the model file (e.g., 'flower_classifier_model').
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
        model_path = os.path.join(directory, model_name + ".pkl")
        joblib.dump(self.model, model_path)
        print(f"Model saved to {model_path}")

    def load(self, model_path: str):
        """
        Loads a previously saved model from the given path.

        Args:
            model_path (str): The path to the saved model file.
        """
        self.model = joblib.load(model_path)
        print(f"Model loaded from {model_path}")

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        """
        Returns the predicted probability distribution for each class for the given data.

        Args:
            x (pd.DataFrame): The feature matrix for prediction (rows: samples, columns: features).

        Returns:
            np.ndarray: A 2D array containing the probability distributions for each class.
                        Rows correspond to samples, and columns correspond to class probabilities.
        """
        proba = self.model.predict_proba(x)
        return proba
