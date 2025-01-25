import os
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class Classifier:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)

    def fit(self, x_train: pd.DataFrame, y_train: pd.DataFrame):
        """
        Обучает модель на данных.
        """
        self.model.fit(x_train, y_train)

    def predict(self, x_test: pd.DataFrame) -> pd.DataFrame:
        """
        Предсказывает классы для новых данных.
        """
        predictions = self.model.predict(x_test)
        return predictions

    def evaluate(self, y_test: pd.DataFrame, y_pred: pd.DataFrame) -> float:

        return accuracy_score(y_test, y_pred)

    def download(self, directory: str, model_name: str = "v1FlowerClassifier_model"):
        """
        Скачивает обученную модель в заданную директорию.

        Args:
            directory (str): Путь к директории, куда будет сохранена модель.
            model_name (str): Имя файла модели (по умолчанию 'flower_classifier_model').
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
        model_path = os.path.join(directory, model_name + ".pkl")
        joblib.dump(self.model, model_path)
        print(f"Модель сохранена в {model_path}")

    def load(self, model_path: str):
        """
        Загружает обученную модель.

        Args:
            model_path (str): Путь к файлу модели.
        """
        self.model = joblib.load(model_path)
        print(f"Модель загружена из {model_path}")

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        """
        Выводит распределение вероятностей для каждого класса.

        Args:
            df (pd.DataFrame): Данные для предсказания.

        Returns:
            np.ndarray: Массив вероятностей для каждого класса для каждого примера.
        """
        proba = self.model.predict_proba(x)
        return proba
