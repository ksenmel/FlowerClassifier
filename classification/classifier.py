import os
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


class Classifier:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)

    def fit(self, df: pd.DataFrame):
        """
        Обучает модель на данных.

        Args:
            X (np.ndarray): Признаки (например, извлеченные признаки изображений).
            y (np.ndarray): Метки классов (например, номера типов цветов).
            :param df:
        """
        X = df.drop(columns=['label']).values
        y = df['label'].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)

        self.model.fit(X_train, y_train)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Предсказывает классы для новых данных.
        """
        X = df.drop(columns=['label']).values

        predictions = self.model.predict(X)
        return predictions

    def evaluate(self, df: pd.DataFrame, y_pred) -> float:
        """
        Returns:
            float: Точность модели на данных.
        """
        X = df.drop(columns=['label']).values
        y = df['label'].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)

        return accuracy_score(y_test, y_pred)

    def download(self, directory: str, model_name: str = 'v1FlowerClassifier_model'):
        """
        Скачивает обученную модель в заданную директорию.

        Args:
            directory (str): Путь к директории, куда будет сохранена модель.
            model_name (str): Имя файла модели (по умолчанию 'flower_classifier_model').
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
        model_path = os.path.join(directory, model_name + '.pkl')
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
