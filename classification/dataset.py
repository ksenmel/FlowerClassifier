import os

import numpy as np
import pandas as pd
import cv2
from PIL import Image
from torch.utils.data import Dataset

from classification.features import feature_functions


class FlowerDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.image_paths = []
        self.labels = []
        self.classes = sorted(os.listdir(path))

        for label, class_name in enumerate(self.classes):
            class_dir = os.path.join(path, class_name)
            if os.path.isdir(class_dir):
                for file_name in os.listdir(class_dir):
                    if file_name.endswith((".jpg", ".jpeg", ".png")):
                        self.image_paths.append(os.path.join(class_dir, file_name))
                        self.labels.append(label)

        self.feature_functions = feature_functions

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(image_path)

        return image, label

    def extract_features(self, image: np.ndarray) -> dict:
        """
        Извлекает признаки из изображения и возвращает их в виде словаря.

        Args:
            image (np.ndarray): Изображение (например, считанное с помощью OpenCV).

        Returns:
            dict: Словарь, где ключи — это имена признаков, а значения — соответствующие признаки.
        """
        features_dict = {
            key: func(image) for key, func in self.feature_functions.items()
        }
        return features_dict

    def create_df(self):
        features = []
        for i in self.image_paths:
            image = cv2.imread(i)
            image_features = self.extract_features(image)
            features.append(image_features)

        df = pd.DataFrame(features)
        df["label"] = self.labels

        return df
