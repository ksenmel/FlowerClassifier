import os
from PIL import Image
import numpy as np
import cv2
import pandas as pd
from torch.utils.data import Dataset

from classification.features import feature_functions


class FlowerDataset(Dataset):
    """
    A custom dataset class for loading flower images and extracting their features.

    Args:
        path (str): The directory path where flower images are stored, organized into subdirectories
                    by class (e.g., daisy, sunflower, etc.).

    Attributes:
        path (str): Path to the dataset.
        image_paths (list): List of paths to all images in the dataset.
        labels (list): Corresponding list of labels for each image.
        classes (list): List of unique flower classes.
        feature_functions (dict): Dictionary of feature extraction functions to apply to images.

    Methods:
        __len__: Returns the total number of images in the dataset.
        __getitem__: Retrieves an image and its label by index.

        extract_features: Extracts a set of predefined features from an image.
        create_df: Creates a DataFrame containing all extracted features and their corresponding labels.
    """

    def __init__(self, path):
        """
        Initializes the dataset by loading image paths and assigning labels based on the subdirectories.

        Args:
            path (str): Directory path to the flower image dataset.
        """
        self.path = path
        self.image_paths = []
        self.labels = []
        self.classes = sorted(
            os.listdir(path)
        )  # List of flower classes (subdirectories)

        # Iterating over each class and image in the subdirectories
        for label, class_name in enumerate(self.classes):
            class_dir = os.path.join(path, class_name)
            if os.path.isdir(class_dir):
                for file_name in os.listdir(class_dir):
                    if file_name.endswith((".jpg", ".jpeg", ".png")):
                        self.image_paths.append(os.path.join(class_dir, file_name))
                        self.labels.append(label)

        # Dictionary of feature functions for feature extraction
        self.feature_functions = feature_functions

    def __len__(self):
        """
        Returns the total number of images in the dataset.

        Returns:
            int: Number of images in the dataset.
        """
        return len(self.image_paths)

    def __getitem__(self, idx):
        """
        Retrieves an image and its label by the given index.

        Args:
            idx (int): The index of the image to retrieve.

        Returns:
            tuple: A tuple containing the image and its corresponding label.
        """
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(image_path)

        return image, label

    def extract_features(self, image: np.ndarray) -> dict:
        """
        Extracts predefined features from the input image.

        Args:
            image (np.ndarray): Image to extract features from (e.g., read using OpenCV).

        Returns:
            dict: A dictionary containing the extracted features, where keys are feature names
                  and values are the corresponding feature values.
        """
        features_dict = {
            key: func(image) for key, func in self.feature_functions.items()
        }
        return features_dict

    def create_df(self):
        """
        Creates a DataFrame of extracted features for all images in the dataset.

        Iterates over all image paths, extracts features for each image, and stores them
        in a pandas DataFrame along with their corresponding labels.

        Returns:
            pd.DataFrame: A DataFrame where each row represents an image and its extracted features,
                          and the last column contains the image label.
        """
        features = []
        for i in self.image_paths:
            image = cv2.imread(i)
            image_features = self.extract_features(image)
            features.append(image_features)

        df = pd.DataFrame(features)
        df["label"] = self.labels

        return df
