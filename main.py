import argparse
import cv2

import pandas as pd
from sklearn.model_selection import train_test_split

from classification.classifier import Classifier
from classification.dataset import FlowerDataset
from classification.features import feature_functions


def train_mode(args):
    dataset = FlowerDataset(path=args.dataset_path)

    print(
        f"Number of samples in the dataset: {len(dataset)}\nThere are {len(dataset.classes)} classes: {dataset.classes}\n"
    )

    df = dataset.create_df()

    X = df.drop(columns=["label"]).values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, shuffle=True
    )

    classifier = Classifier()

    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)
    eval = classifier.evaluate(y_test, predictions)

    print(predictions)
    print(eval)

    classifier.download(args.path_to_save)


def predict_mode(args):
    classifier = Classifier()
    classifier.load(args.model_path)

    image = cv2.imread(args.img_path)

    features = {key: func(image) for key, func in feature_functions.items()}
    df = pd.DataFrame([features])

    X = df.values

    class_names = ["daisy", "dandelion", "rose", "sunflower", "tulip"]

    predictions = classifier.predict(X)
    print(f"I think the flower on the image is {class_names[predictions[0]]}!")

    probs = classifier.predict_proba(X)

    print(
        f"Classes probability distribution for image: \n{probs[0][0]} daisy\n{probs[0][1]} dandelion\n{probs[0][2]} rose\n{probs[0][3]} sunflower\n{probs[0][4]} tulip\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse parameters for the flower classification application"
    )
    parser.add_argument(
        "--mode",
        help="Mode of the application: 'train' for training, 'predict' for evaluating",
        choices=["train", "predict"],
        default="predict",
    )
    parser.add_argument(
        "-i",
        "--img_path",
        help="Path to the input image with flower to classify",
        default="tests/0.jpeg",
    )
    parser.add_argument("-m", "--model_path", help="Path to classification model")
    parser.add_argument(
        "-d",
        "--dataset_path",
        help="Path to the dataset for training",
        default="models/v1FlowerClassifier_model.pkl",
    )
    parser.add_argument(
        "-s", "--path_to_save", help="path to save classification model"
    )

    args = parser.parse_args()

    # Вызываем нужную функцию в зависимости от режима
    if args.mode == "train":
        train_mode(args)
    elif args.mode == "predict":
        predict_mode(args)
