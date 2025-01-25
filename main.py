import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from classification.classifier import Classifier
from classification.dataset import FlowerDataset

#
# dataset = FlowerDataset(path="/Users/kseniia/Desktop/test")
#
# print(f"Number of samples in the dataset: {len(dataset)}")
# print(f"In the dataset there are {len(dataset.classes)} classes: {dataset.classes}")
# print(f"Index: {dataset[0]}")
#
# df = dataset.create_df()
# print(df)
#
# df.to_csv('test.csv', index=False)


# df = pd.read_csv("filename.csv")
#
classifier = Classifier()
# classifier.fit(df)
# predictions = classifier.predict(df)
#
# print(predictions)
#
# classifier.download('models')

classifier.load("models/v1FlowerClassifier_model.pkl")
new_df = pd.read_csv("tests/test.csv")

predictions = classifier.predict(new_df)
eval = classifier.evaluate(new_df)

print("Accuracy:", eval)
print("Predictions:", predictions)

probs = classifier.predict_proba(new_df)

# Пример: показываем вероятности для первого изображения
print("Распределение вероятностей для первого изображения:")
print(
    probs[14]
)  # Вероятности для всех классов (например, daisy, dandelion, rose, sunflower, tulip)
