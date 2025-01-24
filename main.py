from classification.dataset import FlowerDataset

dataset = FlowerDataset(path="/Users/kseniia/Desktop/flowers")

print(f"Number of samples in the dataset: {len(dataset)}")
print(f"In the dataset there are {len(dataset.classes)} classes: {dataset.classes}")
print(f"Index: {dataset[0]}")

print(dataset.create_df())
