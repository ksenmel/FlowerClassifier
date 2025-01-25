# FlowerClassifier

<img src=https://github.com/user-attachments/assets/02a34512-1fda-4629-9839-c36544e0f2aa alt="logo" width="100" align="right" style="margin-left: 16px; margin-bottom: 16px">

**FlowerClassifier** is a Machine Learning project that aims to classify flowers :tulip: in images into five distinct categories: _daisy_, _dandelion_, _rose_, _sunflower_, and _tulip_.
The classifier uses Computer Vision techniques to analyze visual features such as color, shape, and texture to distinguish between different types of flowers.
No more confusion between sunflowers and daisies — unless you're a bee :bee:.

## Setup

```bash
make install-deps # or just "make"
```

## Run

### Using `python` directly

To start **training** the model, run the following command:

```bash
./.venv/bin/python main.py --mode train -d {dataset_path} -s {dir_to_save_model} -n {model_name} 
```

This command does the following:
- **Dataset loading**: Loads a flower dataset from a specified path using `FlowerDataset` class (you can use your own dataset or download it [here](https://www.kaggle.com/datasets/alxmamaev/flowers-recognition/data))
- **Data processing**: Creates a DataFrame from the dataset and splits it into training and testing sets using `train_test_split`
- **Model training**: Initializes a classifier (`Classifier`), fits it on the training data, and evaluates it on the test data
- **Saving the model**: After training and evaluation, it saves the trained model to the specified path

To **predict** the type of the flower based on your image, run the following command:

```bash
./.venv/bin/python main.py --mode predict -i {image_path} -m {model_path}
```

This command does the following:
- **Model loading**: Loads a trained classifier model
- **Image processing**: Reads the image from the specified path and computes features using predefined functions (from `feature_functions`)
- **Prediction**: Makes a prediction based on the extracted features and prints out the predicted class
- **Probability distribution**: Prints the probability distribution for each flower class

#### Command-line arguments:

- `--mode`: Specifies whether to run in train or predict mode
- `-i` (`--img_path`): Path to the input image to classify (only used in predict mode)
- `-m` (`--model_path`): Path to the trained model (only used in predict mode)
- `-d` (`--dataset_path`): Path to the dataset for training (only used in train mode)
- `-s` (`--path_to_save`): Path to save the trained model (only used in train mode)
- `-n` (`--name`): The name for the model to be saved (only used in train mode)

## License

Distributed under the MIT License.
See [LICENSE](https://github.com/ksenmel/FlowerClassifier/blob/main/LICENSE) for more information.
