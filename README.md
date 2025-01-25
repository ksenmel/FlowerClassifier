# FlowerClassifier

<img src=https://github.com/user-attachments/assets/02a34512-1fda-4629-9839-c36544e0f2aa alt="logo" width="100" align="right" style="margin-left: 16px; margin-bottom: 16px">

FlowerClassifier is a Machine Learning project that aims to classify flowers in images into five distinct categories: daisy, dandelion, rose, sunflower, and tulip. The classifier uses Computer Vision techniques to analyze visual features such as color, shape, and texture to distinguish between different types of flowers. No more confusion between sunflowers and daisies — unless you're a bee.

## Run

### Using `python` directly
To start **training** the model, run the following command:

```bash
./.venv/bin/python main.py --mode train -d {dataset_path} -s {dir_to_save_model} -n {model_name} 
```


To predict the type of the flower based on your image, run the following command:

```bash
./.venv/bin/python main.py --mode predict -i {image_path} -m {model_path}
```

## License

Distributed under the MIT License.
See [LICENSE](https://github.com/ksenmel/FlowerClassifier/blob/main/LICENSE) for more information.