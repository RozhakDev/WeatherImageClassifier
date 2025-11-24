# Dataset

## Source

This project uses the [Multiclass Weather Dataset on Kaggle](https://www.kaggle.com/datasets/vijaygiitk/multiclass-weather-dataset) (`vijaygiitk/multiclass-weather-dataset`). The data is retrieved through the Kaggle API and is not stored in this repository.

## Scope and Preparation

The experiment uses five classes: `shine`, `cloudy`, `foggy`, `sunrise`, and `rainy`. Images are split into training, validation, and test sets at 60:20:20 with `seed=42`. Each image is resized to 224 × 224 pixels and processed with MobileNetV2's `preprocess_input`. Random augmentation is applied to training images only.

## Access and Licensing

Use of the dataset is subject to the terms stated on its Kaggle page. Review those terms before redistribution or commercial use. Do not commit Kaggle API credentials (`kaggle.json`) to version control.
