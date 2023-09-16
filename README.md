# Single Cell Segmentation using UNet

This repository contains code for performing single cell segmentation using the UNet architecture. It includes data preprocessing, model training, and evaluation. The code is designed to work with PyTorch and is well-documented to help you get started with your own single cell segmentation project.

## Overview

The code provided here is a complete pipeline for performing single cell segmentation. It includes the following components:

1. **Data Preparation**: The dataset is organized into train, validation, and test sets. Image and mask pairs are loaded and preprocessed for training and evaluation.

2. **Model Architecture**: The UNet architecture is used for single cell segmentation. It consists of an encoder-decoder network that learns to segment cells from input images.

3. **Training**: The model is trained using the training dataset. The training loop includes loss computation, backpropagation, and optimization.

4. **Validation**: The model's performance is evaluated on a validation dataset. Metrics such as loss, DICE score, accuracy, and confusion matrix are computed.

5. **Testing**: The trained model is tested on a separate test dataset to assess its generalization performance.

6. **Results Visualization**: Visualization of input images, ground truth masks, predicted masks, and evaluation metrics.

## Requirements

Make sure you have the following libraries installed:

- PyTorch
- NumPy
- OpenCV
- Matplotlib
- Seaborn
- scikit-learn

You can install these libraries using `pip`:

```bash
pip install torch numpy opencv-python matplotlib seaborn scikit-learn
```

## Usage

1. Clone this repository:

   ```bash
   git clone <repository_url>
   cd single-cell-segmentation
   ```

2. Execute the provided Jupyter Notebook or integrate the code into your own project.

3. Customize the code to fit your specific dataset and requirements. Be sure to update the data paths, model architecture, hyperparameters, and evaluation metrics as needed.

4. Run the code to train, validate, and test your single cell segmentation model.

## Results

The code provides detailed results for training, validation, and testing. You can visualize the learning curve, including loss and DICE score, to assess the model's performance during training. Additionally, the confusion matrix and accuracy are provided for evaluating the model's segmentation quality.

## License

This code is provided under the MIT License. Feel free to use and modify it for your own projects.

## Acknowledgments

This code is based on various sources and references, including lectures and assignments from relevant courses. Special thanks to the contributors and maintainers who have helped shape this project.

**Note**: Please ensure you have the required dataset and permissions to use it when working with real data. This code assumes that you have access to the single cell segmentation dataset, and the paths to the data are correctly configured.

If you encounter any issues or have questions, feel free to open an issue in the repository or seek assistance from the community. Good luck with your single cell segmentation project!
