# Emotion Detection with Transfer Learning

## Overview

This project implements an emotion detection system using transfer learning with pre-trained deep learning models. The system allows users to upload images, and the model predicts the dominant emotion expressed in the image. The implementation includes model training, evaluation, and deployment using Gradio.

## Features

- Utilizes pre-trained deep learning models (VGG16 and ResNet50) for feature extraction.
- Implements data augmentation and class weighting to handle imbalanced datasets.
- Evaluates model performance using accuracy, confusion matrix, and classification report.
- Deploys the trained model using Gradio for real-time emotion detection from images.

## Sequence of Implementation

1. **Data Preparation**: Organize the dataset into training and testing directories.

2. **Model Training**:
   - **Custom_CNN_From_Scratch**:
     - Implements a custom CNN architecture from scratch.
     - Defines and trains the model without pre-trained weights.
     - Evaluates the model's performance.

   - **Custom_CNN_With_Augmentation**:
     - Implements a custom CNN architecture with data augmentation.
     - Incorporates data augmentation techniques such as rotation, zoom, and horizontal flip.
     - Trains the model and evaluates its performance.

   - **VGG16 Transfer Learning**:
     - Utilizes the VGG16 architecture with transfer learning.
     - Implements data augmentation and class weighting.
     - Trains the model and evaluates performance.

   - **ResNet50 Transfer Learning**:
     - Utilizes the ResNet50 architecture with transfer learning.
     - Implements data augmentation and class weighting.
     - Trains the model and evaluates performance.

3. **Model Deployment**:
   - Implements a Gradio interface for real-time emotion detection from uploaded images.

## Getting Started

### Prerequisites

- Python 3.x
- TensorFlow
- keras
- Gradio
- Other dependencies (specified in requirements.txt)

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/sahiltambe/Deep-Learning-for-Real-Time-Emotion-Recognition.git
   

### Install dependencies:

pip install -r requirements.txt


### Usage

Train the models by running the appropriate scripts (train_vgg16.py and train_resnet50.py).
Evaluate the models using the evaluation scripts.
Deploy the trained model using Gradio (deploy_gradio.py).

### Results

## VGG16 Transfer Learning:

- Training Accuracy: 55.93%
- Validation Accuracy: 55.00%
- Final Test Accuracy: 24.00%

## ResNet50 Transfer Learning:

- Training Accuracy: 62.61%
- Validation Accuracy: 60.80%
- Final Test Accuracy: 61%

## Future Improvements

- Experiment with different pre-trained models and architectures.
- Fine-tune hyperparameters for better performance.
- Explore ensemble methods for combining multiple models.

Acknowledgements
- The dataset used in this project is sourced from [[insert source](https://www.kaggle.com/datasets/msambare/fer2013)].
- Gradio library for providing an intuitive interface for model deployment.

## Contributors & Contributing

**Sahil Tambe**
Contributions are welcome! Please fork the repository and submit a pull request with your enhancements.

## Contact

For questions or inquiries about the project, feel free to contact the project maintainers at [sahiltambe1996@gmail.com](mailto:sahiltambe1996@gmail.com).