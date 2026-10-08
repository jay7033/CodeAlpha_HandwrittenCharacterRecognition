# CodeAlpha Handwritten Character Recognition

## Overview

This project is developed as part of the CodeAlpha Machine Learning Internship.

The project uses a Convolutional Neural Network (CNN) to recognize handwritten digits from images.

## Objective

The objective of this project is to build a machine learning system that can identify handwritten digits from 0 to 9.

## Dataset

The project uses the MNIST handwritten digit dataset.

The dataset contains:

- 60,000 training images
- 10,000 testing images
- Image size: 28 × 28 pixels
- 10 classes: digits 0 to 9

## Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Pillow
- Streamlit
- CNN

## Machine Learning Approach

The images are first normalized and reshaped into the format required by the CNN.

The CNN architecture contains:

- Convolutional Layer
- Max Pooling Layer
- Convolutional Layer
- Max Pooling Layer
- Flatten Layer
- Dense Layer
- Dropout Layer
- Output Layer

The final output contains 10 classes representing digits from 0 to 9.

## Project Structure

```text
CodeAlpha_HandwrittenCharacterRecognition/
│
├── data/
├── models/
│   └── handwritten_digit_model.keras
│
├── app.py
├── train_model.py
├── requirements.txt
└── README.md