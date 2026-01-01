Waste Classification using Image Classification (CNN)
 Project Overview

This project is a computer vision image classification application that classifies uploaded waste images into three categories:

Organic

Recyclable

Non-Recyclable

The model is trained using a Convolutional Neural Network (CNN) and deployed using Streamlit, where users can upload an image and get the predicted waste category along with a confidence score.

 Problem Statement

Proper waste segregation is a major challenge in waste management systems. Incorrect segregation leads to environmental pollution and inefficient recycling.

This project aims to assist waste segregation by automatically classifying waste images into appropriate categories using deep learning.

Solution Approach

Used supervised image classification

Trained a CNN from scratch using TensorFlow/Keras

Used folder-based datasets for training and validation

Implemented data augmentation to improve generalization

Built a Streamlit web app for image upload and prediction

 Dataset

Dataset sourced from Kaggle

Images are divided into:

train/

val/

Classes:

Organic

Recyclable

Non-Recyclable

A subset of the dataset was used for training to ensure laptop-friendly execution

Project Structure
Waste Classification/
├── dataset/
│   ├── train/
│   └── val/
├── model.py
├── app.py
├── waste_classifier.h5
├── class_names.json
└── requirements.txt

 Technologies Used

Python

TensorFlow / Keras

NumPy

Pillow (PIL)

Streamlit

 How to Run the Project
1️Create virtual environment
python -m venv venv
source venv/bin/activate

2️ Install dependencies
pip install -r requirements.txt

3️ Train the model
python model.py

4 Run the application
streamlit run app.py

 Application Features

Upload image (JPG / PNG)

Real-time prediction

Displays predicted class

Shows confidence score

Simple and beginner-friendly UI

 Model Performance

Training Accuracy: ~60%

Validation Accuracy: ~58%




