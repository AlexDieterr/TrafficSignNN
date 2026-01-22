# Traffic Sign Classification (CNN)

This project is a convolutional neural network (CNN) built to classify traffic sign images into **43 different categories** using the **German Traffic Sign Recognition Benchmark (GTSRB)** dataset. The goal of the project was to build an accurate image classification model and deploy it in a **real, interactive web application** rather than keeping it as a notebook-only project.

The final result is a trained deep learning model that can classify traffic signs with high accuracy and is exposed through a FastAPI backend, which is then consumed by an Angular frontend via a drag-and-drop demo.

Although this model was trained locally, the pipeline is structured to mirror a production cloud workflow. Data ingestion, preprocessing, training, and inference are clearly separated, allowing each stage to run as an independent batch job in a distributed analytics platform such as Databricks. Preprocessing and training logic is stateless and reproducible, making it suitable for scalable execution on cloud compute.

---

## Project Overview

- Built a CNN from scratch using TensorFlow/Keras
- Trained and evaluated the model on the GTSRB dataset
- Achieved ~99% validation accuracy across 43 traffic sign classes
- Deployed the trained model behind a FastAPI REST API
- Integrated the API into a live Angular website with an interactive demo

Users can generate random traffic sign images from the dataset, drag an image into the prediction box, and see the model’s predicted label and confidence in real time.

---

## Dataset

- **German Traffic Sign Recognition Benchmark (GTSRB)**
- 43 classes of traffic signs
- Images resized to **32×32 RGB**
- Pixel values normalized to `[0, 1]`
- Dataset split into training and validation sets using stratified sampling

---

## Model Architecture

The CNN architecture consists of:
- 3 convolutional layers with ReLU activation
- Max pooling after each convolution block
- A fully connected dense layer with dropout to reduce overfitting
- Softmax output layer for multi-class classification

This architecture balances performance and simplicity while remaining fast enough for real-time inference.

---

## Model Performance

- Validation accuracy: **~99%**
- Strong precision and recall across almost all classes
- Very few misclassifications, mostly between visually similar signs
- Confusion matrix confirms consistent performance across classes

---

## Live Demo

The model is deployed and accessible through a live website:

- Users can generate sample images from the dataset
- Images can be **dragged and dropped** into the prediction area
- The model returns:
  - Predicted traffic sign label
  - Confidence score for the prediction

This demonstrates how the model could be used in a real-world application rather than only in a notebook.

---

## Tech Stack

**Machine Learning**
- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib

**Backend**
- FastAPI
- Uvicorn
- Python Multipart
- KaggleHub (dataset access)

**Frontend**
- Angular
- TypeScript
- HTML / CSS

**Deployment**
- Render (API hosting)
- Angular web application

---

## Repository Structure
