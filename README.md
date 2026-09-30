# CNN Object Detection from Scratch with NumPy

This project implements a **Convolutional Neural Network (CNN) from scratch using NumPy** for **image classification and single-object detection**.  
No deep learning frameworks (TensorFlow / PyTorch) are used — all forward and backward passes, optimizers, and layers are manually implemented.

---

## What this project demonstrates

- ✅ Custom CNN built **entirely with NumPy**
- ✅ Manual implementation of:
  - Convolutional layers
  - Max pooling
  - ReLU / LeakyReLU
  - Fully connected layers
  - Dropout
  - Softmax
- ✅ **Object detection** (classification + bounding box regression)
- ✅ Combined loss:
  - Cross-Entropy (classification)
  - Mean Squared Error (bounding boxes)
- ✅ Optimizers:
  - Adam
  - SGD
- ✅ Training progress bars with `tqdm`
- ✅ Visualization of predictions with bounding boxes

---

## 🖼️ Dataset

- **Classes**:
  - Butterfly 🦋
  - Dalmatian 🐕
  - Dolphin 🐬
- Images are resized to **64 × 64**
- Single object per image
- Dataset directory structure:
dataset/
├── butterfly/
├── dalmatian/
└── dolphin/

---

## 🏗️ Model Architecture

Input (3 × 64 × 64)
↓
Conv2D (16 filters) → LeakyReLU → MaxPool
↓
Conv2D (32 filters) → LeakyReLU → MaxPool
↓
Conv2D (64 filters) → LeakyReLU → MaxPool
↓
Flatten
↓
Linear (256) → LeakyReLU → Dropout
↓
Linear (128) → LeakyReLU → Dropout
↓
Linear (NUM_CLASSES + 4)
↓
DetectionSplitLayer


- **Output**:
  - Class probabilities
  - Bounding box coordinates `(x, y, w, h)`

<img src="output/architecture.png" width="600" />

---

## 📉 Loss Function

**DetectionLoss**  
A combination of:
- **Cross-Entropy Loss** for classification
- **Mean Squared Error (MSE)** for bounding box regression

Total Loss = CE + α × MSE


---

## 🚀 Training Configuration

- Epochs: `50`
- Batch size: `32`
- Optimizer: `Adam`
- Learning rate: `0.001`
- Train / Validation split: `80 / 20`

---

## 📊 Evaluation

- Accuracy is computed using **classification predictions**
- Bounding boxes are visualized on test images
- Training & validation loss/accuracy plots included

---

<p align="center">
  <img src="output/combined_training_history_adam.png" width="500" />
  <img src="output/training_history_adam.png" width="600" />
  <img src="test_predictions_adam/9.png" width="500" />
  <img src="output/test_statistics_adam.png" width="500" />
</p>

## Why build this from scratch?

Using PyTorch or TensorFlow is the standard approach for practical deep-learning development. This project takes a different approach for educational purposes: implementing the core components manually makes the underlying mathematics and data flow explicit.

The project was built to better understand:

- How convolutional layers transform images
- How gradients flow backward through a CNN
- How optimizers update parameters
- How classification and localization objectives can be combined
- How a neural network can produce both class predictions and bounding-box coordinates

## Technologies

- Python
- NumPy
- Matplotlib
- tqdm
- Convolutional Neural Networks
- Backpropagation
- Adam / SGD
- Object localization
- Computer Vision

## Future improvements

Possible next steps include:

- Add more rigorous detection metrics such as IoU
- Add a dedicated test split
- Compare the from-scratch implementation with a PyTorch baseline
- Improve bounding-box parameterization and localization loss
- Experiment with data augmentation
- Add reproducible training configuration and random seeds

## 🧪 How to Run

1. Install dependencies:
   ```bash
   pip install numpy tqdm matplotlib

2. Set dataset path in the script:

    dataset_root = "path/to/dataset"

You can reload weights before evaluation:

    model.load_weights("model_weights_with_adam.npz")





