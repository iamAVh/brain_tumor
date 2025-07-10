# Brain Tumor Image Classification

This project implements a brain tumor classification system using a deep learning model (VGG19) and a Flask web application. Users can upload MRI brain images and receive predictions indicating the presence or absence of a brain tumor.

---

## Features

- Classifies MRI brain images as **Tumorous** or **Non-tumorous**
- Uses pre-trained VGG19 model with custom classification layers
- Applies data augmentation to balance and expand the dataset
- Crops brain region from MRI images to focus on relevant features
- Provides a simple Flask web interface for image upload and prediction

---

## Dataset

- Brain MRI images categorized into `yes` (tumorous) and `no` (non-tumorous)
- Original dataset is imbalanced (~155 tumorous, ~98 non-tumorous images)
- Data augmentation used to balance dataset (~1085 tumorous and ~979 non-tumorous after augmentation)
- Dataset provided as `archive.zip`, extracted to `brain_tumor_dataset/`

---

## Model Architecture

- Base model: Pre-trained **VGG19** with `include_top=False`
- Custom layers added on top: Flatten, Dense layers with ReLU, Dropout (0.2), and final Dense layer with softmax for binary classification
- Model weights loaded from `vgg_unfrozen.h5`

---

## Data Preprocessing & Augmentation

- Files renamed systematically for consistency (e.g., Y_1.jpg, N_1.jpg)
- Augmentation includes rotation, shifts, shear, brightness changes, flips
- Custom function crops brain region from images by isolating the largest contour
- Images resized to 240x240 pixels and normalized

---

## Web Application

- Built with **Flask**
- `/` route renders upload page (`index.html`)
- `/predict` route handles image uploads, preprocesses images, and returns prediction
- Frontend provides intuitive image upload and displays prediction results dynamically

---

