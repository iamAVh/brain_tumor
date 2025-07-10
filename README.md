Brain Tumor Image Classification
This project implements a Brain Tumor Image Classification system using a deep learning model (VGG19) and a Flask web application for user interaction. The system allows users to upload MRI brain images and receive a prediction on whether the image indicates the presence of a brain tumor.

Table of Contents
Features

Dataset

Model Architecture

Data Preprocessing and Augmentation

Web Application

Installation

Usage

Project Structure

Contributing

License

Contact

Features
Image Classification: Classifies MRI brain images into "Tumorous" or "Non-tumorous" categories.

Pre-trained Model: Utilizes the VGG19 convolutional neural network as a base model for transfer learning.

Data Augmentation: Employs various image augmentation techniques to enhance the dataset and improve model robustness.

Image Cropping: Includes a preprocessing step to automatically crop the brain region from the MRI images, focusing the model on relevant features.

Flask Web Interface: Provides a simple and intuitive web interface for users to upload images and view predictions.

Dataset
The project uses a dataset of brain MRI images, which is expected to be structured with subdirectories for 'yes' (tumorous) and 'no' (non-tumorous) cases. The initial dataset has an imbalance, with approximately 155 tumorous images and 98 non-tumorous images.

To address this imbalance and increase the dataset size for better model training, data augmentation techniques are applied. After augmentation, the dataset is balanced, with around 1085 positive (tumorous) samples and 979 negative (non-tumorous) samples.

The dataset is expected to be provided in a archive.zip file, which will be extracted into a brain_tumor_dataset directory.

Model Architecture
The core of the classification system is built upon the VGG19 pre-trained convolutional neural network. The include_top parameter is set to False to remove the classification layers of the original VGG19 model, allowing us to add custom classification layers suitable for our specific task.

The custom layers added on top of the VGG19 base include:

A Flatten layer to convert the output of the convolutional base into a 1D feature vector.

Two Dense layers with relu activation for feature learning.

A Dropout layer (with a rate of 0.2) for regularization to prevent overfitting.

A final Dense layer with softmax activation for binary classification (2 classes: "No Brain Tumor" and "Yes Brain Tumor").

The model weights are loaded from vgg_unfrozen.h5, indicating that the VGG19 layers might have been fine-tuned during training.

Data Preprocessing and Augmentation
To prepare the images for the model and improve its performance, the following steps are performed:

Renaming Files: Files in the 'yes' and 'no' directories are systematically renamed (e.g., Y_1.jpg, N_1.jpg).

Data Augmentation: ImageDataGenerator from tensorflow.keras.preprocessing.image is used to create augmented samples. Techniques include:

Rotation (up to 10 degrees)

Width and Height Shifts (up to 10%)

Shear transformations (up to 10%)

Brightness adjustments (range 0.3 to 1.0)

Horizontal and Vertical Flips

fill_mode set to 'nearest' for filling new pixels.
This process significantly increases the number of training samples and helps the model generalize better to unseen data.

Brain Tumor Cropping: A custom function crop_brain_tumor is implemented to:

Convert images to grayscale.

Apply Gaussian Blur for noise reduction.

Perform thresholding, erosion, and dilation to isolate the brain region.

Find contours and extract the largest contour to identify the brain.

Crop the image to the bounding box of the identified brain region. This helps the model focus on the relevant part of the image, reducing noise and irrelevant features.

Resizing: Input images are resized to (240, 240) pixels to match the input shape required by the VGG19 model.

Normalization: Images are converted to NumPy arrays and expanded dimensions to fit the model's expected input format.

Web Application
The project includes a simple web application built with Flask that provides an interface for real-time brain tumor classification.

Endpoints
/: Renders the index.html template, which is the main page for uploading images.

/predict: Handles image uploads via POST requests.

It saves the uploaded image temporarily.

Calls the getResult function to preprocess the image and get a prediction from the loaded model.

Uses get_className to convert the numerical prediction into a human-readable label ("No Brain Tumor" or "Yes Brain Tumor").

Returns the prediction result to the user.

Installation
To set up and run this project locally, you will need to:

Clone the repository.

Create a virtual environment and activate it.

Install the required Python packages (Flask, TensorFlow, OpenCV, NumPy, Pillow, imutils).

Download the dataset (archive.zip) and ensure it's in the project's root directory for extraction.

Place the pre-trained model weights (vgg_unfrozen.h5) in the same directory as your app.py file.

Create an uploads directory to store temporary uploaded images.

Create a templates directory and an index.html file inside it for the web interface.

Usage
Start the Flask application.

Open your web browser and navigate to the local server address (e.g., http://127.0.0.1:5000/).

Upload an MRI brain image using the provided interface.

Click "Predict" to get the classification result.

Project Structure
The project typically has the following structure:

app.py: The main Flask web application file for handling predictions.

Advance DL Project Brain Tumor Image Classification.ipynb: The Jupyter notebook containing the model training, data augmentation, and preprocessing steps.

archive.zip: The zipped dataset.

vgg_unfrozen.h5: The pre-trained VGG19 model weights file.

uploads/: A directory for temporarily storing uploaded images.

templates/: A directory containing the index.html file for the web interface.

Contributing
Contributions are welcome! If you have suggestions for improvements, bug fixes, or new features, please feel free to:

Fork the repository.

Create a new branch.

Make your changes.

Commit your changes.

Push to the branch.

Open a Pull Request.

License
This project is open-source and available under the MIT License. (You might want to create a LICENSE file in your repository if you haven't already).

Contact
If you have any questions or feedback, please feel free to reach out.
