Image Denoiser using Residual CNN (RCNN)

Overview
This project implements an image denoising system using a Residual Convolutional Neural Network (RCNN).
The model removes noise from images by learning the residual noise instead of directly predicting the clean image, resulting in better convergence and detail preservation.

Core Concept
The network learns to predict the noise present in a noisy image.
Noise = Noisy Image − Clean Image
The denoised image is obtained as:
Denoised Image = Noisy Image − Predicted Noise
Residual learning simplifies training and improves denoising performance.

Features
Residual CNN based denoising
Preserves edges and fine textures
Supports grayscale and RGB images
No pooling layers to maintain spatial resolution
Effective for synthetic noise such as Gaussian noise

Model Architecture
Multiple convolutional layers with 3×3 kernels
ReLU activation functions
Residual (skip) connections
Final convolution layer predicts noise
Fully convolutional network
Pooling layers are avoided to prevent loss of spatial details.

Clean images act as ground truth
Noise is added programmatically to create noisy images
Training data consists of (noisy image, clean image) pairs

datasets:
CelebA

Training
Training is performed inside the Jupyter notebook.

Steps:
Load clean images
Add synthetic noise
Train the RCNN to predict noise
Save the trained model
Open and run:
denoiser_rcnn.ipynb

Testing / Inference
Load the saved RCNN model from rcnn_model/saved_model/
Pass a noisy image to the model
Subtract predicted noise from the input image
Inference is demonstrated in the notebook.

Loss Function
Mean Squared Error (MSE) is used between the predicted noise and the actual noise.
MSE provides stable optimization and works well for image restoration tasks.

Technologies Used
Python
TensorFlow
NumPy
Jupyter Notebook

Results
Clear reduction of noise
Improved edge and texture preservation
Consistent denoising across different noise levels
