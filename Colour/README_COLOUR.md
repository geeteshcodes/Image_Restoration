Image Colorizer using Deep Learning
Overview

This project implements an automatic image colorization system that converts grayscale images into color images using a deep learning model.
The model learns visual and semantic features from grayscale inputs and predicts realistic color information.

Core Idea
Colorization is performed using the LAB color space, which separates luminance from color information.
L channel represents lightness (grayscale information)
A and B channels represent color information
The model predicts color channels while preserving the original image structure.

Colorization Pipeline
Input color images are converted from RGB to LAB color space
The L channel is extracted and used as model input
The model predicts the A and B channels
Predicted L, A, B channels are merged
The merged LAB image is converted back to RGB
This approach ensures structural consistency and stable color predictions.

Model Approach
Convolutional neural network
AutoEncoders
Learns mapping: L channel → A and B channels
Fully convolutional architecture
Trained end-to-end on paired grayscale and color images

Features
Automatic grayscale to color conversion
LAB color space based learning
Preserves edges and textures
Produces visually plausible colors
No manual color hints required

Loss Function
Weighted Sum of Mean Absolute Error (MAE) and SSIM
Loss is computed between predicted and ground truth A and B channels
These losses provide stable gradients and smooth color outputs.

Dataset
CelebA

Results
Generates realistic colorized images
Maintains image structure and brightness
Performs well on common objects and scenes
Color accuracy depends on dataset diversity and semantic complexity.

Limitations
Some colors may be ambiguous or inaccurate
Rare objects may be colorized incorrectly
Cannot guarantee true original colors
