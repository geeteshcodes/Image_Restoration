AI-Based Image Restoration & Colorization System

Overview
This project is a modular deep learning image processing system designed to restore, enhance, and colorize images using task-specific neural network architectures.
Each module is designed based on the nature of the problem it solves—corruption removal, enhancement, or information generation—rather than relying on a single generic model.
The system supports large images, arbitrary aspect ratios, and efficient inference through patch-based processing and image stitching.

Supported Modules
1. Image Denoiser
Removes stochastic noise while preserving fine image details.
Uses a Residual CNN trained to predict noise
Clean image is reconstructed by subtracting predicted noise
Optimized for realistic noise levels
Preserves edges and textures without hallucination

Task type: Corruption removal
Model type: Residual CNN (noise prediction)

2. Image Deblurrer
Restores sharpness from motion blur or defocus blur.
Learns to reverse convolutional blur patterns
Uses multi-scale receptive fields and skip connections
Applied after denoising to avoid noise amplification

Task type: Inverse degradation
Model type: Residual / multi-scale CNN

3. Image Sharpener
Enhances edges and local contrast after restoration.
Lightweight residual refinement
Improves perceptual clarity without inventing textures
Optional, user-controlled stage

Task type: Enhancement
Model type: Lightweight residual CNN

4. Grayscale Image Colorizer
Adds realistic color to grayscale images using semantic understanding.
Operates in LAB color space
Takes L (luminance) channel as input
Predicts ab (chrominance) channels
Recombines original L with predicted ab for final RGB output
Preserves structure and lighting by design

Task type: Information generation
Model type: Conditional Autoencoder

Image Stitching & Large Image Support
The system supports high-resolution images and arbitrary aspect ratios through a patch-based inference strategy.
Patch Processing
Large images are divided into overlapping patches (e.g., 128×128 or 256×256)
Each patch is processed independently
Overlapping regions reduce boundary artifacts

Stitching Strategy
Processed patches are placed back into their original spatial positions
Overlapping areas are merged using weighted blending
Final image is reconstructed at original resolution
This enables:
  Memory-efficient inference
  Mobile and edge deployment
  No retraining required for larger images

System Pipeline
Input Image
   ↓
[Denoiser]
   ↓
[Deblurrer]
   ↓
[Sharpener] (optional)
   ↓
[Colorizer] (if grayscale)
   ↓
Final Output Image


Each module:
Has a single responsibility
Can be enabled or disabled independently
Can be replaced without affecting the rest of the system

Key Design Principles
Residual learning for corruption removal
Conditional autoencoders for missing information generation
LAB color space for stable and realistic colorization
Modular architecture for scalability and maintainability
Patch-based inference for large image handling

Applications
Photo restoration
Old and degraded image enhancement
Grayscale image colorization
Mobile and desktop image enhancement pipelines
Large-resolution image processing

Summary
This project demonstrates a system-level, architecture-aware approach to image restoration and colorization.
Each task is solved using an appropriate model design, ensuring realistic results, structural fidelity, and scalability to real-world image sizes.
