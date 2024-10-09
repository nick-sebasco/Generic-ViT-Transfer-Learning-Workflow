# Generic-ViT-Transfer-Learning-Workflow

An efficient, push-button workflow for vision transformer transfer learning on whole slide images.

## Project Overview
Transfer learning based on pre-trained machine vision networks is a powerful and highly adopted strategy in digital pathology. In this strategy, a neural network trained to classify natural images (colloquially “cats vs. dogs”) is repurposed to predict categories of interest in biomedical images (e.g., “tumor vs. normal,” “old vs. young,” or “wild type vs. mutant”).

With innovation in the machine vision field stabilizing around large Vision Transformer (ViT) models, this project adapts these pre-trained networks to gigapixel whole-slide images (WSIs). The workflow integrates next-generation file formats (NGFF) and the NextFlow pipeline framework to efficiently leverage ViTs for biological image analysis.

Our goal is to provide a user-friendly experience that abstracts away the complexities of the neural network, PyTorch model construction, and NGFF file handling. As a pilot, we applied a popular pre-trained ViT model to classify brightfield WSIs of mouse kidney tissue based on genotype.

## Features
+ Transfer learning for ViT models on WSIs
+ Efficient patch-based processing for large gigapixel images
+ Automatic region detection and feature extraction
+ Flexible training pipeline with Ignite integration