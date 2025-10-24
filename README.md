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

## Environment
You can run the workflow using the following singularity container: 

```
/projects/korstanje-lab/singularity-containers/vit_tiatoolbox.sif
```

## TODO

+ [x] Diff ViTScanner:ScanPrep with Generic-ViT:ScanPrep 
+ [x] Diff ViTScanner:ViTScanner with Generic-ViT:ViTScanner 
+ [x] Diff ViTScanner:ScanAggregators with Generic-ViT:ScanAggregators 
+ [x] Diff ViTScanner:csvCombiner with Generic-ViT:csvCombiner 
+ [x] Diff ViTScanner:vitscanner.nf with Generic-ViT:vitscanner.nf 
+ [x] Diff ViTScanner:main.nf with Generic-ViT:main.nf


+ [] scan test with new workflow repo.  create a small test dataset of 2 images.  Run once on old workflow and
    once with the new one.
+ [ ] New requirement for running workflow is two separate configs: scanprep & vitscanner

### Low Priority
+ [ ] keep only conf or local conf
+ [ ] add basic test for scanning code
+ [ ] make sure stain normalization tests run
+ [ ] make sure training tests run
