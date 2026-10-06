## Eye Disease Classification – Multi-Model Ensemble

This repository contains the implementation of a deep-learning-based multi-model ensemble system for automated classification of eye diseases from retinal (fundus) images. The model distinguishes 10 disease categories—including diabetic retinopathy, glaucoma, myopia, and healthy eyes—using a curated dataset of 16,534 images.

***

## Overview

Early and accurate detection of retinal diseases is essential for preventing irreversible vision loss. This project investigates how complementary deep architectures and ensemble learning can improve classification reliability in medical imaging, especially under realistic data and annotation constraints.

The system combines three strong backbone networks—EfficientNetV2-M with CBAM attention, ConvNeXt-Base, and Vision Transformer (ViT-Base/16)—and fuses their predictions using weighted soft voting to enhance robustness and generalization.

***

## Features

- **Multi-model ensemble**
  - EfficientNetV2-M + CBAM attention
  - ConvNeXt-Base
  - Vision Transformer (ViT-Base/16)
- **Training strategies**
  - Transfer learning from ImageNet
  - Class balancing and targeted data augmentation
  - Differential learning rates, label smoothing
  - Mixed-precision training for faster GPU utilization
- **Evaluation**
  - Accuracy, precision, recall, macro F1-score
  - Confusion matrices and per-class classification reports
  - Weighted soft-voting ensemble for improved stability

***

## Dataset

- **Size:** 16,534 retinal images  
- **Classes:** 10 disease categories (e.g., diabetic retinopathy, glaucoma, myopia, healthy, etc.)  
- **Preprocessing:** Resizing, normalization, artifact removal, and class-balanced splitting into train/validation/test sets.

*(If you have a specific dataset name or link, you can add it here.)*

***

## Model Architecture

The ensemble consists of:

1. **EfficientNetV2-M + CBAM**  
   - Efficient feature extraction with channel and spatial attention for fine-grained retinal patterns.
2. **ConvNeXt-Base**  
   - Modern CNN backbone with transformer-inspired design for strong representation learning.
3. **Vision Transformer (ViT-Base/16)**  
   - Global self-attention over image patches to capture long-range dependencies in fundus images.

Final predictions are obtained by weighted averaging of the three models’ softmax outputs.

***

## Results

*(Fill in with your actual numbers when ready. Example structure below.)*

- Top-1 accuracy: **XX.X%**
- Macro F1-score: **XX.X**
- Best-performing class: **[Class name]** (F1: XX.X)
- Most challenging class: **[Class name]** (F1: XX.X)

Detailed confusion matrices and per-class metrics are available in the `results/` directory.

***

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/eye-disease-classification.git
cd eye-disease-classification

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

Example `requirements.txt` entries:

```text
torch
torchvision
opencv-python
albumentations
numpy
pandas
matplotlib
seaborn
scikit-learn
```

***

## Usage

### Directory structure (example)

```text
data/
  ├── train/
  ├── val/
  └── test/
models/
  ├── efficientnet_cbam.py
  ├── convnext_model.py
  ├── vit_model.py
  └── ensemble.py
train.py
evaluate.py
utils/
results/
```

### Training

```bash
python train.py \
  --data_dir data/ \
  --num_classes 10 \
  --batch_size 32 \
  --epochs 50 \
  --device cuda
```

Key training flags can be adjusted for learning rate, augmentation strength, and ensemble weights.

### Evaluation

```bash
python evaluate.py \
  --data_dir data/ \
  --checkpoint_dir checkpoints/ \
  --num_classes 10
```

This generates:
- Confusion matrices
- Classification reports
- ROC/PR curves (if implemented)

***

## Pretrained Models

Pretrained weights for each backbone and the final ensemble can be placed in:

```text
checkpoints/
  ├── efficientnet_cbam_best.pth
  ├── convnext_best.pth
  ├── vit_best.pth
  └── ensemble_best.pth
```

*(Add download links or instructions if you plan to release them.)*

***

## Project Context

This work was developed as part of undergraduate research in electrical and electronic engineering, with a focus on data-efficient deep learning and medical image analysis. It strengthens the foundation for future work in healthcare AI, computer vision, and translational biomedical engineering.

***

## License

Specify your chosen license here, e.g.:

```text
MIT License
```

or

```text
This project is for research and educational purposes only.
```

***

## Citation

If you use this code or model in your research, please cite:

```bibtex
@misc{alsamin2026eye,
  author = {Al Samin, Yasar},
  title = {Eye Disease Classification: Multi-Model Ensemble for Retinal Image Analysis},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/your-username/eye-disease-classification}
}
```

***

## Contact

For questions, collaboration, or feedback, please open an issue or contact:

- **Name:** Yasar Al Samin  
- **Email:** [your-email@example.com]  
- **LinkedIn / Personal Website:** [optional links]

***

