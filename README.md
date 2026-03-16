# Medical Image Classification

Deep learning and machine learning approaches for medical image binary classification using transfer learning with modern CNN architectures.

## Datasets

### CNMC Leukemia (C-NMC)
Binary classification of blood cell microscopy images to detect **Acute Lymphoblastic Leukemia (ALL)**.

- **Source:** [C-NMC Leukemia Dataset](https://www.kaggle.com/datasets/avk256/cnmc-leukemia)
- **Classes:** ALL (cancer) vs Normal
- **Splits:** fold_0 (train) / fold_1 (val) / fold_2 (test)

### OCT Binary
Binary classification of **Optical Coherence Tomography (OCT)** retinal images.

- **Source:** [OCT Binary Dataset](https://www.kaggle.com/datasets/mohamedaminedrif/oct-binary)
- **Classes:** Disease vs Normal
- **Splits:** train / val / test

## Models

| Model | Architecture | Parameters | Input Size |
|-------|-------------|------------|------------|
| **VGG16** | Classic deep CNN | 138M | 224×224 |
| **EfficientNetB0** | Compound scaling | 5.3M | 224×224 |
| **MobileNetV2** | Inverted residuals | 3.4M | 224×224 |

## Methods (4 per model)

| Method | Description | Technique |
|--------|-------------|-----------|
| **Method 1 — Base Model** | Frozen pretrained base, train classifier head only | Transfer Learning |
| **Method 2 — Fine-Tuning** | Phase 1: frozen base → Phase 2: unfreeze top layers with lower LR | Fine-Tuning |
| **Method 3 — Augmentation** | Frozen base + data augmentation (rotation, flips, zoom, brightness) | Data Augmentation |
| **Method 4 — ML Classifier** | Extract CNN features → train SVM + Random Forest | Feature Extraction + ML |

## Repository Structure

```
├── CNMC/                          # Leukemia classification
│   ├── vgg16/
│   │   ├── VGG16-Method1-BaseModel.ipynb
│   │   ├── VGG16-Method2-FineTuning.ipynb
│   │   ├── VGG16-Method3-Augmentation.ipynb
│   │   └── VGG16-Method4-MLClassifier.ipynb
│   ├── EfficientNet/
│   │   ├── EfficientNetB0-Method1-BaseModel.ipynb
│   │   ├── EfficientNetB0-Method2-FineTuning.ipynb
│   │   ├── EfficientNetB0-Method3-Augmentation.ipynb
│   │   └── EfficientNetB0-Method4-MLClassifier.ipynb
│   └── MobileNet/
│       ├── MobileNetV2-Method1-BaseModel.ipynb
│       ├── MobileNetV2-Method2-FineTuning.ipynb
│       ├── MobileNetV2-Method3-Augmentation.ipynb
│       └── MobileNetV2-Method4-MLClassifier.ipynb
│
└── OCT-Binary/                    # OCT retinal classification
    ├── vgg16/
    │   └── (same 4 methods)
    ├── EfficientNet/
    │   └── (same 4 methods)
    └── MobileNet/
        └── (same 4 methods)
```

## Features

- **Pickle Checkpointing** — Training state saved after every epoch for Kaggle session recovery
- **No Callbacks** — Simple training loops, easy to understand and modify
- **Complete Evaluation** — Confusion matrix, classification report, prediction visualization
- **Kaggle GPU Ready** — All notebooks configured for Kaggle GPU acceleration

## How to Run

1. Open any notebook on [Kaggle](https://www.kaggle.com)
2. Add the corresponding dataset
3. Enable GPU accelerator
4. Run all cells — training resumes automatically if session restarts

## Tech Stack

- **TensorFlow / Keras** — Model building and training
- **scikit-learn** — SVM, Random Forest, metrics
- **Matplotlib / Seaborn** — Visualization
