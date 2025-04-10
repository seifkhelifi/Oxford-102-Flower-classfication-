# Oxford 102 Flower Classification

This project implements a fine-tuned Vision Transformer (ViT) model to classify flower species from the Oxford 102 Flower dataset. Through systematic experimentation with various architectures and hyperparameters, the model achieves 97.06% validation accuracy.

## Project Overview

The Oxford 102 Flower dataset consists of 102 flower categories with significant intra-class variations and fine-grained visual differences between classes. This project explores how different modern architectures perform on this dataset and implements strategies to address common challenges such as overfitting.

## Methodology

### Data Augmentation Strategy

The augmentation pipeline was specifically designed for flower images, taking into account their natural variations:

```python
transforms.RandomResizedCrop(224, scale=(0.7, 1.0))
transforms.RandomHorizontalFlip(p=0.5)
transforms.RandomVerticalFlip(p=0.1)  # Some flowers have natural rotations
transforms.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.2, hue=0.1)
transforms.RandomApply([transforms.GaussianBlur(kernel_size=(3,3))], p=0.2)
transforms.RandomRotation(30)
transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))  # Small shifts
transforms.ToTensor()
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transforms.RandomErasing(p=0.1, scale=(0.02, 0.1))  # Random masking
```

### Experimental Approach

1. **Model Benchmarking**: Initial evaluation of ResNet50, ViT-B/16, and EfficientNet-B0 to determine the most promising architecture.
2. **Addressing Overfitting**: Implemented regularization techniques at various strengths to find optimal configuration.
3. **Architecture Selection**: After observing memorization issues with larger models, evaluated smaller variants for better generalization.

## Experiments and Results

### Initial Benchmarking

Training configuration: batch_size=32, lr=1e-4

#### ResNet50 Performance

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
| ----- | ---------- | --------- | -------- | ------- |
| 10/10 | 0.1495     | 98.14%    | 0.4262   | 89.41%  |

#### ViT-B/16 Performance

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
| ----- | ---------- | --------- | -------- | ------- |
| 10/10 | 0.0439     | 99.90%    | 0.3852   | 92.25%  |

**Observation**: ViT-B/16 outperformed ResNet50 but showed signs of overfitting with nearly 100% training accuracy.

### Regularization Experiments

Various regularization strengths were tested to address overfitting:

#### Dropout = 0.2 before classification layer, Weight Decay = 3e-4

Best result: 93.04% validation accuracy (Epoch 9/10)

#### Dropout = 0.3 before classification layer, Weight Decay = 3e-4

Best result: 92.84% validation accuracy (Epoch 10/10)

#### Dropout = 0.5 before classification layer, Weight Decay = 3e-4

Best result: 93.82% validation accuracy (Epoch 16/20)

#### Dropout = 0.7 before classification layer, Weight Decay = 3e-4

Best result: 91.47% validation accuracy (Epoch 17/20)

#### Dropout = 0.5 before classification layer , 0.1 in MLP , Weight Decay = 3e-4

Best result: 91.47% validation accuracy (Epoch 17/20)

**Observation**: While dropout=0.5 achieved the highest accuracy, all configurations still exhibited memorization issues with larger ViT-B/16 models.

### Model Size Reduction

Switched to ViT-Small-Patch16-224 to reduce model complexity while maintaining the transformer architecture benefits.

Initial results showed reduced memorization but also reduced accuracy (35.69% validation accuracy after 60 epochs).

### Final Configuration

The optimal configuration combined architectural and regularization adjustments:

```python
model = ViT_SMALL_PATCH16_224(
    num_classes=102,
    drop_rate=0.2,
    attn_drop_rate=0.1,
    drop_path_rate=0.1  # stochastic depth
)
```

Training parameters:

-   Learning rate: 1e-4
-   Batch size: 64
-   Training epochs: 20

### Final Results

Best performance achieved with early stopping at epoch 17:

-   Training accuracy: 98.14%
-   Training loss: 0.1826
-   Validation accuracy: 97.06%
-   Validation loss: 0.1598

<img src="vit_small_plot.png" alt="" width="600"/>

## Conclusion

This project demonstrates that Vision Transformers can achieve excellent performance on fine-grained flower classification when properly regularized. The ViT-Small model with appropriate dropout and attention dropout rates outperforms larger architectures by reducing overfitting while maintaining high classification accuracy.

The systematic experimentation process highlights the importance of selecting an appropriate model complexity for the dataset size and implementing effective regularization strategies.

## Future Work

-   Implement learning rate scheduling for potentially better convergence
-   Explore test-time augmentation to further improve accuracy
-   Evaluate ensemble methods combining different architectures
-   Investigate knowledge distillation from larger to smaller models
