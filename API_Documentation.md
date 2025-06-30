# Comprehensive API Documentation
## Solar PV Anomaly Detection using Deep Learning Methods

### Project Overview

This documentation covers the public APIs, functions, and components for the autonomous monitoring and classification of solar PV anomalies using deep learning methods. The system employs Vision Transformers (ViT) and Data-efficient Image Transformers (DeiT) for anomaly detection in thermal images of floating solar panels.

---

## Table of Contents

1. [Dataset API](#dataset-api)
2. [Model Architectures](#model-architectures)
3. [Data Processing Components](#data-processing-components)
4. [Training and Evaluation APIs](#training-and-evaluation-apis)
5. [Utility Functions](#utility-functions)
6. [Configuration Parameters](#configuration-parameters)
7. [Usage Examples](#usage-examples)

---

## Dataset API

### IRDataset Class

The IRDataset class provides access to the InfraredSolarModules dataset for solar panel anomaly detection.

#### Class Definition

```python
class IRDataset:
    """
    Dataset class for loading and processing IR images of solar panels.
    
    The dataset contains 20,000 images with 12 classes:
    - 11 different types of anomalies
    - 1 no-anomaly class (nominal solar modules)
    """
```

#### Constructor

```python
def __init__(self, data_path: str, transform=None, target_transform=None):
    """
    Initialize the IR dataset.
    
    Args:
        data_path (str): Path to the dataset directory
        transform (callable, optional): Transform to apply to images
        target_transform (callable, optional): Transform to apply to labels
    
    Returns:
        IRDataset: Initialized dataset object
    """
```

#### Methods

```python
def __len__(self) -> int:
    """Return the total number of samples in the dataset."""

def __getitem__(self, idx: int) -> tuple:
    """
    Get a sample from the dataset.
    
    Args:
        idx (int): Index of the sample to retrieve
        
    Returns:
        tuple: (image, label) pair
    """

def get_class_distribution(self) -> dict:
    """
    Get the distribution of classes in the dataset.
    
    Returns:
        dict: Dictionary with class names as keys and counts as values
    """
```

#### Dataset Classes

| Class | Count | Description |
|-------|--------|-------------|
| Cell | 1,877 | Hot spot occurring with square geometry in a single cell |
| Cell-Multi | 1,288 | Hot spots occurring with square geometry in multiple cells |
| Cracking | 941 | Module anomaly caused by cracking on the module surface |
| Hot-Spot | 251 | Hot spot on a thin film module |
| Hot-Spot-Multi | 247 | Multiple hot spots on a thin film module |
| Shadowing | 1,056 | Sunlight obstructed by vegetation, man-made structures, or adjacent rows |
| Diode | 1,499 | Activated bypass diode, typically 1/3 of the module |
| Diode-Multi | 175 | Multiple activated bypass diodes, typically affecting 2/3 of the module |
| Vegetation | 1,639 | Panels blocked by vegetation |
| Soiling | 205 | Dirt, dust, or other debris on the surface of the module |
| Offline-Module | 828 | The entire module is heated |
| No-Anomaly | 10,000 | Nominal solar module |

#### Usage Example

```python
from torch.utils.data import DataLoader
from torchvision import transforms

# Define transforms
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

# Initialize dataset
dataset = IRDataset(
    data_path="/path/to/dataset",
    transform=transform
)

# Create data loader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Get class distribution
distribution = dataset.get_class_distribution()
print(f"Dataset contains {len(dataset)} samples")
```

---

## Model Architectures

### Vision Transformer (ViT)

#### VisionTransformer Class

```python
class VisionTransformer:
    """
    Vision Transformer implementation for image classification.
    
    Treats an image as a sequence of patches and applies transformer
    architecture for classification tasks.
    """
```

#### Constructor

```python
def __init__(self, 
             image_size: int = 160,
             patch_size: int = 16,
             num_classes: int = 12,
             dim: int = 768,
             depth: int = 12,
             heads: int = 12,
             mlp_dim: int = 3072,
             dropout: float = 0.1):
    """
    Initialize Vision Transformer.
    
    Args:
        image_size (int): Input image size (default: 160)
        patch_size (int): Size of image patches (default: 16)
        num_classes (int): Number of output classes (default: 12)
        dim (int): Hidden dimension (default: 768)
        depth (int): Number of transformer layers (default: 12)
        heads (int): Number of attention heads (default: 12)
        mlp_dim (int): MLP dimension (default: 3072)
        dropout (float): Dropout rate (default: 0.1)
    """
```

#### Core Methods

```python
def patch_embedding(self, x):
    """
    Convert input image to patch embeddings.
    
    Args:
        x (torch.Tensor): Input image tensor [B, C, H, W]
        
    Returns:
        torch.Tensor: Patch embeddings [B, N, D]
    """

def positional_encoding(self, pos: int, d_model: int):
    """
    Generate positional encodings.
    
    Mathematical formulation:
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    Args:
        pos (int): Position in sequence
        d_model (int): Model dimension
        
    Returns:
        torch.Tensor: Positional encoding vector
    """

def multi_head_attention(self, query, key, value, mask=None):
    """
    Multi-head self-attention mechanism.
    
    Mathematical formulation:
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
    
    Args:
        query (torch.Tensor): Query tensor
        key (torch.Tensor): Key tensor  
        value (torch.Tensor): Value tensor
        mask (torch.Tensor, optional): Attention mask
        
    Returns:
        torch.Tensor: Attention output
    """

def forward(self, x):
    """
    Forward pass through the Vision Transformer.
    
    Args:
        x (torch.Tensor): Input image tensor [B, C, H, W]
        
    Returns:
        torch.Tensor: Classification logits [B, num_classes]
    """
```

#### Hyperparameters

| Parameter | Value | Description |
|-----------|--------|-------------|
| Learning Rate | 1e-4 | Initial learning rate |
| Batch Size | 32 | Training batch size |
| Epochs | 100 | Maximum training epochs |
| Optimizer | AdamW | Optimization algorithm |
| Weight Decay | 0.01 | L2 regularization |
| Image Size | 160×160 | Input image dimensions |
| Patch Size | 16×16 | Size of image patches |
| Number of Heads | 12 | Multi-head attention heads |
| Number of Layers | 12 | Transformer encoder layers |
| Hidden Dimension | 768 | Hidden layer dimension |
| MLP Dimension | 3072 | Feed-forward network dimension |
| Dropout Rate | 0.1 | Dropout probability |

### Data-efficient Image Transformer (DeiT)

#### DeiTModel Class

```python
class DeiTModel:
    """
    Data-efficient Image Transformer with knowledge distillation.
    
    Extends ViT with distillation token and teacher-student training
    for improved data efficiency.
    """
```

#### Constructor

```python
def __init__(self,
             image_size: int = 160,
             patch_size: int = 16,
             num_classes: int = 12,
             dim: int = 768,
             depth: int = 12,
             heads: int = 12,
             mlp_dim: int = 3072,
             dropout: float = 0.1,
             distillation_alpha: float = 0.5,
             distillation_beta: float = 0.5,
             temperature: float = 3.0):
    """
    Initialize DeiT model with distillation parameters.
    
    Args:
        distillation_alpha (float): Balance between CE and distillation loss
        distillation_beta (float): Balance between hard and soft distillation
        temperature (float): Temperature for soft distillation
    """
```

#### Distillation Methods

```python
def hard_distillation_loss(self, student_logits, teacher_logits):
    """
    Compute hard distillation loss.
    
    Mathematical formulation:
    L_hard = CE(y_s, y_t)
    
    Args:
        student_logits (torch.Tensor): Student model predictions
        teacher_logits (torch.Tensor): Teacher model predictions
        
    Returns:
        torch.Tensor: Hard distillation loss
    """

def soft_distillation_loss(self, student_logits, teacher_logits, temperature):
    """
    Compute soft distillation loss using KL divergence.
    
    Mathematical formulation:
    L_soft = τ² * KL(z_s/τ || z_t/τ)
    
    Args:
        student_logits (torch.Tensor): Student model logits
        teacher_logits (torch.Tensor): Teacher model logits
        temperature (float): Distillation temperature
        
    Returns:
        torch.Tensor: Soft distillation loss
    """

def combined_loss(self, student_logits, teacher_logits, true_labels, alpha, beta):
    """
    Compute combined training loss.
    
    Mathematical formulation:
    L_total = α * L_CE + (1-α) * (β * L_hard + (1-β) * L_soft)
    
    Args:
        student_logits (torch.Tensor): Student predictions
        teacher_logits (torch.Tensor): Teacher predictions  
        true_labels (torch.Tensor): Ground truth labels
        alpha (float): CE loss weight
        beta (float): Hard distillation weight
        
    Returns:
        torch.Tensor: Combined loss value
    """
```

---

## Data Processing Components

### DataPreprocessor Class

```python
class DataPreprocessor:
    """
    Handles data preprocessing pipeline for solar panel images.
    """
    
    def __init__(self, image_size: tuple = (160, 160)):
        """
        Initialize data preprocessor.
        
        Args:
            image_size (tuple): Target image size for resizing
        """
```

#### Methods

```python
def normalize_image(self, image):
    """
    Normalize image pixels to [0, 1] range.
    
    Args:
        image (numpy.ndarray): Input image array
        
    Returns:
        numpy.ndarray: Normalized image
    """

def resize_image(self, image, target_size):
    """
    Resize image to target dimensions.
    
    Args:
        image (numpy.ndarray): Input image
        target_size (tuple): Target (height, width)
        
    Returns:
        numpy.ndarray: Resized image
    """

def apply_transforms(self, image):
    """
    Apply preprocessing transforms to image.
    
    Args:
        image (numpy.ndarray): Input image
        
    Returns:
        torch.Tensor: Preprocessed image tensor
    """
```

### DataAugmentation Class

#### Offline Augmentation

```python
class OfflineAugmentation:
    """
    Precomputed data augmentation techniques.
    """
    
    def __init__(self, output_dir: str):
        """
        Initialize offline augmentation.
        
        Args:
            output_dir (str): Directory to save augmented images
        """
    
    def rotate_image(self, image, angle):
        """Rotate image by specified angle."""
    
    def flip_image(self, image, horizontal=True, vertical=False):
        """Flip image horizontally and/or vertically."""
    
    def add_noise(self, image, noise_factor=0.1):
        """Add Gaussian noise to image."""
    
    def generate_augmented_dataset(self, dataset, augmentations_per_image=5):
        """Generate augmented versions of the entire dataset."""
```

#### Online Augmentation

```python
class OnlineAugmentation:
    """
    Real-time data augmentation during training.
    """
    
    def __init__(self, 
                 rotation_range: float = 15.0,
                 brightness_range: float = 0.2,
                 contrast_range: float = 0.2):
        """
        Initialize online augmentation parameters.
        
        Args:
            rotation_range (float): Maximum rotation angle in degrees
            brightness_range (float): Brightness adjustment range
            contrast_range (float): Contrast adjustment range
        """
    
    def random_transform(self, image):
        """Apply random augmentation to image during training."""
```

### DataSplitter Class

```python
class DataSplitter:
    """
    Handles dataset splitting for training, validation, and testing.
    """
    
    def __init__(self, 
                 train_ratio: float = 0.7,
                 val_ratio: float = 0.15,
                 test_ratio: float = 0.15,
                 random_seed: int = 42):
        """
        Initialize data splitter with specified ratios.
        
        Args:
            train_ratio (float): Training set proportion (default: 0.7)
            val_ratio (float): Validation set proportion (default: 0.15)
            test_ratio (float): Test set proportion (default: 0.15)
            random_seed (int): Random seed for reproducibility
        """
    
    def split_dataset(self, dataset):
        """
        Split dataset into train, validation, and test sets.
        
        Args:
            dataset: Input dataset to split
            
        Returns:
            tuple: (train_dataset, val_dataset, test_dataset)
        """
    
    def stratified_split(self, dataset):
        """
        Perform stratified splitting to maintain class distribution.
        
        Args:
            dataset: Input dataset to split
            
        Returns:
            tuple: (train_dataset, val_dataset, test_dataset)
        """
```

---

## Training and Evaluation APIs

### Trainer Class

```python
class ModelTrainer:
    """
    Handles model training with support for transfer learning and fine-tuning.
    """
    
    def __init__(self, 
                 model,
                 device: str = 'cuda',
                 learning_rate: float = 1e-4,
                 weight_decay: float = 0.01,
                 scheduler_type: str = 'cosine'):
        """
        Initialize model trainer.
        
        Args:
            model: Model to train (ViT or DeiT)
            device (str): Training device ('cuda' or 'cpu')
            learning_rate (float): Initial learning rate
            weight_decay (float): Weight decay for regularization
            scheduler_type (str): Learning rate scheduler type
        """
```

#### Training Methods

```python
def train_epoch(self, dataloader, epoch):
    """
    Train model for one epoch.
    
    Args:
        dataloader: Training data loader
        epoch (int): Current epoch number
        
    Returns:
        dict: Training metrics (loss, accuracy)
    """

def validate_epoch(self, dataloader):
    """
    Validate model on validation set.
    
    Args:
        dataloader: Validation data loader
        
    Returns:
        dict: Validation metrics (loss, accuracy)
    """

def train_model(self, 
                train_loader, 
                val_loader, 
                num_epochs: int = 100,
                early_stopping_patience: int = 10):
    """
    Complete training loop with early stopping.
    
    Args:
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs (int): Maximum number of epochs
        early_stopping_patience (int): Patience for early stopping
        
    Returns:
        dict: Training history and final metrics
    """

def fine_tune_model(self, 
                   pretrained_weights_path: str,
                   train_loader,
                   val_loader,
                   freeze_layers: int = 8):
    """
    Fine-tune pretrained model on dataset.
    
    Args:
        pretrained_weights_path (str): Path to pretrained weights
        train_loader: Training data loader
        val_loader: Validation data loader
        freeze_layers (int): Number of layers to freeze
        
    Returns:
        dict: Fine-tuning results
    """
```

### Evaluator Class

```python
class ModelEvaluator:
    """
    Comprehensive model evaluation and metrics computation.
    """
    
    def __init__(self, model, device: str = 'cuda'):
        """
        Initialize model evaluator.
        
        Args:
            model: Trained model to evaluate
            device (str): Evaluation device
        """
    
    def evaluate_model(self, test_loader):
        """
        Evaluate model on test set.
        
        Args:
            test_loader: Test data loader
            
        Returns:
            dict: Comprehensive evaluation metrics
        """
    
    def compute_confusion_matrix(self, y_true, y_pred, class_names):
        """
        Compute and visualize confusion matrix.
        
        Args:
            y_true (array): True labels
            y_pred (array): Predicted labels
            class_names (list): Class name labels
            
        Returns:
            numpy.ndarray: Confusion matrix
        """
    
    def classification_report(self, y_true, y_pred, class_names):
        """
        Generate detailed classification report.
        
        Args:
            y_true (array): True labels
            y_pred (array): Predicted labels  
            class_names (list): Class name labels
            
        Returns:
            dict: Classification metrics per class
        """
    
    def compute_metrics(self, y_true, y_pred):
        """
        Compute standard classification metrics.
        
        Returns:
            dict: Dictionary containing:
                - accuracy: Overall accuracy
                - precision: Macro-averaged precision
                - recall: Macro-averaged recall
                - f1_score: Macro-averaged F1-score
        """
```

---

## Utility Functions

### Image Processing Utilities

```python
def load_image(image_path: str) -> numpy.ndarray:
    """
    Load image from file path.
    
    Args:
        image_path (str): Path to image file
        
    Returns:
        numpy.ndarray: Loaded image array
    """

def save_image(image: numpy.ndarray, output_path: str):
    """
    Save image array to file.
    
    Args:
        image (numpy.ndarray): Image array to save
        output_path (str): Output file path
    """

def visualize_predictions(images, predictions, true_labels, class_names):
    """
    Visualize model predictions alongside true labels.
    
    Args:
        images (torch.Tensor): Batch of images
        predictions (torch.Tensor): Model predictions
        true_labels (torch.Tensor): True labels
        class_names (list): List of class names
    """

def plot_training_history(history: dict):
    """
    Plot training and validation loss/accuracy curves.
    
    Args:
        history (dict): Training history dictionary
    """
```

### Model Utilities

```python
def save_model(model, optimizer, epoch, loss, filepath: str):
    """
    Save model checkpoint.
    
    Args:
        model: PyTorch model
        optimizer: Optimizer state
        epoch (int): Current epoch
        loss (float): Current loss
        filepath (str): Save path
    """

def load_model(filepath: str, model, optimizer=None):
    """
    Load model checkpoint.
    
    Args:
        filepath (str): Path to checkpoint
        model: Model to load weights into
        optimizer: Optional optimizer to load state
        
    Returns:
        dict: Checkpoint information
    """

def count_parameters(model):
    """
    Count trainable parameters in model.
    
    Args:
        model: PyTorch model
        
    Returns:
        int: Number of trainable parameters
    """
```

---

## Configuration Parameters

### System Requirements

```python
SYSTEM_SPECS = {
    "processor": "12th Gen Intel Core i7-12700H",
    "base_clock": "2.3 GHz",
    "cores": 14,
    "threads": 20,
    "gpu": "NVIDIA GeForce RTX 3060 Laptop GPU",
    "vram": "4GB",
    "integrated_graphics": "Intel Iris Xe Graphics (2GB VRAM)"
}
```

### Model Configuration

```python
VIT_CONFIG = {
    "learning_rate": 1e-4,
    "batch_size": 32,
    "epochs": 100,
    "optimizer": "AdamW",
    "weight_decay": 0.01,
    "image_size": (160, 160),
    "patch_size": (16, 16),
    "num_heads": 12,
    "num_layers": 12,
    "hidden_dim": 768,
    "mlp_dim": 3072,
    "dropout_rate": 0.1,
    "activation": "GELU",
    "scheduler": "CosineAnnealingLR",
    "early_stopping_patience": 10
}

DEIT_CONFIG = {
    **VIT_CONFIG,  # Inherit ViT configuration
    "distillation_alpha": 0.5,
    "distillation_beta": 0.5,
    "temperature": 3.0,
    "teacher_model": "CNN_ResNet50"  # Teacher model for distillation
}
```

### Data Configuration

```python
DATASET_CONFIG = {
    "dataset_url": "https://github.com/RaptorMaps/InfraredSolarModules",
    "total_images": 20000,
    "num_classes": 12,
    "image_format": ["JPEG", "PNG"],
    "train_split": 0.7,
    "val_split": 0.15,
    "test_split": 0.15,
    "normalization": {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225]
    }
}
```

---

## Usage Examples

### Complete Training Pipeline

```python
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

# 1. Initialize dataset and preprocessing
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

dataset = IRDataset("path/to/dataset", transform=transform)

# 2. Split dataset
splitter = DataSplitter(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
train_dataset, val_dataset, test_dataset = splitter.split_dataset(dataset)

# 3. Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 4. Initialize model
model = VisionTransformer(
    image_size=160,
    patch_size=16,
    num_classes=12,
    dim=768,
    depth=12,
    heads=12,
    mlp_dim=3072,
    dropout=0.1
)

# 5. Train model
trainer = ModelTrainer(model, device='cuda', learning_rate=1e-4)
training_history = trainer.train_model(
    train_loader, 
    val_loader, 
    num_epochs=100,
    early_stopping_patience=10
)

# 6. Evaluate model
evaluator = ModelEvaluator(model, device='cuda')
test_results = evaluator.evaluate_model(test_loader)

print(f"Test Accuracy: {test_results['accuracy']:.4f}")
print(f"Test F1-Score: {test_results['f1_score']:.4f}")
```

### DeiT with Knowledge Distillation

```python
# 1. Initialize teacher and student models
teacher_model = load_pretrained_cnn("resnet50_pretrained.pth")
student_model = DeiTModel(
    image_size=160,
    patch_size=16,
    num_classes=12,
    distillation_alpha=0.5,
    distillation_beta=0.5,
    temperature=3.0
)

# 2. Train with distillation
distillation_trainer = DistillationTrainer(
    student_model=student_model,
    teacher_model=teacher_model,
    device='cuda'
)

distillation_results = distillation_trainer.train_with_distillation(
    train_loader,
    val_loader,
    num_epochs=100
)

# 3. Compare with standard training
print(f"Standard ViT Accuracy: {test_results['accuracy']:.4f}")
print(f"DeiT with Distillation Accuracy: {distillation_results['accuracy']:.4f}")
```

### Transfer Learning Example

```python
# 1. Load pretrained model
pretrained_model = VisionTransformer.from_pretrained("vit_base_patch16_224")

# 2. Fine-tune on solar panel dataset
trainer = ModelTrainer(pretrained_model, learning_rate=1e-5)  # Lower LR for fine-tuning

fine_tuning_results = trainer.fine_tune_model(
    pretrained_weights_path="vit_imagenet_weights.pth",
    train_loader=train_loader,
    val_loader=val_loader,
    freeze_layers=8  # Freeze first 8 layers
)

# 3. Evaluate fine-tuned model
evaluator = ModelEvaluator(pretrained_model)
fine_tuned_results = evaluator.evaluate_model(test_loader)
```

### Anomaly Detection Pipeline

```python
# 1. Binary classification (Anomaly vs No-Anomaly)
binary_model = VisionTransformer(num_classes=2)  # 2 classes
binary_trainer = ModelTrainer(binary_model)

# Train for binary classification
binary_results = binary_trainer.train_model(train_loader, val_loader)

# 2. Multi-class anomaly classification (12 classes)
multiclass_model = VisionTransformer(num_classes=12)  # 12 classes
multiclass_trainer = ModelTrainer(multiclass_model)

# Train for 12-class classification
multiclass_results = multiclass_trainer.train_model(train_loader, val_loader)

# 3. Compare performance
print(f"Binary Classification Accuracy: {binary_results['accuracy']:.4f}")
print(f"12-Class Classification Accuracy: {multiclass_results['accuracy']:.4f}")
```

### Real-time Inference Example

```python
def predict_anomaly(image_path: str, model, device: str = 'cuda'):
    """
    Predict anomaly type for a single solar panel image.
    
    Args:
        image_path (str): Path to input image
        model: Trained model
        device (str): Inference device
        
    Returns:
        dict: Prediction results with confidence scores
    """
    # Load and preprocess image
    image = load_image(image_path)
    preprocessed = preprocess_image(image)
    
    # Move to device and add batch dimension
    input_tensor = preprocessed.unsqueeze(0).to(device)
    
    # Inference
    model.eval()
    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    
    # Map to class name
    class_names = ["Cell", "Cell-Multi", "Cracking", "Hot-Spot", 
                   "Hot-Spot-Multi", "Shadowing", "Diode", "Diode-Multi",
                   "Vegetation", "Soiling", "Offline-Module", "No-Anomaly"]
    
    return {
        "predicted_class": class_names[predicted_class],
        "confidence": confidence,
        "all_probabilities": probabilities[0].cpu().numpy()
    }

# Usage
result = predict_anomaly("solar_panel_image.jpg", trained_model)
print(f"Predicted: {result['predicted_class']} (confidence: {result['confidence']:.4f})")
```

---

## Error Handling and Logging

### Exception Classes

```python
class DatasetError(Exception):
    """Raised when dataset loading or processing fails."""

class ModelTrainingError(Exception):
    """Raised when model training encounters an error."""

class InferenceError(Exception):
    """Raised when model inference fails."""

class ConfigurationError(Exception):
    """Raised when configuration parameters are invalid."""
```

### Logging Configuration

```python
import logging

def setup_logging(log_level: str = "INFO", log_file: str = "training.log"):
    """
    Configure logging for the application.
    
    Args:
        log_level (str): Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file (str): Path to log file
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
```

---

## Performance Metrics and Benchmarks

### Expected Performance

| Model | Dataset Split | Accuracy | Precision | Recall | F1-Score |
|-------|---------------|----------|-----------|---------|----------|
| ViT-Base | Binary (2-class) | 0.95+ | 0.94+ | 0.95+ | 0.94+ |
| ViT-Base | Multi-class (12-class) | 0.87+ | 0.86+ | 0.87+ | 0.86+ |
| DeiT-Base | Binary (2-class) | 0.96+ | 0.95+ | 0.96+ | 0.95+ |
| DeiT-Base | Multi-class (12-class) | 0.89+ | 0.88+ | 0.89+ | 0.88+ |

### Training Time Estimates

| Model | Hardware | Training Time (100 epochs) |
|-------|----------|----------------------------|
| ViT-Base | RTX 3060 (4GB) | ~4-6 hours |
| DeiT-Base | RTX 3060 (4GB) | ~5-7 hours |
| ViT-Large | RTX 3060 (4GB) | ~8-12 hours |

---

## Contributing Guidelines

### Code Style

- Follow PEP 8 for Python code
- Use type hints for function parameters and return values
- Document all public methods and classes
- Include docstrings with parameter descriptions and examples

### Testing

```python
# Example test structure
import unittest

class TestVisionTransformer(unittest.TestCase):
    def setUp(self):
        self.model = VisionTransformer(num_classes=12)
        self.sample_input = torch.randn(1, 3, 160, 160)
    
    def test_forward_pass(self):
        output = self.model(self.sample_input)
        self.assertEqual(output.shape, (1, 12))
    
    def test_patch_embedding(self):
        patches = self.model.patch_embedding(self.sample_input)
        expected_patches = (160 // 16) ** 2  # 100 patches
        self.assertEqual(patches.shape[1], expected_patches)
```

---

## License and Citation

### License
This project is licensed under the MIT License. See LICENSE file for details.

### Citation
If you use this code or methodology in your research, please cite:

```bibtex
@mastersthesis{mubshir2025solar,
    title={Autonomous Monitoring and Classification of Solar PVs Anomalies using Deep Learning Methods},
    author={Zeshan Mubshir},
    year={2025},
    school={Norwegian University of Science and Technology},
    department={Department of ICT and Natural Sciences},
    type={Master's thesis},
    supervisor={Saleh Abdel-Afou Alaliyat and Mohammadreza Aghaei}
}
```

---

## Contact and Support

For questions, issues, or contributions, please contact:

- **Author**: Zeshan Mubshir
- **Institution**: Norwegian University of Science and Technology (NTNU)
- **Department**: ICT and Natural Sciences
- **Supervisor**: Saleh Abdel-Afou Alaliyat
- **Co-supervisor**: Mohammadreza Aghaei

---

*This documentation covers the comprehensive API for solar PV anomaly detection using deep learning methods. For implementation details and additional resources, refer to the complete thesis document and accompanying research materials.*