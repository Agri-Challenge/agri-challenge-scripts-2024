#!/usr/bin/env python3
"""
AgrI Challenge -- Cross-Team Validation (CTV) Framework

Command-line script implementing the TOTO (Train-on-One-Team-Only) and LOTO
(Leave-One-Team-Out) evaluation protocols described in:

  Brahimi et al. (2026). AgrI Challenge: Cross-Team Insights from a
  Data-Centric AI Competition in Agricultural Vision.

Supported architectures: DenseNet121, Swin Transformer (via timm).

Usage:
    python agri_challenge_ctv_cli_v3.py --scenario TOTO --architecture swin --epochs 20
    python agri_challenge_ctv_cli_v3.py --scenario LOTO --architecture densenet --epochs 30
"""

import argparse
import sys

# Import everything from the optimized script
import os
import json
import time
import random
import copy
import warnings
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import train_test_split

# Optional imports
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False

warnings.filterwarnings('ignore')


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Agri-Challenge Cross-Team Validation Framework (OPTIMIZED)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # TOTO with Swin, 20 epochs
  python agri_challenge_ctv_cli_v3.py --scenario TOTO --architecture swin --epochs 20

  # LOTO with DenseNet, 20 epochs
  python agri_challenge_ctv_cli_v3.py --scenario LOTO --architecture densenet --epochs 20

  # Quick test with 5 epochs
  python agri_challenge_ctv_cli_v3.py --scenario TOTO --architecture swin --epochs 5 --batch-size 16
        """
    )
    
    # Required arguments
    parser.add_argument('--scenario', type=str, required=True, choices=['TOTO', 'LOTO'],
                        help='Training scenario: TOTO or LOTO')
    parser.add_argument('--architecture', type=str, required=True, choices=['densenet', 'swin'],
                        help='Model architecture: densenet or swin')
    
    # Optional arguments with defaults
    parser.add_argument('--data-dir', type=str, 
                        default='./data/agri_challenge2024_dataset_336x336_2025-08-24/',
                        help='Path to dataset directory')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs (default: 20)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for training (default: 32)')
    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate (default: 0.0001)')
    parser.add_argument('--val-split', type=float, default=0.3,
                        help='Validation split fraction (default: 0.3)')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate (default: 0.3)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers (default: 4)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--no-wandb', action='store_true',
                        help='Disable Weights & Biases tracking')
    parser.add_argument('--no-augmentation', action='store_true',
                        help='Disable data augmentation')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: auto-generated)')
    parser.add_argument('--no-compile', action='store_true',
                        help='Disable torch.compile (use if PyTorch < 2.0)')
    parser.add_argument('--no-mixed-precision', action='store_true',
                        help='Disable mixed precision training')
    
    args = parser.parse_args()
    
    # Validate data directory
    if not Path(args.data_dir).exists():
        parser.error(f"Data directory not found: {args.data_dir}")
    
    return args


class Config:
    """Configuration class populated from command-line arguments."""
    
    @classmethod
    def from_args(cls, args):
        """Create Config from parsed arguments."""
        config = cls()
        
        # From arguments
        config.SCENARIO = args.scenario
        config.ARCHITECTURE = args.architecture
        config.DATA_DIR = args.data_dir
        config.EPOCHS = args.epochs
        config.BATCH_SIZE = args.batch_size
        config.LEARNING_RATE = args.lr
        config.VALIDATION_SPLIT = args.val_split
        config.DROPOUT_RATE = args.dropout
        config.NUM_WORKERS = args.num_workers
        config.RANDOM_SEED = args.seed
        config.USE_DATA_AUGMENTATION = not args.no_augmentation
        config.USE_WANDB = (not args.no_wandb) and WANDB_AVAILABLE
        
        config.USE_COMPILE = not args.no_compile and hasattr(torch, 'compile')
        config.USE_MIXED_PRECISION = (not args.no_mixed_precision and
                                     torch.cuda.is_available() and
                                     torch.cuda.get_device_capability()[0] >= 7)

        # Fixed parameters
        config.INPUT_SIZE = 224
        config.WEIGHT_DECAY = 1e-4
        config.EARLY_STOPPING_PATIENCE = 30
        config.SCHEDULER_TYPE = "cosine"
        config.PIN_MEMORY = True
        config.PERSISTENT_WORKERS = config.NUM_WORKERS > 0
        config.PREFETCH_FACTOR = 2 if config.NUM_WORKERS > 0 else None

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # W&B settings -- API key is read from the WANDB_API_KEY environment variable
        config.WANDB_PROJECT = f"agri-ctv-{config.SCENARIO.lower()}-{config.ARCHITECTURE}-ep{config.EPOCHS}-{timestamp}"

        if args.output_dir:
            config.OUTPUT_DIR = args.output_dir
        else:
            config.OUTPUT_DIR = f"./experiments_ctv/CTV_{config.SCENARIO}_{config.ARCHITECTURE}_epochs_{config.EPOCHS}_{timestamp}"

        return config
    
    def print_config(self):
        """Print configuration summary."""
        print(f"\n{'='*60}")
        print("AgrI Challenge CTV Framework -- Configuration")
        print(f"{'='*60}")
        print(f"Scenario:           {self.SCENARIO}")
        print(f"Architecture:       {self.ARCHITECTURE}")
        print(f"Epochs:             {self.EPOCHS}")
        print(f"Batch Size:         {self.BATCH_SIZE}")
        print(f"Learning Rate:      {self.LEARNING_RATE}")
        print(f"Validation Split:   {self.VALIDATION_SPLIT}")
        print(f"Data Augmentation:  {self.USE_DATA_AUGMENTATION}")
        print(f"W&B Enabled:        {self.USE_WANDB}")
        print(f"Num Workers:        {self.NUM_WORKERS}")
        print(f"Random Seed:        {self.RANDOM_SEED}")
        print(f"Data Directory:     {self.DATA_DIR}")
        print(f"Output Directory:   {self.OUTPUT_DIR}")
        print(f"Mixed Precision:    {self.USE_MIXED_PRECISION}")
        print(f"Torch Compile:      {self.USE_COMPILE}")
        print(f"{'='*60}\n")


# -----------------------------------------------------------------------
# CORE FUNCTIONS
# -----------------------------------------------------------------------

def set_random_seeds(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_directories(base_dir):
    """Create all necessary output directories."""
    dirs = {
        'base': Path(base_dir),
        'models': Path(base_dir) / 'models',
        'plots': Path(base_dir) / 'plots',
        'results': Path(base_dir) / 'results',
        'matrices': Path(base_dir) / 'matrices',
        'predictions': Path(base_dir) / 'predictions',
        'logs': Path(base_dir) / 'logs'
    }
    
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    
    return dirs


def cleanup_gpu_memory():
    """Clean up GPU memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


class PlantDataset(Dataset):
    """Dataset for plant images."""
    
    def __init__(self, image_paths, labels, class_to_idx, transform=None, return_paths=False):
        self.image_paths = image_paths
        self.labels = labels
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.return_paths = return_paths
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"⚠️ Error loading {image_path}: {e}")
            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        if self.transform:
            image = self.transform(image)
        
        if self.return_paths:
            return image, label, image_path
        return image, label


class DenseNetClassifier(nn.Module):
    """DenseNet121 classifier."""
    
    def __init__(self, num_classes, dropout_rate=0.3):
        super(DenseNetClassifier, self).__init__()
        self.backbone = models.densenet121(pretrained=True)
        num_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)


class SwinTransformerClassifier(nn.Module):
    """Swin Transformer classifier."""
    
    def __init__(self, num_classes, model_name='swin_base_patch4_window7_224', dropout_rate=0.3):
        super(SwinTransformerClassifier, self).__init__()
        if not TIMM_AVAILABLE:
            raise ImportError("timm required for Swin. Install: pip install timm")
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
        num_features = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)


def get_model(architecture, num_classes, dropout_rate, device, use_compile=False):
    """Get model instance with optional compilation."""
    if architecture == "densenet":
        model = DenseNetClassifier(num_classes, dropout_rate)
    elif architecture == "swin":
        model = SwinTransformerClassifier(num_classes, dropout_rate=dropout_rate)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    
    model = model.to(device)
    
    if use_compile:
        print("Compiling model with torch.compile()...")
        try:
            model = torch.compile(model, mode='default')
            print("Model compilation successful.")
        except Exception as e:
            print(f"Warning: model compilation failed ({e}), continuing without compile.")
    
    return model


def get_transforms(input_size=224, use_augmentation=True):
    """Get data transforms."""
    if use_augmentation:
        train_transforms = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=30),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.3, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.15))
        ])
    else:
        train_transforms = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    val_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transforms, val_transforms


def analyze_dataset(data_dir):
    """Analyze dataset structure."""
    print("\n" + "="*60)
    print("📊 ANALYZING DATASET STRUCTURE")
    print("="*60)
    
    SKIP_FOLDERS = {'.ipynb_checkpoints', '__pycache__', '.git', '.DS_Store'}
    VALID_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif', '.webp'}
    
    teams = []
    all_classes = set()
    data_path = Path(data_dir)
    
    if not data_path.exists():
        raise ValueError(f"Dataset directory not found: {data_dir}")
    
    for team_path in sorted(data_path.iterdir()):
        if not team_path.is_dir() or team_path.name in SKIP_FOLDERS:
            continue
        
        team_name = team_path.name
        team_classes = []
        
        for class_path in sorted(team_path.iterdir()):
            if not class_path.is_dir() or class_path.name in SKIP_FOLDERS:
                continue
            
            images = [f for f in class_path.iterdir() 
                     if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS]
            
            if images:
                team_classes.append(class_path.name)
                all_classes.add(class_path.name)
        
        if team_classes:
            teams.append(team_name)
    
    all_classes = sorted(all_classes)
    
    print(f"✅ Found {len(teams)} teams: {teams}")
    print(f"✅ Found {len(all_classes)} classes")
    
    return teams, all_classes


class DataCache:
    """Cache for efficient data loading."""
    
    def __init__(self, data_dir, all_classes):
        self.data_dir = Path(data_dir)
        self.all_classes = all_classes
        self.class_to_idx = {cls: idx for idx, cls in enumerate(all_classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        self._cache = self._load_all_data()
    
    def _load_all_data(self):
        """Load all data paths."""
        print("\n📂 Caching dataset...")
        cache = defaultdict(lambda: {'paths': [], 'labels': []})
        
        VALID_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff'}
        
        for team_path in self.data_dir.iterdir():
            if not team_path.is_dir():
                continue
            
            team_name = team_path.name
            
            for class_path in team_path.iterdir():
                if not class_path.is_dir() or class_path.name not in self.class_to_idx:
                    continue
                
                class_idx = self.class_to_idx[class_path.name]
                image_files = [f for f in class_path.iterdir() 
                              if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS]
                
                for img_file in image_files:
                    cache[team_name]['paths'].append(str(img_file))
                    cache[team_name]['labels'].append(class_idx)
        
        print(f"✅ Cached {sum(len(v['paths']) for v in cache.values())} images")
        return cache
    
    def get_data_for_scenario(self, scenario, train_teams, test_teams, validation_split, random_seed):
        """Get data splits."""
        train_paths, train_labels = [], []
        test_paths, test_labels, test_teams_info = [], [], []
        
        for team in train_teams:
            if team in self._cache:
                train_paths.extend(self._cache[team]['paths'])
                train_labels.extend(self._cache[team]['labels'])
        
        for team in test_teams:
            if team in self._cache:
                test_paths.extend(self._cache[team]['paths'])
                test_labels.extend(self._cache[team]['labels'])
                test_teams_info.extend([team] * len(self._cache[team]['paths']))
        
        if len(train_paths) > 0 and validation_split > 0:
            train_paths, val_paths, train_labels, val_labels = train_test_split(
                train_paths, train_labels,
                test_size=validation_split,
                random_state=random_seed,
                stratify=train_labels
            )
        else:
            val_paths, val_labels = [], []
        
        return (train_paths, train_labels, val_paths, val_labels,
                test_paths, test_labels, test_teams_info)


def create_dataloader(dataset, batch_size, shuffle, num_workers, pin_memory,
                     persistent_workers, prefetch_factor, drop_last=False):
    """Create a DataLoader with optional persistent workers and prefetching."""
    loader_kwargs = {
        'batch_size': batch_size,
        'shuffle': shuffle,
        'num_workers': num_workers,
        'pin_memory': pin_memory,
        'drop_last': drop_last,  # Drop last incomplete batch to avoid BatchNorm issues
    }
    
    if num_workers > 0:
        loader_kwargs['persistent_workers'] = persistent_workers
        if prefetch_factor is not None:
            loader_kwargs['prefetch_factor'] = prefetch_factor
    
    return DataLoader(dataset, **loader_kwargs)


def evaluate_model(model, data_loader, criterion, device, use_amp=False):
    """Evaluate model on a data loader, returning loss, accuracy, predictions, and probabilities."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_targets = []
    all_probabilities = []
    
    with torch.no_grad():
        for batch in data_loader:
            # Handle both 2-tuple (inputs, targets) and 3-tuple (inputs, targets, paths)
            if len(batch) == 2:
                inputs, targets = batch
            else:
                inputs, targets, _ = batch

            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    avg_loss = total_loss / len(data_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy, all_predictions, all_targets, all_probabilities


def train_model(model, train_loader, val_loader, test_loader, config, device, wandb_run=None):
    """Train model with cosine annealing, early stopping, and optional mixed precision."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    
    if config['scheduler_type'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=1e-6)
    elif config['scheduler_type'] == 'step':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    else:
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    
    use_amp = config.get('use_mixed_precision', False)
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    best_val_acc = 0.0
    best_model_state = None
    patience_counter = 0
    
    history = {
        'train_losses': [], 'train_accs': [],
        'val_losses': [], 'val_accs': [],
        'test_losses': [], 'test_accs': []
    }
    
    print(f"\n{'='*60}")
    print(f"🚀 TRAINING: {config['scenario']} - {config['architecture'].upper()}")
    print(f"{'='*60}")
    print(f"Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)} | Test: {len(test_loader.dataset)}")
    
    for epoch in range(1, config['epochs'] + 1):
        epoch_start = time.time()
        
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{config["epochs"]}', leave=False)
        for inputs, targets in pbar:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad()

            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
            
            pbar.set_postfix({
                'Loss': f'{train_loss/(pbar.n+1):.4f}',
                'Acc': f'{100.*train_correct/train_total:.2f}%'
            })
        
        train_loss /= len(train_loader)
        train_acc = 100. * train_correct / train_total
        
        val_loss, val_acc, val_preds, val_targets, val_probs = evaluate_model(
            model, val_loader, criterion, device, use_amp
        )

        val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(
            val_targets, val_preds, average='weighted', zero_division=0
        )
        
        test_loss, test_acc, _, _, _ = evaluate_model(model, test_loader, criterion, device, use_amp)
        
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        
        history['train_losses'].append(train_loss)
        history['train_accs'].append(train_acc)
        history['val_losses'].append(val_loss)
        history['val_accs'].append(val_acc)
        history['test_losses'].append(test_loss)
        history['test_accs'].append(test_acc)
        
        if wandb_run:
            wandb.log({
                f"{config['scenario']}/epoch": epoch,
                f"{config['scenario']}/train_loss": train_loss,
                f"{config['scenario']}/train_accuracy": train_acc,
                f"{config['scenario']}/val_loss": val_loss,
                f"{config['scenario']}/val_accuracy": val_acc,
                f"{config['scenario']}/val_precision": val_precision * 100,
                f"{config['scenario']}/val_recall": val_recall * 100,
                f"{config['scenario']}/val_f1": val_f1 * 100,
                f"{config['scenario']}/test_loss": test_loss,
                f"{config['scenario']}/test_accuracy": test_acc,
                f"{config['scenario']}/learning_rate": current_lr
            })
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Deep copy tensors to preserve the true best checkpoint state.
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
        
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch:02d} | Train: {train_acc:.2f}% | Val: {val_acc:.2f}% | "
              f"Test: {test_acc:.2f}% | Time: {epoch_time:.1f}s | LR: {current_lr:.6f}")
        
        if patience_counter >= config['patience']:
            print(f"\n✋ Early stopping at epoch {epoch}")
            break
    
    if best_model_state:
        model.load_state_dict(best_model_state)
        print(f"✅ Loaded best model (val_acc: {best_val_acc:.2f}%)")
    
    return model, history, best_val_acc


def plot_training_curves(history, save_path, scenario):
    """Plot training curves."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'{scenario} Training Progress', fontsize=16, fontweight='bold')
    
    epochs = range(1, len(history['train_losses']) + 1)
    
    ax1.plot(epochs, history['train_losses'], 'b-', label='Train', linewidth=2)
    ax1.plot(epochs, history['val_losses'], 'r-', label='Val', linewidth=2)
    ax1.plot(epochs, history['test_losses'], 'g-', label='Test', linewidth=2)
    ax1.set_title('Loss Curves')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(epochs, history['train_accs'], 'b-', label='Train', linewidth=2)
    ax2.plot(epochs, history['val_accs'], 'r-', label='Val', linewidth=2)
    ax2.plot(epochs, history['test_accs'], 'g-', label='Test', linewidth=2)
    ax2.set_title('Accuracy Curves')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    ax3.plot(epochs, history['train_accs'], 'b-', label='Train', linewidth=2)
    ax3.plot(epochs, history['val_accs'], 'r-', label='Val', linewidth=2)
    ax3.set_title('Train vs Validation')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Accuracy (%)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    final_train = history['train_accs'][-1]
    final_val = history['val_accs'][-1]
    final_test = history['test_accs'][-1]
    
    bars = ax4.bar(['Train', 'Val', 'Test'],
                   [final_train, final_val, final_test],
                   color=['blue', 'red', 'green'], alpha=0.7)
    ax4.set_title('Final Accuracies')
    ax4.set_ylabel('Accuracy (%)')
    ax4.grid(True, axis='y', alpha=0.3)
    
    for bar, val in zip(bars, [final_train, final_val, final_test]):
        ax4.text(bar.get_x() + bar.get_width()/2, val + 0.5,
                f'{val:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(f"{save_path}_data.csv")
    
    plt.figure(figsize=(max(12, len(class_names)), max(10, len(class_names))))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{save_path}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return cm


def plot_cross_team_matrix(cross_team_results, teams, save_path, scenario):
    """Plot cross-team accuracy matrix."""
    matrix_data = []
    for train_team in teams:
        row = {'Train_Team': train_team}
        for test_team in teams:
            if train_team in cross_team_results and test_team in cross_team_results[train_team]:
                row[f'Test_{test_team}'] = cross_team_results[train_team][test_team]
            else:
                row[f'Test_{test_team}'] = np.nan
        matrix_data.append(row)
    
    df = pd.DataFrame(matrix_data)
    df.to_csv(f"{save_path}.csv", index=False)
    
    plt.figure(figsize=(12, 10))
    matrix_values = df.drop('Train_Team', axis=1).values
    
    sns.heatmap(matrix_values, annot=True, fmt='.2f', cmap='RdYlGn',
                xticklabels=[t.replace('Test_', '') for t in df.columns[1:]],
                yticklabels=df['Train_Team'],
                cbar_kws={'label': 'Accuracy (%)'}, vmin=0, vmax=100)
    
    plt.title(f'{scenario} Cross-Team Accuracy Matrix', fontsize=16, fontweight='bold')
    plt.xlabel('Test Team', fontsize=12)
    plt.ylabel('Train Team', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{save_path}_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return df


def save_predictions(file_paths, teams_info, predictions, probabilities,
                    true_labels, class_names, save_path):
    """Save predictions to CSV."""
    data = []
    for i, (path, team, pred, true_label) in enumerate(zip(file_paths, teams_info, predictions, true_labels)):
        pred_class = class_names[pred]
        true_class = class_names[true_label]
        confidence = probabilities[i][pred] * 100
        correct = pred == true_label
        
        data.append({
            'file_path': path,
            'team': team,
            'true_label': true_class,
            'predicted_label': pred_class,
            'confidence': confidence,
            'correct': correct
        })
    
    df = pd.DataFrame(data)
    df.to_csv(save_path, index=False)
    
    print(f"✅ Predictions saved: {save_path}")
    return df


def run_toto_experiment(config, data_cache, teams, all_classes, dirs, device):
    """Run TOTO experiment."""
    print(f"\n{'#'*60}")
    print(f"# STARTING TOTO EXPERIMENT")
    print(f"{'#'*60}\n")
    
    train_transform, val_transform = get_transforms(config.INPUT_SIZE, config.USE_DATA_AUGMENTATION)
    all_results = {}
    cross_team_results = {}
    
    for train_team in teams:
        try:
            test_teams = [t for t in teams if t != train_team]
            
            print(f"\n{'='*60}")
            print(f"🔄 TOTO Iteration: Train on {train_team}, Test on {test_teams}")
            print(f"{'='*60}")
            
            if config.USE_WANDB:
                wandb_run = wandb.init(
                    project=config.WANDB_PROJECT,
                    name=f"TOTO_{config.ARCHITECTURE}_train_{train_team}",
                    config={
                        'scenario': config.SCENARIO,
                        'architecture': config.ARCHITECTURE,
                        'train_team': train_team,
                        'test_teams': test_teams
                    },
                    reinit=True
                )
            else:
                wandb_run = None
            
            train_paths, train_labels, val_paths, val_labels, test_paths, test_labels, test_teams_info = \
                data_cache.get_data_for_scenario(config.SCENARIO, [train_team], test_teams, 
                                                config.VALIDATION_SPLIT, config.RANDOM_SEED)
            
            if not train_paths or not val_paths or not test_paths:
                print(f"⚠️ Insufficient data for train team {train_team}, skipping...")
                if wandb_run:
                    wandb.finish()
                continue
            
            train_dataset = PlantDataset(train_paths, train_labels, data_cache.class_to_idx, train_transform)
            val_dataset = PlantDataset(val_paths, val_labels, data_cache.class_to_idx, val_transform)
            test_dataset = PlantDataset(test_paths, test_labels, data_cache.class_to_idx, val_transform)
            
            train_loader = create_dataloader(train_dataset, config.BATCH_SIZE, True, 
                                           config.NUM_WORKERS, config.PIN_MEMORY,
                                           config.PERSISTENT_WORKERS, config.PREFETCH_FACTOR, drop_last=True)
            val_loader = create_dataloader(val_dataset, 4*config.BATCH_SIZE, False,
                                          config.NUM_WORKERS, config.PIN_MEMORY,
                                          config.PERSISTENT_WORKERS, config.PREFETCH_FACTOR, drop_last=False)
            test_loader = create_dataloader(test_dataset, 4*config.BATCH_SIZE, False,
                                           config.NUM_WORKERS, config.PIN_MEMORY,
                                           config.PERSISTENT_WORKERS, config.PREFETCH_FACTOR, drop_last=False)
            
            cleanup_gpu_memory()
            model = get_model(config.ARCHITECTURE, len(all_classes), config.DROPOUT_RATE, 
                            device, config.USE_COMPILE)
            
            train_config = {
                'scenario': config.SCENARIO,
                'architecture': config.ARCHITECTURE,
                'epochs': config.EPOCHS,
                'lr': config.LEARNING_RATE,
                'weight_decay': config.WEIGHT_DECAY,
                'scheduler_type': config.SCHEDULER_TYPE,
                'patience': config.EARLY_STOPPING_PATIENCE,
                'use_mixed_precision': config.USE_MIXED_PRECISION
            }
            
            model, history, best_val_acc = train_model(
                model, train_loader, val_loader, test_loader, train_config, device, wandb_run
            )
            
            print(f"\n{'='*60}")
            print("FINAL EVALUATION")
            print(f"{'='*60}")
            
            test_loss, test_acc, test_preds, test_targets, test_probs = evaluate_model(
                model, test_loader, nn.CrossEntropyLoss(), device, config.USE_MIXED_PRECISION
            )
            
            precision, recall, f1, _ = precision_recall_fscore_support(
                test_targets, test_preds, average='weighted', zero_division=0
            )
            
            final_results = {
                'test_accuracy': test_acc,
                'test_precision': precision * 100,
                'test_recall': recall * 100,
                'test_f1': f1 * 100,
                'best_val_accuracy': best_val_acc
            }
            
            print(f"✅ Test Accuracy: {test_acc:.2f}%")
            print(f"✅ Test F1-Score: {f1*100:.2f}%")
            print(f"✅ Best Val Accuracy: {best_val_acc:.2f}%")
            
            curves_path = dirs['plots'] / f"TOTO_{config.ARCHITECTURE}_train{train_team}_training_curves.png"
            plot_training_curves(history, str(curves_path), config.SCENARIO)
            
            cm_path = dirs['plots'] / f"TOTO_{config.ARCHITECTURE}_train{train_team}_confusion_matrix"
            plot_confusion_matrix(test_targets, test_preds, all_classes, str(cm_path))
            
            # Get predictions for train, val, and test sets
            # Train set predictions
            train_dataset_paths = PlantDataset(train_paths, train_labels, data_cache.class_to_idx, 
                                              val_transform, return_paths=True)
            train_loader_paths = create_dataloader(train_dataset_paths, config.BATCH_SIZE, False,
                                                  config.NUM_WORKERS, False, False, None, drop_last=False)
            
            _, _, train_preds, train_targets, train_probs = evaluate_model(
                model, train_loader_paths, nn.CrossEntropyLoss(), device, config.USE_MIXED_PRECISION
            )
            
            train_file_paths = []
            train_teams_info = []
            for _, _, paths in train_loader_paths:
                train_file_paths.extend(paths)
                for path in paths:
                    team_name = Path(path).parent.parent.name
                    train_teams_info.append(team_name)
            
            # Val set predictions
            val_dataset_paths = PlantDataset(val_paths, val_labels, data_cache.class_to_idx, 
                                            val_transform, return_paths=True)
            val_loader_paths = create_dataloader(val_dataset_paths, config.BATCH_SIZE, False,
                                                config.NUM_WORKERS, False, False, None, drop_last=False)
            
            _, _, val_preds, val_targets, val_probs = evaluate_model(
                model, val_loader_paths, nn.CrossEntropyLoss(), device, config.USE_MIXED_PRECISION
            )
            
            val_file_paths = []
            val_teams_info = []
            for _, _, paths in val_loader_paths:
                val_file_paths.extend(paths)
                for path in paths:
                    team_name = Path(path).parent.parent.name
                    val_teams_info.append(team_name)
            
            # Test set predictions
            test_dataset_paths = PlantDataset(test_paths, test_labels, data_cache.class_to_idx, 
                                             val_transform, return_paths=True)
            test_loader_paths = create_dataloader(test_dataset_paths, config.BATCH_SIZE, False,
                                                 config.NUM_WORKERS, False, False, None, drop_last=False)
            
            test_file_paths = []
            test_teams_info = []
            for _, _, paths in test_loader_paths:
                test_file_paths.extend(paths)
                for path in paths:
                    team_name = Path(path).parent.parent.name
                    test_teams_info.append(team_name)
            
            # Combine all predictions with split labels
            all_file_paths = train_file_paths[:len(train_preds)] + val_file_paths[:len(val_preds)] + test_file_paths[:len(test_preds)]
            all_teams_info = train_teams_info[:len(train_preds)] + val_teams_info[:len(val_preds)] + test_teams_info[:len(test_preds)]
            all_preds = np.concatenate([train_preds, val_preds, test_preds])
            all_targets = np.concatenate([train_targets, val_targets, test_targets])
            all_probs = np.concatenate([train_probs, val_probs, test_probs])
            all_splits = ['train'] * len(train_preds) + ['val'] * len(val_preds) + ['test'] * len(test_preds)
            
            pred_path = dirs['predictions'] / f"TOTO_{config.ARCHITECTURE}_train{train_team}_predictions.csv"
            save_predictions_with_split(
                all_file_paths, all_teams_info, all_preds, all_probs, 
                all_targets, all_classes, all_splits, str(pred_path)
            )
            
            results_file = dirs['results'] / f"TOTO_{config.ARCHITECTURE}_train{train_team}_results.json"
            with open(results_file, 'w') as f:
                json.dump({
                    'scenario': config.SCENARIO,
                    'architecture': config.ARCHITECTURE,
                    'train_team': train_team,
                    'test_teams': test_teams,
                    'final_results': final_results
                }, f, indent=2)
            
            all_results[train_team] = final_results
            
            # Build cross-team results matrix
            if train_team not in cross_team_results:
                cross_team_results[train_team] = {}
            
            for test_team in test_teams:
                # Calculate accuracy for each test team
                team_mask = np.array([team == test_team for team in test_teams_info[:len(test_preds)]])
                if team_mask.sum() > 0:
                    team_acc = 100.0 * (np.array(test_preds)[team_mask] == np.array(test_targets)[team_mask]).sum() / team_mask.sum()
                    cross_team_results[train_team][test_team] = team_acc
            
            model_path = dirs['models'] / f"TOTO_{config.ARCHITECTURE}_train{train_team}_best.pth"
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': train_config,
                'results': final_results
            }, model_path)
            
            if wandb_run:
                wandb.log({
                    f"{config.SCENARIO}/final_test_accuracy": test_acc,
                    f"{config.SCENARIO}/final_test_f1": f1 * 100
                })
                wandb.finish()
            
            del model, train_loader, val_loader, test_loader
            cleanup_gpu_memory()
        
        except Exception as e:
            print(f"✗ Error with train team {train_team}: {e}")
            import traceback
            traceback.print_exc()
            if wandb_run:
                wandb.finish()
            continue

    if all_results:
        accuracies = [r['test_accuracy'] for r in all_results.values()]
        toto_summary = {
            'scenario': config.SCENARIO,
            'architecture': config.ARCHITECTURE,
            'results': all_results,
            'avg_accuracy': np.mean(accuracies),
            'std_accuracy': np.std(accuracies)
        }
        
        summary_file = dirs['results'] / f"TOTO_{config.ARCHITECTURE}_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(toto_summary, f, indent=2)
        
        # Plot cross-team matrix
        matrix_path = dirs['matrices'] / f"TOTO_{config.ARCHITECTURE}_cross_team_matrix"
        plot_cross_team_matrix(cross_team_results, teams, str(matrix_path), config.SCENARIO)
        print(f"✅ Cross-team matrix saved")
        
        print(f"\n{'='*60}")
        print(f"TOTO SUMMARY - {config.ARCHITECTURE.upper()}")
        print(f"{'='*60}")
        print(f"Average Accuracy: {toto_summary['avg_accuracy']:.2f}% ± {toto_summary['std_accuracy']:.2f}%")
    
    return all_results


def run_loto_experiment(config, data_cache, teams, all_classes, dirs, device):
    """Run LOTO experiment."""
    print(f"\n{'#'*60}")
    print(f"# STARTING LOTO EXPERIMENT")
    print(f"{'#'*60}\n")
    
    train_transform, val_transform = get_transforms(config.INPUT_SIZE, config.USE_DATA_AUGMENTATION)
    all_results = {}
    
    for held_out_team in teams:
        try:
            train_teams = [t for t in teams if t != held_out_team]
            
            print(f"\n{'='*60}")
            print(f"🔄 LOTO Iteration: Train on {train_teams}, Test on {held_out_team}")
            print(f"{'='*60}")
            
            if config.USE_WANDB:
                wandb_run = wandb.init(
                    project=config.WANDB_PROJECT,
                    name=f"LOTO_{config.ARCHITECTURE}_heldout_{held_out_team}",
                    config={
                        'scenario': config.SCENARIO,
                        'architecture': config.ARCHITECTURE,
                        'train_teams': train_teams,
                        'held_out_team': held_out_team
                    },
                    reinit=True
                )
            else:
                wandb_run = None
            
            train_paths, train_labels, val_paths, val_labels, test_paths, test_labels, _ = \
                data_cache.get_data_for_scenario(config.SCENARIO, train_teams, [held_out_team],
                                                config.VALIDATION_SPLIT, config.RANDOM_SEED)
            
            if not train_paths or not val_paths or not test_paths:
                print(f"⚠️ Insufficient data for held-out team {held_out_team}, skipping...")
                if wandb_run:
                    wandb.finish()
                continue
            
            train_dataset = PlantDataset(train_paths, train_labels, data_cache.class_to_idx, train_transform)
            val_dataset = PlantDataset(val_paths, val_labels, data_cache.class_to_idx, val_transform)
            test_dataset = PlantDataset(test_paths, test_labels, data_cache.class_to_idx, val_transform)
            
            train_loader = create_dataloader(train_dataset, config.BATCH_SIZE, True,
                                           config.NUM_WORKERS, config.PIN_MEMORY,
                                           config.PERSISTENT_WORKERS, config.PREFETCH_FACTOR, drop_last=True)
            val_loader = create_dataloader(val_dataset, config.BATCH_SIZE, False,
                                          config.NUM_WORKERS, config.PIN_MEMORY,
                                          config.PERSISTENT_WORKERS, config.PREFETCH_FACTOR, drop_last=False)
            test_loader = create_dataloader(test_dataset, config.BATCH_SIZE, False,
                                           config.NUM_WORKERS, config.PIN_MEMORY,
                                           config.PERSISTENT_WORKERS, config.PREFETCH_FACTOR, drop_last=False)
            
            cleanup_gpu_memory()
            model = get_model(config.ARCHITECTURE, len(all_classes), config.DROPOUT_RATE,
                            device, config.USE_COMPILE)
            
            train_config = {
                'scenario': config.SCENARIO,
                'architecture': config.ARCHITECTURE,
                'epochs': config.EPOCHS,
                'lr': config.LEARNING_RATE,
                'weight_decay': config.WEIGHT_DECAY,
                'scheduler_type': config.SCHEDULER_TYPE,
                'patience': config.EARLY_STOPPING_PATIENCE,
                'use_mixed_precision': config.USE_MIXED_PRECISION
            }
            
            model, history, best_val_acc = train_model(
                model, train_loader, val_loader, test_loader, train_config, device, wandb_run
            )
            
            print(f"\n{'='*60}")
            print("FINAL EVALUATION")
            print(f"{'='*60}")
            
            test_loss, test_acc, test_preds, test_targets, test_probs = evaluate_model(
                model, test_loader, nn.CrossEntropyLoss(), device, config.USE_MIXED_PRECISION
            )
            
            precision, recall, f1, _ = precision_recall_fscore_support(
                test_targets, test_preds, average='weighted', zero_division=0
            )
            
            final_results = {
                'test_accuracy': test_acc,
                'test_precision': precision * 100,
                'test_recall': recall * 100,
                'test_f1': f1 * 100,
                'best_val_accuracy': best_val_acc
            }
            
            print(f"✅ Test Accuracy: {test_acc:.2f}%")
            print(f"✅ Test F1-Score: {f1*100:.2f}%")
            print(f"✅ Best Val Accuracy: {best_val_acc:.2f}%")
            
            curves_path = dirs['plots'] / f"LOTO_{config.ARCHITECTURE}_heldout{held_out_team}_training_curves.png"
            plot_training_curves(history, str(curves_path), config.SCENARIO)
            
            cm_path = dirs['plots'] / f"LOTO_{config.ARCHITECTURE}_heldout{held_out_team}_confusion_matrix"
            plot_confusion_matrix(test_targets, test_preds, all_classes, str(cm_path))
            
            # Get predictions for train, val, and test sets
            # Train set predictions
            train_dataset_paths = PlantDataset(train_paths, train_labels, data_cache.class_to_idx, 
                                              val_transform, return_paths=True)
            train_loader_paths = create_dataloader(train_dataset_paths, config.BATCH_SIZE, False,
                                                  config.NUM_WORKERS, False, False, None, drop_last=False)
            
            _, _, train_preds, train_targets, train_probs = evaluate_model(
                model, train_loader_paths, nn.CrossEntropyLoss(), device, config.USE_MIXED_PRECISION
            )
            
            train_file_paths = []
            train_teams_info = []
            for _, _, paths in train_loader_paths:
                train_file_paths.extend(paths)
                for path in paths:
                    team_name = Path(path).parent.parent.name
                    train_teams_info.append(team_name)
            
            # Val set predictions
            val_dataset_paths = PlantDataset(val_paths, val_labels, data_cache.class_to_idx, 
                                            val_transform, return_paths=True)
            val_loader_paths = create_dataloader(val_dataset_paths, config.BATCH_SIZE, False,
                                                config.NUM_WORKERS, False, False, None, drop_last=False)
            
            _, _, val_preds, val_targets, val_probs = evaluate_model(
                model, val_loader_paths, nn.CrossEntropyLoss(), device, config.USE_MIXED_PRECISION
            )
            
            val_file_paths = []
            val_teams_info = []
            for _, _, paths in val_loader_paths:
                val_file_paths.extend(paths)
                for path in paths:
                    team_name = Path(path).parent.parent.name
                    val_teams_info.append(team_name)
            
            # Test set predictions
            test_dataset_paths = PlantDataset(test_paths, test_labels, data_cache.class_to_idx,
                                             val_transform, return_paths=True)
            test_loader_paths = create_dataloader(test_dataset_paths, config.BATCH_SIZE, False,
                                                 config.NUM_WORKERS, False, False, None, drop_last=False)
            
            test_file_paths = []
            test_teams_info = []
            for _, _, paths in test_loader_paths:
                test_file_paths.extend(paths)
                for path in paths:
                    team_name = Path(path).parent.parent.name
                    test_teams_info.append(team_name)
            
            # Combine all predictions with split labels
            all_file_paths = train_file_paths[:len(train_preds)] + val_file_paths[:len(val_preds)] + test_file_paths[:len(test_preds)]
            all_teams_info = train_teams_info[:len(train_preds)] + val_teams_info[:len(val_preds)] + test_teams_info[:len(test_preds)]
            all_preds = np.concatenate([train_preds, val_preds, test_preds])
            all_targets = np.concatenate([train_targets, val_targets, test_targets])
            all_probs = np.concatenate([train_probs, val_probs, test_probs])
            all_splits = ['train'] * len(train_preds) + ['val'] * len(val_preds) + ['test'] * len(test_preds)
            
            pred_path = dirs['predictions'] / f"LOTO_{config.ARCHITECTURE}_heldout{held_out_team}_predictions.csv"
            save_predictions_with_split(
                all_file_paths, all_teams_info, all_preds, all_probs, 
                all_targets, all_classes, all_splits, str(pred_path)
            )
            
            results_file = dirs['results'] / f"LOTO_{config.ARCHITECTURE}_heldout{held_out_team}_results.json"
            with open(results_file, 'w') as f:
                json.dump({
                    'scenario': config.SCENARIO,
                    'architecture': config.ARCHITECTURE,
                    'train_teams': train_teams,
                    'held_out_team': held_out_team,
                    'final_results': final_results
                }, f, indent=2)
            
            all_results[held_out_team] = final_results
            
            model_path = dirs['models'] / f"LOTO_{config.ARCHITECTURE}_heldout{held_out_team}_best.pth"
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': train_config,
                'results': final_results
            }, model_path)
            
            if wandb_run:
                wandb.log({
                    f"{config.SCENARIO}/final_test_accuracy": test_acc,
                    f"{config.SCENARIO}/final_test_f1": f1 * 100
                })
                wandb.finish()
            
            del model, train_loader, val_loader, test_loader
            cleanup_gpu_memory()
        
        except Exception as e:
            print(f"✗ Error with held-out team {held_out_team}: {e}")
            import traceback
            traceback.print_exc()
            if wandb_run:
                wandb.finish()
            continue

    if all_results:
        accuracies = [r['test_accuracy'] for r in all_results.values()]
        loto_summary = {
            'scenario': config.SCENARIO,
            'architecture': config.ARCHITECTURE,
            'results': all_results,
            'avg_accuracy': np.mean(accuracies),
            'std_accuracy': np.std(accuracies)
        }
        
        summary_file = dirs['results'] / f"LOTO_{config.ARCHITECTURE}_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(loto_summary, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"LOTO SUMMARY - {config.ARCHITECTURE.upper()}")
        print(f"{'='*60}")
        print(f"Average Accuracy: {loto_summary['avg_accuracy']:.2f}% ± {loto_summary['std_accuracy']:.2f}%")
    
    return all_results


def save_predictions_with_split(file_paths, teams_info, predictions, probabilities,
                                true_labels, class_names, splits, save_path):
    """Save predictions to CSV with split information."""
    data = []
    for i, (path, team, pred, true_label, split) in enumerate(zip(file_paths, teams_info, predictions, true_labels, splits)):
        pred_class = class_names[pred]
        true_class = class_names[true_label]
        confidence = probabilities[i][pred] * 100
        correct = pred == true_label
        
        data.append({
            'file_path': path,
            'team': team,
            'split': split,
            'true_label': true_class,
            'predicted_label': pred_class,
            'confidence': confidence,
            'correct': correct
        })
    
    df = pd.DataFrame(data)
    df.to_csv(save_path, index=False)
    
    print(f"✅ Predictions saved: {save_path}")
    return df

def main():
    """Main execution function."""
    # Parse command-line arguments
    args = parse_args()
    
    # Create config from arguments
    config = Config.from_args(args)
    config.print_config()
    
    # Set random seeds
    set_random_seeds(config.RANDOM_SEED)
    print(f"✅ Random seeds set to {config.RANDOM_SEED}")
    
    # Create directories
    dirs = create_directories(config.OUTPUT_DIR)
    print(f"✅ Output directories created")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"✅ Using device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Compute Capability: {torch.cuda.get_device_capability()}")
    
    # Initialize W&B
    if config.USE_WANDB:
        try:
            wandb.login()
            print("W&B login successful.")
        except Exception as e:
            print(f"W&B login failed: {e}")
            config.USE_WANDB = False
    
    # Analyze dataset
    teams, all_classes = analyze_dataset(config.DATA_DIR)
    
    # Create data cache
    data_cache = DataCache(config.DATA_DIR, all_classes)
    
    # Run experiment
    if config.SCENARIO == "TOTO":
        all_results = run_toto_experiment(config, data_cache, teams, all_classes, dirs, device)
    elif config.SCENARIO == "LOTO":
        all_results = run_loto_experiment(config, data_cache, teams, all_classes, dirs, device)
    else:
        raise ValueError(f"Unknown scenario: {config.SCENARIO}")
    
    # Print final summary
    print(f"\n{'#'*60}")
    print("# EXPERIMENT COMPLETED SUCCESSFULLY! 🎉")
    print(f"{'#'*60}")
    print(f"\n📁 All results saved to: {config.OUTPUT_DIR}")
    print(f"\n📊 Summary:")
    print(f"   Scenario: {config.SCENARIO}")
    print(f"   Architecture: {config.ARCHITECTURE}")
    print(f"   Number of experiments: {len(all_results)}")
    
    if all_results:
        accuracies = [r['test_accuracy'] for r in all_results.values()]
        print(f"\n   Average Test Accuracy: {np.mean(accuracies):.2f}% ± {np.std(accuracies):.2f}%")
        print(f"   Best: {np.max(accuracies):.2f}%")
        print(f"   Worst: {np.min(accuracies):.2f}%")


if __name__ == "__main__":
    main()