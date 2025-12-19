"""
Helper functions for training and evaluation.
"""

import os
import random
from typing import Dict, Any, Optional

import numpy as np
import torch
import yaml


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], save_path: str):
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        save_path: Path to save config
    """
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def save_checkpoint(
    state: Dict[str, Any],
    save_dir: str,
    filename: str = 'checkpoint.pth',
    is_best: bool = False
):
    """
    Save training checkpoint.
    
    Args:
        state: State dictionary containing model, optimizer, etc.
        save_dir: Directory to save checkpoint
        filename: Checkpoint filename
        is_best: Whether this is the best model so far
    """
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    torch.save(state, filepath)
    
    if is_best:
        best_path = os.path.join(save_dir, 'best_model.pth')
        torch.save(state, best_path)


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = 'cpu'
) -> Dict[str, Any]:
    """
    Load training checkpoint.
    
    Args:
        model: Model to load weights into
        checkpoint_path: Path to checkpoint file
        optimizer: Optional optimizer to load state
        device: Device to load checkpoint to
        
    Returns:
        Checkpoint dictionary with metadata
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint


def get_device() -> torch.device:
    """
    Get the best available device.
    
    Returns:
        torch.device for cuda if available, else cpu
    """
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f'Using GPU: {torch.cuda.get_device_name(0)}')
    else:
        device = torch.device('cpu')
        print('Using CPU')
    return device


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(model: torch.nn.Module, input_size: tuple = (3, 84, 84)):
    """
    Print a summary of the model architecture.
    
    Args:
        model: PyTorch model
        input_size: Input tensor size (C, H, W)
    """
    print('=' * 60)
    print('Model Summary')
    print('=' * 60)
    print(model)
    print('=' * 60)
    print(f'Total Parameters: {count_parameters(model):,}')
    print('=' * 60)


class EarlyStopping:
    """
    Early stopping to terminate training when validation loss stops improving.
    """
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = 'max'
    ):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as an improvement
            mode: 'min' for loss, 'max' for accuracy
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current metric value
            
        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'max':
            if score > self.best_score + self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        else:  # mode == 'min'
            if score < self.best_score - self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        
        if self.counter >= self.patience:
            self.early_stop = True
            return True
        
        return False


def get_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str = 'step',
    **kwargs
) -> torch.optim.lr_scheduler._LRScheduler:
    """
    Get learning rate scheduler.
    
    Args:
        optimizer: PyTorch optimizer
        scheduler_type: Type of scheduler ('step', 'cosine', 'plateau')
        **kwargs: Scheduler-specific arguments
        
    Returns:
        Learning rate scheduler
    """
    schedulers = {
        'step': torch.optim.lr_scheduler.StepLR,
        'cosine': torch.optim.lr_scheduler.CosineAnnealingLR,
        'plateau': torch.optim.lr_scheduler.ReduceLROnPlateau,
        'exponential': torch.optim.lr_scheduler.ExponentialLR
    }
    
    if scheduler_type not in schedulers:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    return schedulers[scheduler_type](optimizer, **kwargs)
