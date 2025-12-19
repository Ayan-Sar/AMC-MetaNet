"""
Logging utilities for training and evaluation.
"""

import os
import sys
import logging
from datetime import datetime
from typing import Optional, Dict, Any

import torch


def setup_logger(
    name: str = 'AMC-MetaNet',
    log_dir: Optional[str] = None,
    level: int = logging.INFO,
    console: bool = True
) -> logging.Logger:
    """
    Set up logger with file and console handlers.
    
    Args:
        name: Logger name
        log_dir: Directory for log files
        level: Logging level
        console: Whether to log to console
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    
    # Formatter
    formatter = logging.Formatter(
        '[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(log_dir, f'{name}_{timestamp}.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


class TensorBoardLogger:
    """
    TensorBoard logging wrapper.
    """
    
    def __init__(self, log_dir: str, enabled: bool = True):
        """
        Args:
            log_dir: Directory for TensorBoard logs
            enabled: Whether logging is enabled
        """
        self.enabled = enabled
        self.log_dir = log_dir
        
        if enabled:
            from torch.utils.tensorboard import SummaryWriter
            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir)
        else:
            self.writer = None
    
    def log_scalar(self, tag: str, value: float, step: int):
        """Log a scalar value."""
        if self.writer:
            self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float], step: int):
        """Log multiple scalars with a common main tag."""
        if self.writer:
            self.writer.add_scalars(main_tag, tag_scalar_dict, step)
    
    def log_histogram(self, tag: str, values: torch.Tensor, step: int):
        """Log a histogram."""
        if self.writer:
            self.writer.add_histogram(tag, values, step)
    
    def log_image(self, tag: str, img_tensor: torch.Tensor, step: int):
        """Log an image."""
        if self.writer:
            self.writer.add_image(tag, img_tensor, step)
    
    def log_text(self, tag: str, text: str, step: int):
        """Log text."""
        if self.writer:
            self.writer.add_text(tag, text, step)
    
    def log_hyperparams(self, hparams: Dict[str, Any], metrics: Dict[str, float]):
        """Log hyperparameters and associated metrics."""
        if self.writer:
            self.writer.add_hparams(hparams, metrics)
    
    def log_model_graph(self, model: torch.nn.Module, input_tensor: torch.Tensor):
        """Log model graph."""
        if self.writer:
            self.writer.add_graph(model, input_tensor)
    
    def log_embedding(
        self,
        tag: str,
        embeddings: torch.Tensor,
        metadata: Optional[list] = None,
        step: int = 0
    ):
        """Log embeddings for visualization."""
        if self.writer:
            self.writer.add_embedding(
                embeddings, metadata=metadata, tag=tag, global_step=step
            )
    
    def flush(self):
        """Flush the writer."""
        if self.writer:
            self.writer.flush()
    
    def close(self):
        """Close the writer."""
        if self.writer:
            self.writer.close()


class ProgressLogger:
    """
    Console progress logger with formatting.
    """
    
    def __init__(self, total_epochs: int, episodes_per_epoch: int):
        self.total_epochs = total_epochs
        self.episodes_per_epoch = episodes_per_epoch
    
    def log_episode(
        self,
        epoch: int,
        episode: int,
        loss: float,
        accuracy: float,
        lr: float
    ):
        """Log episode progress."""
        progress = f'[Epoch {epoch}/{self.total_epochs}]'
        progress += f' [{episode}/{self.episodes_per_epoch}]'
        progress += f' Loss: {loss:.4f}'
        progress += f' Acc: {accuracy:.2f}%'
        progress += f' LR: {lr:.6f}'
        print(f'\r{progress}', end='', flush=True)
    
    def log_epoch_end(
        self,
        epoch: int,
        train_loss: float,
        train_acc: float,
        val_acc: float,
        best_acc: float
    ):
        """Log epoch summary."""
        print()  # New line after episode progress
        print(f'Epoch {epoch} Summary:')
        print(f'  Train Loss: {train_loss:.4f}')
        print(f'  Train Acc:  {train_acc:.2f}%')
        print(f'  Val Acc:    {val_acc:.2f}%')
        print(f'  Best Acc:   {best_acc:.2f}%')
        print('-' * 50)
