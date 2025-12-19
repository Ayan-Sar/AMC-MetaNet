"""
Evaluation metrics for few-shot learning.
"""

import numpy as np
from typing import Tuple, List, Optional

import torch


def compute_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor
) -> float:
    """
    Compute classification accuracy.
    
    Args:
        logits: Prediction logits (B, N)
        labels: Ground truth labels (B,)
        
    Returns:
        Accuracy as a percentage
    """
    predictions = logits.argmax(dim=1)
    correct = (predictions == labels).float().sum()
    accuracy = correct / labels.size(0) * 100
    return accuracy.item()


def compute_confidence_interval(
    accuracies: List[float],
    confidence: float = 0.95
) -> Tuple[float, float]:
    """
    Compute mean accuracy with confidence interval.
    
    Args:
        accuracies: List of accuracy values from multiple episodes
        confidence: Confidence level (default 0.95 for 95% CI)
        
    Returns:
        Tuple of (mean, confidence_interval)
    """
    from scipy import stats
    
    accuracies = np.array(accuracies)
    n = len(accuracies)
    mean = np.mean(accuracies)
    std = np.std(accuracies, ddof=1)
    
    # Compute t-statistic for confidence interval
    t_value = stats.t.ppf((1 + confidence) / 2, n - 1)
    ci = t_value * std / np.sqrt(n)
    
    return mean, ci


def compute_per_class_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    n_classes: int
) -> List[float]:
    """
    Compute per-class accuracy.
    
    Args:
        logits: Prediction logits (B, N)
        labels: Ground truth labels (B,)
        n_classes: Number of classes
        
    Returns:
        List of per-class accuracies
    """
    predictions = logits.argmax(dim=1)
    
    per_class_acc = []
    for cls in range(n_classes):
        mask = labels == cls
        if mask.sum() > 0:
            cls_correct = (predictions[mask] == cls).float().sum()
            cls_acc = cls_correct / mask.sum() * 100
            per_class_acc.append(cls_acc.item())
        else:
            per_class_acc.append(0.0)
    
    return per_class_acc


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking metrics during training.
    """
    
    def __init__(self, name: str = 'Metric'):
        self.name = name
        self.reset()
    
    def reset(self):
        """Reset all statistics."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.history = []
    
    def update(self, val: float, n: int = 1):
        """
        Update the meter with a new value.
        
        Args:
            val: New value
            n: Number of samples (for weighted average)
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.history.append(val)
    
    def __str__(self) -> str:
        return f'{self.name}: {self.val:.4f} (avg: {self.avg:.4f})'


class MetricTracker:
    """
    Tracks multiple metrics during training.
    """
    
    def __init__(self, *metrics: str):
        self.metrics = {name: AverageMeter(name) for name in metrics}
    
    def update(self, **kwargs):
        """
        Update metrics.
        
        Args:
            **kwargs: metric_name=value pairs
        """
        for name, value in kwargs.items():
            if name in self.metrics:
                self.metrics[name].update(value)
    
    def reset(self):
        """Reset all metrics."""
        for meter in self.metrics.values():
            meter.reset()
    
    def get_averages(self) -> dict:
        """Get average values for all metrics."""
        return {name: meter.avg for name, meter in self.metrics.items()}
    
    def __getitem__(self, name: str) -> AverageMeter:
        return self.metrics[name]


def compute_few_shot_metrics(
    episode_results: List[dict]
) -> dict:
    """
    Compute aggregate metrics from episode results.
    
    Args:
        episode_results: List of dicts with 'accuracy' key
        
    Returns:
        Dictionary with mean, std, and 95% CI
    """
    accuracies = [r['accuracy'] for r in episode_results]
    mean, ci = compute_confidence_interval(accuracies)
    
    return {
        'mean_accuracy': mean,
        'std_accuracy': np.std(accuracies),
        'ci_95': ci,
        'min_accuracy': min(accuracies),
        'max_accuracy': max(accuracies),
        'n_episodes': len(accuracies)
    }
