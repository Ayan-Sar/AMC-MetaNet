"""
Adaptive Metric Module for AMC-MetaNet.

Learns task-adaptive distance metrics using attention-based feature reweighting.
This allows the model to focus on task-relevant features for each episode.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class ChannelAttention(nn.Module):
    """
    Channel attention mechanism.
    
    Learns to weight different feature channels based on their importance
    for the current task.
    """
    
    def __init__(self, feature_dim: int, reduction: int = 4):
        super().__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim // reduction, feature_dim, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Feature tensor of shape (B, D)
            
        Returns:
            Attention weights of shape (B, D)
        """
        return self.fc(x)


class TaskConditioner(nn.Module):
    """
    Task conditioning module.
    
    Generates task-specific parameters based on the support set.
    """
    
    def __init__(self, feature_dim: int, hidden_dim: int = 128):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feature_dim),
            nn.Sigmoid()
        )
    
    def forward(self, support_features: torch.Tensor) -> torch.Tensor:
        """
        Generate task-specific conditioning.
        
        Args:
            support_features: Support set features (N*K, D)
            
        Returns:
            Task conditioning vector (D,)
        """
        # Aggregate support features
        task_representation = support_features.mean(dim=0)
        return self.mlp(task_representation)


class AdaptiveMetricModule(nn.Module):
    """
    Adaptive Metric Module for task-specific distance computation.
    
    Key components:
    1. Channel attention for feature reweighting
    2. Task conditioning from support set
    3. Learnable temperature scaling
    4. Multiple distance metric options
    """
    
    def __init__(
        self,
        feature_dim: int = 640,
        attention_dim: int = 128,
        temperature: float = 1.0,
        use_attention: bool = True,
        use_task_conditioning: bool = True,
        metric: str = 'euclidean'
    ):
        """
        Args:
            feature_dim: Dimension of input features
            attention_dim: Hidden dimension for attention
            temperature: Initial temperature for scaling
            use_attention: Whether to use channel attention
            use_task_conditioning: Whether to use task conditioning
            metric: Distance metric ('euclidean', 'cosine', 'mahalanobis')
        """
        super().__init__()
        
        self.feature_dim = feature_dim
        self.metric = metric
        self.use_attention = use_attention
        self.use_task_conditioning = use_task_conditioning
        
        # Learnable temperature
        self.temperature = nn.Parameter(torch.tensor(temperature))
        
        # Channel attention
        if use_attention:
            self.channel_attention = ChannelAttention(
                feature_dim, reduction=4
            )
        
        # Task conditioning
        if use_task_conditioning:
            self.task_conditioner = TaskConditioner(
                feature_dim, attention_dim
            )
        
        # For Mahalanobis distance
        if metric == 'mahalanobis':
            self.metric_transform = nn.Linear(feature_dim, feature_dim, bias=False)
            # Initialize as identity
            nn.init.eye_(self.metric_transform.weight)
    
    def compute_prototypes(
        self,
        support_features: torch.Tensor,
        support_labels: torch.Tensor,
        n_way: int
    ) -> torch.Tensor:
        """
        Compute class prototypes from support set.
        
        Args:
            support_features: Feature tensor (N*K, D)
            support_labels: Label tensor (N*K,)
            n_way: Number of classes
            
        Returns:
            Prototypes tensor (N, D)
        """
        prototypes = []
        for cls in range(n_way):
            cls_mask = support_labels == cls
            cls_features = support_features[cls_mask]
            prototype = cls_features.mean(dim=0)
            prototypes.append(prototype)
        
        return torch.stack(prototypes)  # (N, D)
    
    def apply_attention(
        self,
        features: torch.Tensor,
        support_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply channel attention to features.
        
        Args:
            features: Features to reweight (B, D)
            support_features: Support features for context (N*K, D)
            
        Returns:
            Reweighted features (B, D)
        """
        if not self.use_attention:
            return features
        
        # Compute attention from support set context
        context = support_features.mean(dim=0, keepdim=True)
        attention_weights = self.channel_attention(context)
        
        return features * attention_weights
    
    def apply_task_conditioning(
        self,
        features: torch.Tensor,
        support_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply task-specific conditioning.
        
        Args:
            features: Features to condition (B, D)
            support_features: Support features (N*K, D)
            
        Returns:
            Conditioned features (B, D)
        """
        if not self.use_task_conditioning:
            return features
        
        conditioning = self.task_conditioner(support_features)
        return features * conditioning
    
    def compute_distance(
        self,
        query_features: torch.Tensor,
        prototypes: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute distance between query features and prototypes.
        
        Args:
            query_features: Query feature tensor (Q, D)
            prototypes: Class prototypes (N, D)
            
        Returns:
            Distance matrix (Q, N)
        """
        if self.metric == 'euclidean':
            # Euclidean distance
            distances = torch.cdist(query_features, prototypes, p=2)
        
        elif self.metric == 'cosine':
            # Cosine similarity (converted to distance)
            query_norm = F.normalize(query_features, dim=-1)
            proto_norm = F.normalize(prototypes, dim=-1)
            similarities = torch.mm(query_norm, proto_norm.t())
            distances = 1 - similarities
        
        elif self.metric == 'mahalanobis':
            # Mahalanobis-like distance with learned transform
            query_transformed = self.metric_transform(query_features)
            proto_transformed = self.metric_transform(prototypes)
            distances = torch.cdist(query_transformed, proto_transformed, p=2)
        
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
        
        return distances
    
    def forward(
        self,
        support_features: torch.Tensor,
        support_labels: torch.Tensor,
        query_features: torch.Tensor,
        n_way: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: compute logits for query samples.
        
        Args:
            support_features: Support set features (N*K, D)
            support_labels: Support set labels (N*K,)
            query_features: Query set features (Q, D)
            n_way: Number of classes
            
        Returns:
            logits: Classification logits (Q, N)
            prototypes: Class prototypes (N, D)
        """
        # Apply attention and conditioning
        support_features = self.apply_attention(
            support_features, support_features
        )
        support_features = self.apply_task_conditioning(
            support_features, support_features
        )
        
        query_features = self.apply_attention(
            query_features, support_features
        )
        query_features = self.apply_task_conditioning(
            query_features, support_features
        )
        
        # Compute prototypes
        prototypes = self.compute_prototypes(
            support_features, support_labels, n_way
        )
        
        # Compute distances
        distances = self.compute_distance(query_features, prototypes)
        
        # Convert to logits with temperature scaling
        logits = -distances / self.temperature
        
        return logits, prototypes
