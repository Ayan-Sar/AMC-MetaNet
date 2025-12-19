"""
Model module for AMC-MetaNet.
Contains backbone networks, metric learning modules, and the main model.
"""

from .backbone import ResNet12, Conv4, get_backbone
from .metric_module import AdaptiveMetricModule
from .amc_metanet import AMCMetaNet

__all__ = [
    'ResNet12',
    'Conv4',
    'get_backbone',
    'AdaptiveMetricModule',
    'AMCMetaNet'
]
