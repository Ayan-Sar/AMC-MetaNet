"""
Testing/Evaluation script for AMC-MetaNet.

Evaluates trained model on the test set with confidence intervals.

Usage:
    python test.py --checkpoint checkpoints/best_model.pth
    python test.py --checkpoint checkpoints/best_model.pth --n_shot 1
"""

import argparse
import os
import json
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import RSDataset, EpisodeDataset, get_test_transforms
from models import AMCMetaNet
from utils import (
    set_seed, load_config, get_device,
    compute_accuracy, compute_confidence_interval, compute_few_shot_metrics,
    setup_logger
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate AMC-MetaNet')
    
    # Required
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    
    # Dataset
    parser.add_argument('--dataset', type=str, default=None,
                        help='Dataset name (uses checkpoint config if not specified)')
    parser.add_argument('--data_root', type=str, default='./data',
                        help='Data root directory')
    
    # Few-shot settings
    parser.add_argument('--n_way', type=int, default=5,
                        help='Number of classes per episode')
    parser.add_argument('--n_shot', type=int, default=1,
                        help='Number of support samples per class (1 or 5)')
    parser.add_argument('--n_query', type=int, default=15,
                        help='Number of query samples per class')
    
    # Evaluation
    parser.add_argument('--episodes', type=int, default=600,
                        help='Number of evaluation episodes')
    
    # Misc
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save results JSON')
    
    return parser.parse_args()


def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    n_way: int,
    n_shot: int,
    device: torch.device
) -> List[Dict]:
    """
    Evaluate model on test episodes.
    
    Returns:
        List of episode results
    """
    model.eval()
    
    episode_results = []
    
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(dataloader, desc='Evaluating')):
            support_images = batch['support_images'].squeeze(0).to(device)
            support_labels = batch['support_labels'].squeeze(0).to(device)
            query_images = batch['query_images'].squeeze(0).to(device)
            query_labels = batch['query_labels'].squeeze(0).to(device)
            
            outputs = model(
                support_images, support_labels,
                query_images, n_way, n_shot
            )
            
            accuracy = compute_accuracy(outputs['logits'], query_labels)
            
            # Per-class accuracy
            predictions = outputs['logits'].argmax(dim=1)
            per_class_acc = []
            for cls in range(n_way):
                mask = query_labels == cls
                if mask.sum() > 0:
                    cls_acc = (predictions[mask] == cls).float().mean().item() * 100
                    per_class_acc.append(cls_acc)
            
            episode_results.append({
                'episode_id': idx,
                'accuracy': accuracy,
                'per_class_accuracy': per_class_acc
            })
    
    return episode_results


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Device
    device = get_device()
    
    # Setup logging
    logger = setup_logger('AMC-MetaNet-Test')
    
    # Load checkpoint
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    config = checkpoint.get('config', {})
    
    # Dataset name
    if args.dataset:
        dataset_name = args.dataset
    else:
        dataset_name = config.get('dataset', {}).get('name', 'NWPU-RESISC45')
    
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Evaluation: {args.n_way}-way {args.n_shot}-shot")
    logger.info(f"Episodes: {args.episodes}")
    
    # Create test dataset
    image_size = config.get('dataset', {}).get('image_size', 84)
    test_transform = get_test_transforms(image_size)
    
    test_dataset = RSDataset(
        root=os.path.join(args.data_root, dataset_name),
        split='test',
        transform=None
    )
    
    logger.info(f"Test classes: {len(test_dataset.classes)}")
    
    # Create dataloader
    episode_dataset = EpisodeDataset(
        base_dataset=test_dataset,
        n_way=args.n_way,
        n_shot=args.n_shot,
        n_query=args.n_query,
        n_episodes=args.episodes,
        transform=test_transform
    )
    
    test_loader = DataLoader(
        episode_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Create model
    model = AMCMetaNet(
        backbone=config.get('model', {}).get('backbone', 'resnet12'),
        feature_dim=config.get('model', {}).get('feature_dim', 640),
        attention_dim=config.get('metric', {}).get('attention_dim', 128),
        temperature=config.get('metric', {}).get('temperature', 1.0),
        use_attention=config.get('metric', {}).get('use_adaptive', True),
        use_task_conditioning=True,
        metric='euclidean',
        dropout=0.0  # No dropout during evaluation
    ).to(device)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info("Model loaded successfully")
    
    # Evaluate
    logger.info("Starting evaluation...")
    episode_results = evaluate(
        model, test_loader,
        args.n_way, args.n_shot, device
    )
    
    # Compute metrics
    metrics = compute_few_shot_metrics(episode_results)
    
    # Print results
    print("\n" + "=" * 60)
    print(f"Evaluation Results: {args.n_way}-way {args.n_shot}-shot")
    print("=" * 60)
    print(f"Mean Accuracy:  {metrics['mean_accuracy']:.2f}%")
    print(f"95% CI:         ± {metrics['ci_95']:.2f}%")
    print(f"Std:            {metrics['std_accuracy']:.2f}%")
    print(f"Min Accuracy:   {metrics['min_accuracy']:.2f}%")
    print(f"Max Accuracy:   {metrics['max_accuracy']:.2f}%")
    print(f"Episodes:       {metrics['n_episodes']}")
    print("=" * 60)
    
    # Save results
    if args.output:
        results = {
            'checkpoint': args.checkpoint,
            'dataset': dataset_name,
            'n_way': args.n_way,
            'n_shot': args.n_shot,
            'n_query': args.n_query,
            'n_episodes': args.episodes,
            'metrics': metrics,
            'episode_results': episode_results
        }
        
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to: {args.output}")
    
    return metrics


if __name__ == '__main__':
    main()
