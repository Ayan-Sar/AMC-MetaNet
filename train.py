"""
Training script for AMC-MetaNet.

Implements episodic meta-learning for few-shot remote sensing classification.

Usage:
    python train.py --config configs/config.yaml
    python train.py --dataset NWPU-RESISC45 --n_way 5 --n_shot 5
"""

import argparse
import os
import time
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import RSDataset, EpisodeDataset, get_train_transforms, get_test_transforms
from models import AMCMetaNet, PrototypicalLoss, BalancedLoss
from utils import (
    set_seed, load_config, save_checkpoint, get_device,
    compute_accuracy, compute_confidence_interval, AverageMeter,
    setup_logger, TensorBoardLogger, get_lr_scheduler, EarlyStopping,
    count_parameters
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train AMC-MetaNet')
    
    # Config
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')
    
    # Dataset
    parser.add_argument('--dataset', type=str, default=None,
                        help='Dataset name (overrides config)')
    parser.add_argument('--data_root', type=str, default='./data',
                        help='Data root directory')
    
    # Few-shot settings
    parser.add_argument('--n_way', type=int, default=None,
                        help='Number of classes per episode')
    parser.add_argument('--n_shot', type=int, default=None,
                        help='Number of support samples per class')
    parser.add_argument('--n_query', type=int, default=15,
                        help='Number of query samples per class')
    
    # Training
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs')
    parser.add_argument('--episodes', type=int, default=100,
                        help='Episodes per epoch')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size (number of episodes)')
    
    # Model
    parser.add_argument('--backbone', type=str, default='resnet12',
                        choices=['resnet12', 'conv4'],
                        help='Backbone architecture')
    
    # Misc
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device id')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='Checkpoint save directory')
    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='Log directory')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    parser.add_argument('--dry_run', action='store_true',
                        help='Run a few iterations for testing')
    
    return parser.parse_args()


def create_episode_dataloader(
    dataset: RSDataset,
    n_way: int,
    n_shot: int,
    n_query: int,
    n_episodes: int,
    transform,
    num_workers: int = 4
) -> DataLoader:
    """Create dataloader for episodic training."""
    episode_dataset = EpisodeDataset(
        base_dataset=dataset,
        n_way=n_way,
        n_shot=n_shot,
        n_query=n_query,
        n_episodes=n_episodes,
        transform=transform
    )
    
    return DataLoader(
        episode_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    n_way: int,
    n_shot: int,
    device: torch.device,
    epoch: int,
    logger
) -> Tuple[float, float]:
    """
    Train for one epoch.
    
    Returns:
        Tuple of (average_loss, average_accuracy)
    """
    model.train()
    
    loss_meter = AverageMeter('Loss')
    acc_meter = AverageMeter('Accuracy')
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}', leave=False)
    
    for batch in pbar:
        # Unpack episode
        support_images = batch['support_images'].squeeze(0).to(device)
        support_labels = batch['support_labels'].squeeze(0).to(device)
        query_images = batch['query_images'].squeeze(0).to(device)
        query_labels = batch['query_labels'].squeeze(0).to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(
            support_images, support_labels,
            query_images, n_way, n_shot
        )
        
        # Compute loss
        if isinstance(criterion, BalancedLoss):
            loss_dict = criterion(
                outputs['logits'], query_labels,
                outputs['prototypes'],
                outputs['support_features'], support_labels
            )
            loss = loss_dict['loss']
        else:
            loss = criterion(outputs['logits'], query_labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Compute accuracy
        accuracy = compute_accuracy(outputs['logits'], query_labels)
        
        # Update meters
        loss_meter.update(loss.item())
        acc_meter.update(accuracy)
        
        # Update progress bar
        pbar.set_postfix({
            'Loss': f'{loss_meter.avg:.4f}',
            'Acc': f'{acc_meter.avg:.2f}%'
        })
    
    return loss_meter.avg, acc_meter.avg


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    n_way: int,
    n_shot: int,
    device: torch.device
) -> Tuple[float, float]:
    """
    Validate model on validation set.
    
    Returns:
        Tuple of (mean_accuracy, confidence_interval)
    """
    model.eval()
    
    accuracies = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validation', leave=False):
            support_images = batch['support_images'].squeeze(0).to(device)
            support_labels = batch['support_labels'].squeeze(0).to(device)
            query_images = batch['query_images'].squeeze(0).to(device)
            query_labels = batch['query_labels'].squeeze(0).to(device)
            
            outputs = model(
                support_images, support_labels,
                query_images, n_way, n_shot
            )
            
            accuracy = compute_accuracy(outputs['logits'], query_labels)
            accuracies.append(accuracy)
    
    mean_acc, ci = compute_confidence_interval(accuracies)
    return mean_acc, ci


def main():
    """Main training function."""
    args = parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.dataset:
        config['dataset']['name'] = args.dataset
    if args.n_way:
        config['few_shot']['train_way'] = args.n_way
        config['few_shot']['test_way'] = args.n_way
    if args.n_shot:
        config['few_shot']['train_shot'] = args.n_shot
        config['few_shot']['test_shot'] = args.n_shot
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.lr:
        config['training']['learning_rate'] = args.lr
    
    # Set seed
    set_seed(config.get('seed', args.seed))
    
    # Device
    device = get_device()
    
    # Setup logging
    logger = setup_logger('AMC-MetaNet', args.log_dir)
    tb_logger = TensorBoardLogger(
        os.path.join(args.log_dir, 'tensorboard'),
        enabled=config['logging'].get('tensorboard', True)
    )
    
    logger.info(f"Configuration: {config}")
    
    # Create datasets
    image_size = config['dataset'].get('image_size', 84)
    train_transform = get_train_transforms(image_size)
    test_transform = get_test_transforms(image_size)
    
    train_dataset = RSDataset(
        root=os.path.join(args.data_root, config['dataset']['name']),
        split='train',
        transform=None  # Transform applied in EpisodeDataset
    )
    
    val_dataset = RSDataset(
        root=os.path.join(args.data_root, config['dataset']['name']),
        split='val',
        transform=None
    )
    
    # Few-shot settings
    n_way = config['few_shot']['train_way']
    n_shot = config['few_shot']['train_shot']
    n_query = config['few_shot']['train_query']
    
    logger.info(f"Training: {n_way}-way {n_shot}-shot")
    logger.info(f"Train classes: {len(train_dataset.classes)}")
    logger.info(f"Val classes: {len(val_dataset.classes)}")
    
    # Create dataloaders
    episodes_per_epoch = config['training'].get('episodes_per_epoch', args.episodes)
    
    if args.dry_run:
        episodes_per_epoch = 5
    
    train_loader = create_episode_dataloader(
        train_dataset, n_way, n_shot, n_query,
        n_episodes=episodes_per_epoch,
        transform=train_transform,
        num_workers=config['dataset'].get('num_workers', 4)
    )
    
    val_episodes = config['evaluation'].get('num_episodes', 100)
    if args.dry_run:
        val_episodes = 10
    
    val_loader = create_episode_dataloader(
        val_dataset, n_way, n_shot, n_query,
        n_episodes=val_episodes,
        transform=test_transform,
        num_workers=config['dataset'].get('num_workers', 4)
    )
    
    # Create model
    model = AMCMetaNet(
        backbone=config['model'].get('backbone', args.backbone),
        feature_dim=config['model'].get('feature_dim', 640),
        attention_dim=config['metric'].get('attention_dim', 128),
        temperature=config['metric'].get('temperature', 1.0),
        use_attention=config['metric'].get('use_adaptive', True),
        use_task_conditioning=True,
        metric='euclidean',
        dropout=config['model'].get('dropout', 0.5)
    ).to(device)
    
    logger.info(f"Model parameters: {count_parameters(model):,}")
    
    # Loss and optimizer
    criterion = BalancedLoss(
        classification_weight=1.0,
        diversity_weight=0.1,
        alignment_weight=0.1
    )
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['training'].get('learning_rate', 0.001),
        weight_decay=config['training'].get('weight_decay', 0.0005)
    )
    
    # Learning rate scheduler
    scheduler = get_lr_scheduler(
        optimizer,
        scheduler_type=config['training'].get('lr_scheduler', 'step'),
        step_size=config['training'].get('lr_step_size', 20),
        gamma=config['training'].get('lr_gamma', 0.5)
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=15, mode='max')
    
    # Resume from checkpoint
    start_epoch = 1
    best_acc = 0.0
    
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint.get('best_acc', 0.0)
        logger.info(f"Resumed from epoch {start_epoch - 1}")
    
    # Training loop
    epochs = config['training']['epochs']
    if args.dry_run:
        epochs = 2
    
    logger.info("Starting training...")
    
    for epoch in range(start_epoch, epochs + 1):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion,
            n_way, n_shot, device, epoch, logger
        )
        
        # Validate
        val_acc, val_ci = validate(model, val_loader, n_way, n_shot, device)
        
        # Update scheduler
        scheduler.step()
        
        # Log
        logger.info(
            f"Epoch {epoch}/{epochs} - "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
            f"Val Acc: {val_acc:.2f}% ± {val_ci:.2f}%"
        )
        
        tb_logger.log_scalar('train/loss', train_loss, epoch)
        tb_logger.log_scalar('train/accuracy', train_acc, epoch)
        tb_logger.log_scalar('val/accuracy', val_acc, epoch)
        tb_logger.log_scalar('val/ci_95', val_ci, epoch)
        tb_logger.log_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
        
        # Save checkpoint
        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
            logger.info(f"New best accuracy: {best_acc:.2f}%")
        
        if epoch % config['logging'].get('save_freq', 10) == 0 or is_best:
            save_checkpoint(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_acc': best_acc,
                    'config': config
                },
                args.save_dir,
                filename=f'checkpoint_epoch{epoch}.pth',
                is_best=is_best
            )
        
        # Early stopping
        if early_stopping(val_acc):
            logger.info(f"Early stopping triggered at epoch {epoch}")
            break
    
    logger.info(f"Training completed. Best accuracy: {best_acc:.2f}%")
    tb_logger.close()


if __name__ == '__main__':
    main()
