import os
import hydra
import torch
import logging
import numpy as np
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
import wandb
from tqdm import tqdm

from models.multitask_model import MultiTaskModel
from loss.multitask_loss import MultiTaskLoss
from data.dataset import MultiTaskDataset
from utils.metrics import compute_metrics


def setup_logging(config: DictConfig) -> logging.Logger:
    """Setup logging configuration."""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # Create handlers
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler('training.log')
    
    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    c_handler.setFormatter(formatter)
    f_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)
    
    return logger


def setup_distributed(config: DictConfig) -> None:
    """Setup distributed training."""
    if config['hardware']['device'] == 'cuda':
        torch.cuda.set_device(config['local_rank'])
        torch.distributed.init_process_group(backend='nccl')
    else:
        torch.distributed.init_process_group(backend='gloo')


def get_optimizer(model: torch.nn.Module, config: DictConfig) -> torch.optim.Optimizer:
    """Get optimizer based on config."""
    if config['training']['optimizer']['name'] == 'adamw':
        return torch.optim.AdamW(
            model.parameters(),
            lr=config['training']['optimizer']['lr'],
            weight_decay=config['training']['optimizer']['weight_decay']
        )
    else:
        raise ValueError(f"Unknown optimizer: {config['training']['optimizer']['name']}")


def get_scheduler(optimizer: torch.optim.Optimizer, config: DictConfig) -> torch.optim.lr_scheduler._LRScheduler:
    """Get learning rate scheduler based on config."""
    if config['training']['scheduler']['name'] == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['training']['epochs'],
            eta_min=config['training']['scheduler']['min_lr']
        )
    else:
        raise ValueError(f"Unknown scheduler: {config['training']['scheduler']['name']}")


def train_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: MultiTaskLoss,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device,
    epoch: int,
    logger: logging.Logger,
    writer: SummaryWriter,
    config: DictConfig
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    # Create progress bar
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    
    for batch in pbar:
        # Move data to device
        images = batch['image'].to(device)
        targets = {
            k: v.to(device) for k, v in batch['targets'].items()
        }
        
        # Forward pass
        preds = model(images)
        
        # Get uncertainty weights
        uncertainty_weights = model.get_uncertainty_weights()
        
        # Compute loss
        loss, loss_dict = criterion(preds, targets, uncertainty_weights)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update progress bar
        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
        
        # Log to tensorboard
        if config['logging']['tensorboard']['enabled']:
            for k, v in loss_dict.items():
                writer.add_scalar(f'train/{k}', v, epoch)
    
    # Update learning rate
    scheduler.step()
    
    return total_loss / len(dataloader)


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: MultiTaskLoss,
    device: torch.device,
    epoch: int,
    logger: logging.Logger,
    writer: SummaryWriter,
    config: DictConfig
) -> float:
    """Validate the model."""
    model.eval()
    total_loss = 0
    all_metrics = {}
    
    # Create progress bar
    pbar = tqdm(dataloader, desc='Validation')
    
    for batch in pbar:
        # Move data to device
        images = batch['image'].to(device)
        targets = {
            k: v.to(device) for k, v in batch['targets'].items()
        }
        
        # Forward pass
        preds = model(images)
        
        # Get uncertainty weights
        uncertainty_weights = model.get_uncertainty_weights()
        
        # Compute loss
        loss, loss_dict = criterion(preds, targets, uncertainty_weights)
        
        # Compute metrics
        metrics = compute_metrics(preds, targets)
        
        # Update progress bar
        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
        
        # Accumulate metrics
        for k, v in metrics.items():
            if k not in all_metrics:
                all_metrics[k] = []
            all_metrics[k].append(v)
    
    # Average metrics
    avg_metrics = {
        k: np.mean(v) for k, v in all_metrics.items()
    }
    
    # Log to tensorboard
    if config['logging']['tensorboard']['enabled']:
        for k, v in avg_metrics.items():
            writer.add_scalar(f'val/{k}', v, epoch)
    
    return total_loss / len(dataloader), avg_metrics


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    epoch: int,
    metrics: dict,
    config: DictConfig
) -> None:
    """Save model checkpoint."""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch,
        'metrics': metrics
    }
    
    # Create checkpoint directory if it doesn't exist
    os.makedirs(config['logging']['checkpoint']['save_dir'], exist_ok=True)
    
    # Save checkpoint
    torch.save(
        checkpoint,
        os.path.join(
            config['logging']['checkpoint']['save_dir'],
            f'checkpoint_epoch_{epoch}.pth'
        )
    )
    
    # Remove old checkpoints if needed
    if config['logging']['checkpoint']['keep_last'] > 0:
        checkpoints = sorted([
            f for f in os.listdir(config['logging']['checkpoint']['save_dir'])
            if f.startswith('checkpoint_epoch_')
        ])
        
        if len(checkpoints) > config['logging']['checkpoint']['keep_last']:
            for checkpoint in checkpoints[:-config['logging']['checkpoint']['keep_last']]:
                os.remove(os.path.join(
                    config['logging']['checkpoint']['save_dir'],
                    checkpoint
                ))


@hydra.main(config_path="configs", config_name="config.yaml")
def main(config: DictConfig) -> None:
    """Main training function."""
    # Create necessary directories
    os.makedirs(config['logging']['checkpoint']['save_dir'], exist_ok=True)
    os.makedirs(config['deployment']['export_dir'], exist_ok=True)  # Create export directory
    
    # Setup logging
    logger = setup_logging(config)
    logger.info("Starting training...")
    
    # Setup distributed training
    if config['hardware']['device'] == 'cuda':
        setup_distributed(config)
    
    # Setup device
    device = torch.device(config['hardware']['device'])
    
    # Create model
    model = MultiTaskModel(config).to(device)
    
    # Wrap model for distributed training
    if config['hardware']['device'] == 'cuda':
        model = DistributedDataParallel(
            model,
            device_ids=[config['local_rank']]
        )
    
    # Create criterion
    criterion = MultiTaskLoss(config).to(device)
    
    # Create optimizer and scheduler
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)
    
    # Create datasets and dataloaders
    train_dataset = MultiTaskDataset(config, split='train')
    val_dataset = MultiTaskDataset(config, split='val')
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['hardware']['num_workers'],
        pin_memory=config['hardware']['pin_memory']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['hardware']['num_workers'],
        pin_memory=config['hardware']['pin_memory']
    )
    
    # Setup tensorboard
    if config['logging']['tensorboard']['enabled']:
        writer = SummaryWriter(config['logging']['tensorboard']['log_dir'])
    else:
        writer = None
    
    # Setup wandb
    if config['logging']['wandb']['enabled']:
        wandb.init(
            project=config['logging']['wandb']['project'],
            entity=config['logging']['wandb']['entity'],
            config=OmegaConf.to_container(config, resolve=True)
        )
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(config['training']['epochs']):
        # Train
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer,
            scheduler, device, epoch, logger, writer, config
        )
        
        # Validate
        val_loss, metrics = validate(
            model, val_loader, criterion, device,
            epoch, logger, writer, config
        )
        
        # Log metrics
        logger.info(
            f"Epoch {epoch}: "
            f"train_loss={train_loss:.4f}, "
            f"val_loss={val_loss:.4f}, "
            f"metrics={metrics}"
        )
        
        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer, scheduler,
                epoch, metrics, config
            )
        
        # Log to wandb
        if config['logging']['wandb']['enabled']:
            wandb.log({
                'train_loss': train_loss,
                'val_loss': val_loss,
                **metrics
            })
    
    # Export model
    if config['deployment']['onnx']['enabled']:
        model.eval()
        model.export_onnx(
            os.path.join(
                config['deployment']['export_dir'],
                'model.onnx'
            )
        )
    
    logger.info("Training finished!")


if __name__ == "__main__":
    main()
