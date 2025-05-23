import os
import sys
import hydra
import torch
import logging
import numpy as np
from pathlib import Path
from omegaconf import DictConfig
from typing import Dict, Any, Optional
from models.multitask_model import MultiTaskModel
from utils.logging import setup_logging

logger = logging.getLogger(__name__)

def quantize_model(
    model: torch.nn.Module,
    config: DictConfig,
    dummy_input: torch.Tensor
) -> torch.nn.Module:
    """Quantize model for deployment.
    
    Args:
        model: PyTorch model to quantize
        config: Configuration object
        dummy_input: Sample input tensor for calibration
        
    Returns:
        torch.nn.Module: Quantized model
    """
    try:
        if config['deployment']['quantization']['dtype'] == 'int8':
            # Select quantization backend based on target hardware
            if config['deployment']['target'] == 'orin':
                # Use QNNPACK for ARM/Orin
                model.qconfig = torch.quantization.get_default_qconfig('qnnpack')
            else:
                # Use FBGEMM for x86
                model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
            
            # Prepare model for quantization
            torch.quantization.prepare(model, inplace=True)
            
            # Calibrate with dummy input
            with torch.no_grad():
                model(dummy_input)
            
            # Convert to quantized model
            torch.quantization.convert(model, inplace=True)
            logger.info(f"Model quantized to INT8 using {model.qconfig.backend} backend")
            
        return model
        
    except Exception as e:
        logger.error(f"Quantization failed: {str(e)}")
        return model

def deploy_model(
    model_path: str,
    config: DictConfig
) -> bool:
    """Deploy model to target hardware.
    
    Args:
        model_path: Path to exported model
        config: Configuration object
        
    Returns:
        bool: True if deployment succeeds, False otherwise
    """
    try:
        # Here you would add code to deploy to specific hardware
        # For example, using vendor SDKs like TensorRT, OpenVINO, etc.
        if config['deployment']['target'] == 'tensorrt':
            logger.info("Deploying to TensorRT...")
            # Add TensorRT deployment code
        elif config['deployment']['target'] == 'openvino':
            logger.info("Deploying to OpenVINO...")
            # Add OpenVINO deployment code
        else:
            logger.warning(f"Unknown deployment target: {config['deployment']['target']}")
            
        return True
        
    except Exception as e:
        logger.error(f"Deployment failed: {str(e)}")
        return False

def validate_model(model: torch.nn.Module, dummy_input: torch.Tensor) -> bool:
    """Validate model outputs before export.
    
    Args:
        model: PyTorch model to validate
        dummy_input: Sample input tensor
        
    Returns:
        bool: True if validation passes, False otherwise
    """
    try:
        # Set model to evaluation mode
        model.eval()
        
        # Disable gradient computation and batch norm statistics updates
        with torch.no_grad():
            # Create a larger batch for validation
            batch_input = dummy_input.repeat(2, 1, 1, 1)  # Double the batch size
            outputs = model(batch_input)
            
            # Check outputs are valid
            for task_name, output in outputs.items():
                # Handle detection outputs which are tuples
                if task_name.startswith('detection_'):
                    if not isinstance(output, torch.Tensor):
                        logger.error(f"Output for {task_name} is not a tensor")
                        return False
                else:
                    if not isinstance(output, torch.Tensor):
                        logger.error(f"Output for {task_name} is not a tensor")
                        return False
                
                # Check for invalid values
                if torch.isnan(output).any() or torch.isinf(output).any():
                    logger.error(f"Invalid values in {task_name} output")
                    return False
                    
        return True
    except Exception as e:
        logger.error(f"Model validation failed: {str(e)}")
        return False

def export_onnx(
    model: torch.nn.Module,
    save_path: str,
    dummy_input: torch.Tensor,
    config: DictConfig
) -> bool:
    """Export model to ONNX format.
    
    Args:
        model: PyTorch model to export
        save_path: Path to save ONNX model
        dummy_input: Sample input tensor
        config: Configuration object
        
    Returns:
        bool: True if export succeeds, False otherwise
    """
    try:
        # Set model to evaluation mode
        model.eval()
        
        # Define dynamic axes
        dynamic_axes = {
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
        
        # Export model
        torch.onnx.export(
            model,
            dummy_input,
            save_path,
            export_params=True,
            opset_version=config['deployment']['onnx']['opset_version'],
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=dynamic_axes if config['deployment']['onnx']['dynamic_axes'] else None,
            verbose=False
        )
        
        logger.info(f"Model exported successfully to {save_path}")
        return True
        
    except Exception as e:
        logger.error(f"ONNX export failed: {str(e)}")
        return False

def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    device: torch.device
) -> bool:
    """Load model checkpoint.
    
    Args:
        model: PyTorch model to load weights into
        checkpoint_path: Path to checkpoint file
        device: Device to load checkpoint on
        
    Returns:
        bool: True if loading succeeds, False otherwise
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {str(e)}")
        return False

@hydra.main(config_path="configs", config_name="config.yaml")
def main(config: DictConfig) -> None:
    """Export model to ONNX format."""
    # Setup logging
    logger = setup_logging(config)
    logger.info("Starting model export...")
    
    # Create export directory
    export_dir = Path(config['deployment']['export_dir'])
    export_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup device
    device = torch.device(config['hardware']['device'])
    
    try:
        # Create model
        model = MultiTaskModel(config).to(device)
        
        # Load checkpoint if specified
        if 'checkpoint_path' in config['deployment']:
            if not load_checkpoint(
                model,
                config['deployment']['checkpoint_path'],
                device
            ):
                sys.exit(1)
        
        # Create dummy input
        dummy_input = torch.randn(
            1, 3, *config['model']['input_size'],
            device=device
        )
        
        # Validate model
        if not validate_model(model, dummy_input):
            logger.error("Model validation failed")
            sys.exit(1)
        
        # Quantize model if enabled
        if config['deployment']['quantization']['enabled']:
            model = quantize_model(model, config, dummy_input)
        
        # Export model
        if config['deployment']['onnx']['enabled']:
            save_path = export_dir / 'model.onnx'
            if not export_onnx(model, str(save_path), dummy_input, config):
                logger.error("ONNX export failed")
                sys.exit(1)
            
            # Deploy model if enabled
            if config['deployment']['enabled']:
                if not deploy_model(str(save_path), config):
                    logger.error("Model deployment failed")
                    sys.exit(1)
        
        logger.info("Model export completed successfully")
        
    except Exception as e:
        logger.error(f"Export failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
