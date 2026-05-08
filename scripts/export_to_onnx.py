"""
Export the trained AuresNet-DZ model to ONNX format.

This script loads the trained PyTorch Lightning checkpoint and exports the 
underlying model to ONNX format for deployment and inference on various platforms.

Usage:
    python scripts/export_to_onnx.py --checkpoint checkpoints/last.ckpt --output models/auresnet_dz.onnx
"""

import argparse
from pathlib import Path

import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf

from auresnet_dz.models.unet_smp import build_unet
from auresnet_dz.train.lightning_module import GfsToEra5LightningModule


def load_checkpoint(checkpoint_path: str) -> GfsToEra5LightningModule:
    """Load a PyTorch Lightning checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Extract hyperparameters
    hparams = checkpoint["hyper_parameters"]
    
    # Recreate the model
    model = build_unet(
        in_channels=hparams["in_channels"],
        out_channels=hparams["out_channels"],
        encoder_name=hparams.get("encoder_name", "resnet50"),
        encoder_weights=hparams.get("encoder_weights"),
        architecture=hparams.get("architecture", "unet"),
    )
    
    # Create Lightning module
    lightning_module = GfsToEra5LightningModule(
        model=model,
        learning_rate=hparams.get("learning_rate", 3e-4),
        weight_decay=hparams.get("weight_decay", 1e-5),
        target_variable_names=hparams.get("target_variable_names"),
        target_mean=checkpoint.get("state_dict", {}).get("_target_mean"),
        target_std=checkpoint.get("state_dict", {}).get("_target_std"),
    )
    
    # Load the checkpoint state
    lightning_module.load_state_dict(checkpoint["state_dict"], strict=False)
    lightning_module.eval()
    
    return lightning_module


def export_to_onnx(
    checkpoint_path: str,
    output_path: str,
    input_shape: tuple = (1, 5, 128, 128),
    opset_version: int = 14,
    use_external_data_format: bool = False,
) -> None:
    """
    Export the model to ONNX format.
    
    Args:
        checkpoint_path: Path to the PyTorch Lightning checkpoint
        output_path: Path where the ONNX model will be saved
        input_shape: Input tensor shape (batch, channels, height, width)
        opset_version: ONNX opset version to use
        use_external_data_format: Whether to use external data format for large models
    """
    print(f"Loading checkpoint from: {checkpoint_path}")
    lightning_module = load_checkpoint(checkpoint_path)
    
    # Create output directory if it doesn't exist
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Create dummy input
    dummy_input = torch.randn(*input_shape)
    
    print(f"Exporting model with input shape: {input_shape}")
    print(f"Output ONNX file: {output_path}")
    
    # Export to ONNX
    torch.onnx.export(
        lightning_module,
        dummy_input,
        str(output_file),
        input_names=["gfs_input"],
        output_names=["era5_output"],
        opset_version=opset_version,
        do_constant_folding=True,
        verbose=False,
        dynamic_axes={
            "gfs_input": {0: "batch_size", 2: "height", 3: "width"},
            "era5_output": {0: "batch_size", 2: "height", 3: "width"},
        },
        use_external_data_format=use_external_data_format,
    )
    
    print(f"✓ Successfully exported model to: {output_path}")
    print(f"  Input shape: {input_shape}")
    print(f"  Input channels: 5 (GFS variables)")
    print(f"  Output channels: 5 (ERA5 variables)")
    print(f"  ONNX opset version: {opset_version}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export AuresNet-DZ model to ONNX format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export the latest checkpoint with default settings
  python scripts/export_to_onnx.py --checkpoint checkpoints/last.ckpt

  # Export with custom output path
  python scripts/export_to_onnx.py --checkpoint checkpoints/epoch=99-step=1700.ckpt --output models/auresnet_dz_v99.onnx

  # Export with larger input shape for higher resolution inference
  python scripts/export_to_onnx.py --checkpoint checkpoints/last.ckpt --input-shape 1 5 256 256
        """,
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/last.ckpt",
        help="Path to the PyTorch Lightning checkpoint (default: checkpoints/last.ckpt)",
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="models/auresnet_dz.onnx",
        help="Path where the ONNX model will be saved (default: models/auresnet_dz.onnx)",
    )
    
    parser.add_argument(
        "--input-shape",
        type=int,
        nargs=4,
        default=[1, 5, 128, 128],
        metavar=("BATCH", "CHANNELS", "HEIGHT", "WIDTH"),
        help="Input tensor shape for ONNX export (default: 1 5 128 128)",
    )
    
    parser.add_argument(
        "--opset-version",
        type=int,
        default=14,
        help="ONNX opset version (default: 14)",
    )
    
    parser.add_argument(
        "--external-data-format",
        action="store_true",
        help="Use external data format for large models (>2GB)",
    )
    
    args = parser.parse_args()
    
    try:
        export_to_onnx(
            checkpoint_path=args.checkpoint,
            output_path=args.output,
            input_shape=tuple(args.input_shape),
            opset_version=args.opset_version,
            use_external_data_format=args.external_data_format,
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"Make sure the checkpoint exists at: {args.checkpoint}")
        exit(1)
    except Exception as e:
        print(f"Error during export: {e}")
        exit(1)
