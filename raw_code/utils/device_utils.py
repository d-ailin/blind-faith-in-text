"""
Utility functions for device detection and management.
Supports CUDA (NVIDIA GPU) and CPU only.
MPS (Apple Silicon) is intentionally not used due to compatibility issues.

Environment Variables:
    FORCE_CPU: Set to '1' or 'true' to force CPU usage
"""

import torch
import os


def get_optimal_device():
    """
    Automatically detect the best available device.
    Only uses CUDA or CPU. MPS is intentionally skipped due to compatibility issues.

    Environment variables can override the detection:
    - FORCE_CPU=1: Force CPU usage

    Returns:
        str: Device string ('cuda' or 'cpu')
    """
    # Check for forced CPU
    if os.getenv('FORCE_CPU', '').lower() in ('1', 'true', 'yes'):
        print("FORCE_CPU is set. Using CPU.")
        return "cpu"

    # Check for CUDA
    if torch.cuda.is_available():
        print("CUDA available. Using CUDA.")
        return "cuda"

    # Default to CPU (MPS intentionally not used)
    print("CUDA not available. Using CPU.")
    return "cpu"


def get_device_info():
    """
    Get information about the current device setup.

    Returns:
        dict: Device information
    """
    device = get_optimal_device()
    info = {
        "device": device,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
        "pytorch_version": torch.__version__,
        "env_force_cpu": os.getenv('FORCE_CPU', 'not set'),
        "env_disable_mps": os.getenv('DISABLE_MPS', 'not set'),
        "env_mps_fallback": os.getenv('PYTORCH_ENABLE_MPS_FALLBACK', 'not set'),
    }

    if device == "cuda":
        info["cuda_device_count"] = torch.cuda.device_count()
        info["cuda_device_name"] = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else None

    return info


# For backward compatibility
DEVICE = get_optimal_device()


if __name__ == "__main__":
    import json
    info = get_device_info()
    print("Device Configuration:")
    print(json.dumps(info, indent=2))
    print("\nUsage:")
    print("  Force CPU: export FORCE_CPU=1")
    print("\nNote: MPS (Apple Silicon GPU) is intentionally disabled due to compatibility issues.")
