#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 01/12/2025
🚀 Welcome to the Awesome Python Script 🚀

User: messou
Email: mesabo18@gmail.com / messouaboya17@gmail.com
Github: https://github.com/mesabo
Univ: Hosei University
Dept: Science and Engineering
Lab: Prof YU Keping's Lab
"""

"""
Utility to setup computation device for iTransformers
"""
import logging

import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_device():
    """
    Determines the best computation device available: CUDA, MPS, or CPU.

    Returns:
        torch.device: Selected device.
    """
    # Check if CUDA is available
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(device.index)}")

    # Check if MPS (Apple Silicon) is available
    elif torch.backends.mps.is_available():
        try:
            # Test if MPS device can be used by performing a simple operation
            device = torch.device('mps')
            logger.info("Using MPS device for computation.")
        except RuntimeError as e:
            logger.warning("MPS backend is available but the device could not be used.")
            logger.warning(f"Error: {e}")
            device = torch.device('cpu')  # Fallback to CPU if MPS fails

    # Default to CPU if neither CUDA nor MPS are available
    else:
        device = torch.device('cpu')
        logger.info("Using CPU device for computation.")

    return device
