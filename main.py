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

import sys
from pathlib import Path

# Add the project root to PYTHONPATH
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT))

from utils.argument_parser import get_arguments
from utils.device_utils import setup_device
from examples.test_self_attention import test_scaled_dot_product_attention
from examples.test_multi_head_attention import test_multi_head_attention

def main():
    # Parse arguments
    args = get_arguments()

    # Setup device
    device = setup_device()

    print(f"Task: {args.task}")
    print(f"Model: {args.model}")
    print(f"Data Path: {args.data_path}")
    print(f"Save Path: {args.save_path}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Device: {device}")

    #print(f"{10*''} Test Scaled dot product via self_attention {10*''}")
    #test_scaled_dot_product_attention()

    print(f"{10*''} Test Multihead Attention {10*''}")
    test_multi_head_attention()

if __name__ == "__main__":
    main()