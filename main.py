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

# Add the src directory to PYTHONPATH
PROJECT_ROOT = Path(__file__).parent / "src"
sys.path.append(str(PROJECT_ROOT))

from src.utils.argument_parser import get_arguments
from src.utils.device_utils import setup_device
from src.examples.test_self_attention import test_scaled_dot_product_attention
from src.examples.test_multi_head_attention import test_multi_head_attention
from src.examples.test_positional_encoding import test_positional_encoding
from src.examples.test_transformer_encoder import test_transformer_encoder
from src.examples.test_transformer_encoder_stack import test_transformer_encoder_stack
from src.examples.test_data_preparation import test_data_preparation
from src.examples.test_train_transformer_encoder import test_train_transformer_encoder

# Map test cases to functions
TEST_CASES = {
    "self_attention": test_scaled_dot_product_attention,
    "multi_head_attention": test_multi_head_attention,
    "positional_encoding": test_positional_encoding,
    "transformer_encoder": test_transformer_encoder,
    "transformer_encoder_stack": test_transformer_encoder_stack,
    "data_preparation": test_data_preparation,
    "train_transformer_encoder": test_train_transformer_encoder,
}

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

    # Execute the selected test case
    if args.test_case in TEST_CASES:
        print(f"{10 * ''} Running {args.test_case} {10 * ''}")
        TEST_CASES[args.test_case]()
    else:
        print(f"Invalid test case: {args.test_case}")
        print(f"Available test cases: {list(TEST_CASES.keys())}")

if __name__ == "__main__":
    main()