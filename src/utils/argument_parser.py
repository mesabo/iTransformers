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

import argparse

def get_arguments():
    """
    Parses command-line arguments for the iTransformers project.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="iTransformers: Advanced Transformer-Based Learning"
    )

    # General settings
    parser.add_argument(
        "--task",
        type=str,
        choices=["seq2seq", "classification", "time_series"],
        default="seq2seq",
        help="Select the task type: seq2seq, classification, or time_series",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["transformer", "fine_tune", "custom"],
        default="transformer",
        help="Choose the model type: transformer, fine_tune, or custom",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="./data",
        help="Path to the dataset folder",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="./results",
        help="Path to save model outputs and results",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        choices=["INFO", "DEBUG", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )

    # Model hyperparameters
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate for the optimizer",
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=8,
        help="Number of attention heads in the Transformer",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=512,
        help="Hidden dimension size for the Transformer model",
    )

    # Hardware options
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "mps"],
        default="cpu",
        help="Device to run the training: cpu, cuda (GPU), or mps (Apple M1/M2)",
    )

    # Add this line to the parser
    parser.add_argument(
        "--test_case",
        type=str,
        choices=["self_attention", "multi_head_attention", "positional_encoding",
                 "transformer_encoder", "transformer_encoder_stack", "data_preparation", ],
        #required=True,
        default="self_attention",
        help="Select the test case to run.",
    )

    return parser.parse_args()