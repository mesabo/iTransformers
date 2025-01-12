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

import torch
from src.foundation.transformer_encoder_stack import TransformerEncoder

def test_transformer_encoder_stack():
    input_dim = 4  # Iris dataset has 4 features
    d_model = 64   # Model dimension
    num_heads = 8  # Number of attention heads
    ff_hidden_dim = 128  # Hidden dimension of feedforward network
    num_layers = 3  # Number of encoder blocks
    seq_len = 10    # Sequence length
    batch_size = 2

    # Generate random input tensor
    x = torch.rand(batch_size, seq_len, input_dim)

    # Instantiate the Transformer Encoder
    encoder = TransformerEncoder(input_dim, d_model, num_heads, ff_hidden_dim, num_layers)

    # Forward pass
    output = encoder(x)

    print("Input Shape:", x.shape)  # Expected: (batch_size, seq_len, input_dim)
    print("Output Shape:", output.shape)  # Expected: (batch_size, seq_len, d_model)

# if __name__ == "__main__":
#     test_transformer_encoder_stack()