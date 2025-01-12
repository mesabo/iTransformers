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
from foundation.transformer_encoder import TransformerEncoderBlock

def test_transformer_encoder():
    d_model = 64  # Model dimension
    num_heads = 8  # Number of attention heads
    ff_hidden_dim = 128  # Hidden dimension of feedforward network
    seq_len = 10  # Sequence length
    batch_size = 2

    # Generate random input tensor
    x = torch.rand(batch_size, seq_len, d_model)

    # Instantiate the Transformer Encoder Block
    encoder_block = TransformerEncoderBlock(d_model, num_heads, ff_hidden_dim)

    # Forward pass
    output = encoder_block(x)

    print("Input Shape:", x.shape)  # Expected: (batch_size, seq_len, d_model)
    print("Output Shape:", output.shape)  # Expected: (batch_size, seq_len, d_model)

# if __name__ == "__main__":
#     test_transformer_encoder()