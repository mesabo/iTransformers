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
from src.foundation.positional_encoding import PositionalEncoding

def test_positional_encoding():
    d_model = 64  # Model dimension
    seq_len = 10  # Sequence length
    batch_size = 2

    # Generate a random input tensor
    x = torch.rand(batch_size, seq_len, d_model)

    # Instantiate positional encoding
    pos_enc = PositionalEncoding(d_model)

    # Apply positional encoding
    output = pos_enc(x)

    print("Input Shape:", x.shape)  # Expected: (batch_size, seq_len, d_model)
    print("Output Shape:", output.shape)  # Expected: (batch_size, seq_len, d_model)

# if __name__ == "__main__":
#     test_positional_encoding()