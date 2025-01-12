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
from src.foundation.multi_head_attention import MultiHeadAttention

def test_multi_head_attention():
    d_model = 64  # Model dimension
    num_heads = 8  # Number of attention heads
    batch_size = 2
    seq_len = 5

    # Generate random Q, K, V matrices
    Q = torch.rand(batch_size, seq_len, d_model)
    K = torch.rand(batch_size, seq_len, d_model)
    V = torch.rand(batch_size, seq_len, d_model)

    # Instantiate multi-head attention layer
    mha = MultiHeadAttention(d_model, num_heads)

    # Forward pass
    output, attn_weights = mha(Q, K, V)

    print("Output Shape:", output.shape)  # Expected: (batch_size, seq_len, d_model)
    print("Attention Weights Shape:", attn_weights.shape)  # Expected: (batch_size, num_heads, seq_len, seq_len)

# if __name__ == "__main__":
#     test_multi_head_attention()